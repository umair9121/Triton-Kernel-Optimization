"""Serialize an :class:`~tritonlens.analyze.Analysis` into the JSON consumed by the viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import __version__
from ..analysis.banks import SharedAccess
from ..analysis.coalescing import base_pointer_arg
from ..analysis.lint import Finding, run_lint
from ..analysis.tiling import _main_store, program_tiling
from ..analyze import Analysis
from ..ir import MemDescType, Op, TensorType
from ..layouts import DistributedLayout, SharedLayout, UnsupportedLayout, from_encoding
from ..layouts.shared import bank_of

MAX_ELEMENTS = 1 << 16


def _tensor_key(enc_key: str, shape: list[int]) -> str:
    return f"{enc_key}|{'x'.join(map(str, shape))}"


class _TensorTable:
    """Deduplicated ownership maps for every (layout, shape) pair worth drawing."""

    def __init__(self, an: Analysis):
        self.an = an
        self.items: dict[str, dict] = {}

    def add(self, val, role: str, op: Op) -> str | None:
        t = val.type if val is not None else None
        if not isinstance(t, TensorType) or t.encoding is None or not t.shape:
            return None
        if len(t.shape) > 2 or t.numel > MAX_ELEMENTS:
            return None
        key = _tensor_key(t.encoding.key(), t.shape)
        if key not in self.items:
            try:
                layout = from_encoding(t.encoding, self.an.module.num_warps,
                                       self.an.module.threads_per_warp)
            except (UnsupportedLayout, KeyError):
                return None
            if not isinstance(layout, DistributedLayout):
                return None
            own = layout.ownership(t.shape)
            n = t.numel
            warp, lane, reg, repl = [0] * n, [0] * n, [0] * n, [0] * n
            for i, owners in enumerate(own.owners):
                if owners:
                    warp[i], lane[i], reg[i] = owners[0]
                repl[i] = len(owners)
            self.items[key] = {
                "key": key,
                "shape": list(t.shape),
                "elem": t.element_name,
                "encoding": t.encoding.name,
                "layout": layout.describe(),
                "regsPerThread": own.regs_per_thread,
                "replication": own.replication,
                "warp": warp,
                "lane": lane,
                "reg": reg,
                "repl": repl if own.replication > 1 else None,
                "uses": [],
            }
        self.items[key]["uses"].append({"op": op.id, "opName": op.name, "role": role,
                                        "line": op.line_no})
        return key


def _interesting_values(op: Op):
    if op.is_("load", "store", "async_copy_global_to_local"):
        yield "pointer", op.operands[0]
    if op.is_("load"):
        yield "result", op.results[0]
    if op.is_("local_load", "dot", "convert_layout", "reduce", "elementwise_inline_asm",
              "trans"):
        for res in op.results:
            yield "result", res
    if op.is_("convert_layout", "local_alloc", "local_store", "dot", "reduce", "trans"):
        for i, operand in enumerate(op.operands):
            yield f"operand {i}", operand


def _shared_buffers(an: Analysis, accesses: list[SharedAccess]) -> list[dict]:
    buffers: dict[str, dict] = {}
    for op in an.func.find("local_alloc"):
        t = op.result_type
        if not isinstance(t, MemDescType) or t.encoding is None:
            continue
        try:
            layout = from_encoding(t.encoding)
        except UnsupportedLayout:
            continue
        if not isinstance(layout, SharedLayout) or len(t.shape) < 2:
            continue
        rows, cols = t.shape[-2], t.shape[-1]
        if rows * cols > MAX_ELEMENTS:
            continue
        lead = [0] * (len(t.shape) - 2)
        banks = [[layout.bank(tuple(lead + [r, c]), t.shape, t.element_bytes)
                  for c in range(cols)] for r in range(rows)]
        buffers[str(op.id)] = {
            "id": str(op.id), "op": op.id, "line": op.line_no, "name": t.encoding.name,
            "shape": list(t.shape), "tile": [rows, cols], "elemBytes": t.element_bytes,
            "layout": layout.describe(), "banks": banks,
            "bytes": _prod(t.shape) * t.element_bytes,
        }
    for acc in accesses:
        if not acc.op.is_("convert_layout"):
            continue
        scratch = acc.buffer.layout
        shape = acc.buffer.view_shape
        if len(shape) != 2 or _prod(shape) > MAX_ELEMENTS:
            continue
        rows, cols = shape
        elem = acc.buffer.elem_bytes
        banks = [[bank_of(scratch.byte_offset((r, c), shape, elem)) for c in range(cols)]
                 for r in range(rows)]
        bid = f"cvt{acc.op.id}"
        buffers[bid] = {
            "id": bid, "op": acc.op.id, "line": acc.op.line_no,
            "name": f"convert_layout scratch (op {acc.op.id})", "shape": shape,
            "tile": shape, "elemBytes": elem, "layout": scratch.describe(), "banks": banks,
            "bytes": _prod(scratch.padded_shape) * elem, "estimated": True,
        }
    return list(buffers.values())


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def _shared_access_json(acc: SharedAccess) -> dict:
    data = acc.to_json()
    lead = len(acc.buffer.alloc_shape) - len(acc.buffer.view_shape)
    offsets = acc.buffer.offsets

    def tile_coord(coord):
        full = list(offsets)
        for i, c in enumerate(coord):
            full[lead + i] += c
        return full[-2:]

    for instr in ([data["worst"]] if data["worst"] else []) + data["warp0"]:
        for lane in instr["lanes"]:
            lane["tile"] = tile_coord(lane["coord"]) if acc.buffer.alloc is not None \
                else lane["coord"]
    data["bufferId"] = str(acc.buffer.alloc.id) if acc.buffer.alloc is not None \
        else f"cvt{acc.op.id}"
    return data


def _find_source(an: Analysis, roots: list[Path]) -> dict | None:
    files: dict[str, int] = {}
    for op in an.func.ops:
        if op.loc and op.loc.file:
            files[op.loc.file] = files.get(op.loc.file, 0) + 1
    if not files:
        return None
    main = max(files, key=files.get)
    candidates = [Path(main)] + [r / main for r in roots]
    norm = main.replace("\\", "/")
    candidates += [r / norm for r in roots] + [Path(norm)]
    for cand in candidates:
        try:
            if cand.is_file():
                return {"file": main, "lines": cand.read_text(encoding="utf-8").splitlines()}
        except OSError:
            continue
    return {"file": main, "lines": None}


def _tiling(an: Analysis) -> dict | None:
    launch = an.launch
    if launch is None or not launch.grid:
        return None
    store = _main_store(an.func)
    if store is None:
        return None
    base = base_pointer_arg(store.operands[0])
    py = an.arg_names.get(base.name) if base is not None else None
    tensor = next((vars(a) for a in launch.args if a.name == py), None)
    tiling = program_tiling(an.func, an.arg_values, launch.grid, tensor, py)
    return tiling.to_json() if tiling else None


def _ops_json(an: Analysis) -> list[dict]:
    out = []
    for op in an.func.ops:
        rtype = op.result_type
        enc = getattr(rtype, "encoding", None)
        out.append({
            "id": op.id, "name": op.name, "line": op.line_no, "depth": op.depth,
            "loc": op.loc.to_json() if op.loc else None,
            "type": rtype.text if rtype is not None else None,
            "encoding": enc.name if enc is not None else None,
        })
    return out


def _perf(an: Analysis) -> dict | None:
    try:
        return an.perf.to_json()
    except Exception:  # the report must render even if the perf model cannot
        return None


def build_report(an: Analysis, findings: list[Finding] | None = None,
                 source_roots: list[Path] | None = None) -> dict[str, Any]:
    findings = run_lint(an) if findings is None else findings
    roots = [Path.cwd()] + list(source_roots or [])
    tensors = _TensorTable(an)
    op_tensors: dict[int, list[dict]] = {}
    for op in an.func.ops:
        for role, val in _interesting_values(op):
            key = tensors.add(val, role, op)
            if key:
                op_tensors.setdefault(op.id, []).append({"role": role, "key": key})

    global_json = []
    for acc in an.global_accesses:
        data = acc.to_json()
        data["tensor"] = _tensor_key(acc.op.operands[0].type.encoding.key(), acc.shape)
        data["baseArgName"] = an.arg_names.get(data["baseArg"] or "", data["baseArg"])
        global_json.append(data)

    mod = an.module
    meta = an.metadata
    launch = an.launch
    return {
        "tool": {"name": "tritonlens", "version": __version__},
        "kernel": {
            "name": an.name,
            "numWarps": mod.num_warps,
            "threadsPerWarp": mod.threads_per_warp,
            "numCtas": mod.num_ctas,
            "target": mod.target,
            "grid": launch.grid if launch else None,
            "numStages": meta.get("num_stages"),
            "sharedBytes": meta.get("shared"),
            "registers": meta.get("n_regs"),
            "spills": meta.get("n_spills"),
        },
        "args": [vars(a) for a in launch.args] if launch else [],
        "argNames": an.arg_names,
        "layouts": {name: {"kind": enc.kind, "text": enc.text, "params": {
            k: (v.name if hasattr(v, "kind") else v) for k, v in enc.params.items()}}
            for name, enc in mod.layouts.items()},
        "ops": _ops_json(an),
        "opTensors": {str(k): v for k, v in op_tensors.items()},
        "tensors": list(tensors.items.values()),
        "shared": {
            "buffers": _shared_buffers(an, an.shared_accesses),
            "accesses": [_shared_access_json(a) for a in an.shared_accesses],
        },
        "global": global_json,
        "findings": [f.to_json() for f in findings],
        "tiling": _tiling(an),
        "perf": _perf(an),
        "ptx": an.ptx.to_json() if an.ptx else None,
        "ir": mod.text,
        "source": _find_source(an, roots),
    }
