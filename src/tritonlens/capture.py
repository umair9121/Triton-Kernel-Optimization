"""Capture Triton IR and launch information from real kernel launches.

Two capture paths are supported:

* In-process: :func:`capturing` patches ``JITFunction.run`` so that every launch records the
  compiled kernel (all IR stages), the resolved grid and a description of every argument.
* Dump directory: :func:`load_dump_dir` reads the files Triton writes when the script is run
  with ``TRITON_KERNEL_DUMP=1`` and ``TRITON_DUMP_DIR=<dir>``.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import runpy
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IR_STAGES = ("ttir", "ttgir", "llir", "ptx")


@dataclass
class ArgInfo:
    name: str
    kind: str  # "tensor", "int", "float", "constexpr", "none", "other"
    value: Any = None
    dtype: str | None = None
    shape: list[int] | None = None
    strides: list[int] | None = None
    data_ptr: int | None = None
    element_size: int | None = None
    contiguous: bool | None = None


@dataclass
class CapturedKernel:
    name: str
    hash: str
    ir: dict[str, str]
    metadata: dict[str, Any]
    grid: list[int] | None = None
    args: list[ArgInfo] = field(default_factory=list)
    launches: int = 1

    @property
    def ttgir(self) -> str:
        return self.ir.get("ttgir", "")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hash": self.hash,
            "metadata": self.metadata,
            "grid": self.grid,
            "args": [asdict(a) for a in self.args],
            "launches": self.launches,
        }

    def save(self, out_dir: str | Path) -> Path:
        """Write every IR stage plus ``launch.json`` to ``out_dir/<name>_<hash8>/``."""
        target = Path(out_dir) / f"{self.name}_{self.hash[:8]}"
        target.mkdir(parents=True, exist_ok=True)
        for stage, text in self.ir.items():
            (target / f"{self.name}.{stage}").write_text(text, encoding="utf-8")
        (target / "launch.json").write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> CapturedKernel:
        """Load a kernel saved with :meth:`save` (``path`` is the kernel directory)."""
        path = Path(path)
        meta_file = path / "launch.json"
        info = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        name = info.get("name") or path.name.rsplit("_", 1)[0]
        ir = {}
        for stage in IR_STAGES:
            files = sorted(path.glob(f"*.{stage}"))
            if files:
                ir[stage] = files[0].read_text(encoding="utf-8")
        return cls(
            name=name,
            hash=info.get("hash", ""),
            ir=ir,
            metadata=info.get("metadata", {}),
            grid=info.get("grid"),
            args=[ArgInfo(**a) for a in info.get("args", [])],
            launches=info.get("launches", 1),
        )


def _metadata_dict(compiled) -> dict[str, Any]:
    meta = compiled.metadata
    keys = ("num_warps", "num_stages", "num_ctas", "shared", "target", "name", "debug")
    out: dict[str, Any] = {}
    for key in keys:
        if hasattr(meta, key):
            value = getattr(meta, key)
            if key == "target":
                value = {"backend": value.backend, "arch": value.arch, "warp_size": value.warp_size}
            out[key] = value
    for attr in ("n_regs", "n_spills"):
        value = getattr(compiled, attr, None)
        if isinstance(value, int):
            out[attr] = value
    return out


def from_compiled(compiled, grid=None, args: list[ArgInfo] | None = None) -> CapturedKernel:
    """Build a :class:`CapturedKernel` from a ``triton.compiler.CompiledKernel``."""
    ir = {stage: compiled.asm[stage] for stage in IR_STAGES if stage in compiled.asm}
    return CapturedKernel(
        name=compiled.name,
        hash=str(compiled.hash),
        ir=ir,
        metadata=_metadata_dict(compiled),
        grid=list(grid) if grid is not None else None,
        args=args or [],
    )


def describe_arg(name: str, value: Any, is_constexpr: bool = False) -> ArgInfo:
    if is_constexpr:
        return ArgInfo(name=name, kind="constexpr", value=_jsonable(value))
    if value is None:
        return ArgInfo(name=name, kind="none")
    if hasattr(value, "data_ptr") and hasattr(value, "stride"):
        return ArgInfo(
            name=name,
            kind="tensor",
            dtype=str(value.dtype).replace("torch.", ""),
            shape=list(value.shape),
            strides=list(value.stride()),
            data_ptr=int(value.data_ptr()),
            element_size=int(value.element_size()),
            contiguous=bool(value.is_contiguous()),
        )
    if isinstance(value, bool):
        return ArgInfo(name=name, kind="other", value=value)
    if isinstance(value, int):
        return ArgInfo(name=name, kind="int", value=value)
    if isinstance(value, float):
        return ArgInfo(name=name, kind="float", value=value)
    return ArgInfo(name=name, kind="other", value=_jsonable(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def _bind_arguments(jit_fn, args, kwargs) -> dict[str, Any]:
    sig = inspect.signature(jit_fn.fn)
    bound = sig.bind_partial(*args, **{k: v for k, v in kwargs.items() if k in sig.parameters})
    bound.apply_defaults()
    return dict(bound.arguments)


def _resolve_grid(grid, bound: dict[str, Any], kwargs: dict[str, Any]) -> list[int] | None:
    if grid is None:
        return None
    if callable(grid):
        try:
            grid = grid({**kwargs, **bound})
        except Exception:
            return None
    grid = list(grid)
    while len(grid) < 3:
        grid.append(1)
    return [int(g) for g in grid]


class Recorder:
    """Collects one :class:`CapturedKernel` per unique compiled kernel."""

    def __init__(self) -> None:
        self.kernels: dict[str, CapturedKernel] = {}

    def record(self, jit_fn, compiled, args, kwargs, grid) -> None:
        if compiled is None or not hasattr(compiled, "asm"):
            return
        key = str(compiled.hash)
        if key in self.kernels:
            self.kernels[key].launches += 1
            return
        try:
            bound = _bind_arguments(jit_fn, args, kwargs)
        except TypeError:
            bound = {}
        constexpr_names = {p.name for p in jit_fn.params if p.is_constexpr}
        arg_infos = [describe_arg(n, v, n in constexpr_names) for n, v in bound.items()]
        captured = from_compiled(compiled, _resolve_grid(grid, bound, kwargs), arg_infos)
        self.kernels[key] = captured

    def __iter__(self):
        return iter(self.kernels.values())

    def __len__(self) -> int:
        return len(self.kernels)


@contextlib.contextmanager
def capturing():
    """Context manager that records every Triton kernel launched inside it.

    >>> with capturing() as rec:
    ...     add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    >>> kernel = next(iter(rec))
    """
    from triton.runtime.jit import JITFunction

    recorder = Recorder()
    original_run = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        compiled = original_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        try:
            recorder.record(self, compiled, args, kwargs, None if warmup else grid)
        except Exception as exc:  # capture must never break the user's program
            print(f"[tritonlens] capture failed for {self.__name__}: {exc}", file=sys.stderr)
        return compiled

    JITFunction.run = run
    try:
        yield recorder
    finally:
        JITFunction.run = original_run


def capture_script(script: str | Path, argv: list[str] | None = None) -> list[CapturedKernel]:
    """Run a Python script in-process and return every kernel it launched."""
    script = str(Path(script).resolve())
    old_argv = sys.argv
    old_path = list(sys.path)
    sys.argv = [script, *(argv or [])]
    sys.path.insert(0, str(Path(script).parent))
    try:
        with capturing() as recorder:
            runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    return list(recorder)


def load_dump_dir(dump_dir: str | Path) -> list[CapturedKernel]:
    """Load kernels from a ``TRITON_DUMP_DIR`` or from directories written by ``save``."""
    dump_dir = Path(dump_dir)
    kernels = []
    for ttgir in sorted(dump_dir.rglob("*.ttgir")):
        folder = ttgir.parent
        if (folder / "launch.json").exists():
            kernels.append(CapturedKernel.load(folder))
            continue
        ir = {}
        for stage in IR_STAGES:
            candidate = folder / f"{ttgir.stem}.{stage}"
            if candidate.exists():
                ir[stage] = candidate.read_text(encoding="utf-8")
        meta = {}
        meta_file = folder / f"{ttgir.stem}.json"
        if meta_file.exists():
            raw = json.loads(meta_file.read_text(encoding="utf-8"))
            meta = {k: raw[k] for k in ("num_warps", "num_stages", "shared", "name") if k in raw}
        kernels.append(CapturedKernel(name=ttgir.stem, hash=folder.name, ir=ir, metadata=meta))
    return kernels
