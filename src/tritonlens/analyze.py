"""One-stop analysis of a compiled Triton kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analysis import banks, coalescing
from .analysis.axis import AxisAnalysis
from .analysis.ptx import PtxStats, parse_ptx
from .capture import ArgInfo, CapturedKernel
from .ir import Module, parse_module


@dataclass
class Analysis:
    name: str
    module: Module
    ptx: PtxStats | None
    axis: AxisAnalysis
    global_accesses: list[coalescing.GlobalAccess]
    shared_accesses: list[banks.SharedAccess]
    arg_names: dict[str, str] = field(default_factory=dict)  # %argN -> python name
    arg_values: dict[str, int] = field(default_factory=dict)  # %argN -> int value
    launch: CapturedKernel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def func(self):
        return self.module.func

    def pretty(self, name: str) -> str:
        """Python-level name for an IR value name such as ``%arg3``."""
        py = self.arg_names.get(name)
        return f"`{py}`" if py else f"`{name}`"


def map_arguments(module: Module, args: list[ArgInfo]) -> tuple[dict[str, str], dict[str, int]]:
    """Match captured Python arguments to the IR function arguments.

    Triton drops constexprs, ``None`` and integers equal to 1 (they are specialized into the
    kernel) from the compiled signature.
    """
    runtime = [a for a in args if a.kind not in ("constexpr", "none")
               and not (a.kind == "int" and a.value == 1)]
    func_args = module.func.args
    names: dict[str, str] = {}
    values: dict[str, int] = {}
    if len(runtime) != len(func_args):
        return names, values
    for ir_arg, py_arg in zip(func_args, runtime):
        names[ir_arg.name] = py_arg.name
        if py_arg.kind == "int" and isinstance(py_arg.value, int):
            values[ir_arg.name] = py_arg.value
    return names, values


def analyze(ttgir: str, ptx: str | None = None, launch: CapturedKernel | None = None,
            name: str | None = None) -> Analysis:
    module = parse_module(ttgir)
    arg_names, arg_values = ({}, {})
    if launch is not None and launch.args:
        arg_names, arg_values = map_arguments(module, launch.args)
    ptx_stats = parse_ptx(ptx) if ptx else None
    axis = AxisAnalysis(module.func, arg_values)
    nw, tpw = module.num_warps, module.threads_per_warp
    global_acc = coalescing.analyze_function(module.func, nw, tpw, arg_values, ptx_stats, axis)
    shared_acc = banks.analyze_function(module.func, nw, tpw)
    return Analysis(
        name=name or (launch.name if launch else module.func.name),
        module=module, ptx=ptx_stats, axis=axis, global_accesses=global_acc,
        shared_accesses=shared_acc, arg_names=arg_names, arg_values=arg_values, launch=launch,
        metadata=dict(launch.metadata) if launch else {},
    )


def analyze_captured(kernel: CapturedKernel) -> Analysis:
    return analyze(kernel.ttgir, kernel.ir.get("ptx"), kernel)
