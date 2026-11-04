"""TL006: check launch arguments before Triton compiles the kernel.

Triton specializes every pointer whose ``data_ptr()`` is a multiple of 16 and every integer
that is a multiple of 16 with ``tt.divisibility = 16``; that attribute is what lets the
compiler emit 16-byte vector loads. An argument that misses it silently produces a slower
kernel (and a separate compilation). These checks run on plain argument descriptions, so they
work on captured launches and on live arguments via :func:`check`.
"""

from __future__ import annotations

from ..capture import ArgInfo, describe_arg
from .lint import Finding

ALIGN = 16


def check_args(args: list[ArgInfo], kernel_name: str = "kernel") -> list[Finding]:
    findings: list[Finding] = []
    for arg in args:
        if arg.kind == "tensor" and arg.data_ptr is not None:
            misalign = arg.data_ptr % ALIGN
            if misalign:
                findings.append(Finding(
                    "TL006", "warning",
                    f"`{arg.name}` ({arg.dtype}{list(arg.shape or [])}) starts {misalign} bytes "
                    f"past a 16-byte boundary: Triton will not mark it tt.divisibility=16 and "
                    f"`{kernel_name}` falls back to narrow loads/stores.",
                    fix="allocate a fresh contiguous tensor (or .clone()) instead of passing an "
                        "offset view",
                    data={"arg": arg.name, "misalignment": misalign}))
            if arg.strides and arg.element_size and arg.shape and len(arg.shape) > 1:
                inner = arg.strides[-1]
                if inner != 1:
                    findings.append(Finding(
                        "TL006", "info",
                        f"`{arg.name}` is not contiguous in its last dimension "
                        f"(strides {arg.strides}); loads along it cannot vectorize.",
                        fix="pass .contiguous() or swap the kernel's fast axis",
                        data={"arg": arg.name, "strides": arg.strides}))
                for dim, stride in enumerate(arg.strides[:-1]):
                    if stride * arg.element_size % ALIGN and arg.shape[dim] > 1:
                        findings.append(Finding(
                            "TL006", "info",
                            f"`{arg.name}` has a row pitch of {stride} elements "
                            f"({stride * arg.element_size} B) along dim {dim}, not a multiple "
                            "of 16 bytes: rows after the first are misaligned.",
                            fix="pad the leading dimension to a multiple of 16 bytes",
                            data={"arg": arg.name, "dim": dim, "stride": stride}))
                        break
        elif arg.kind == "int" and isinstance(arg.value, int):
            if arg.value != 1 and arg.value % ALIGN and arg.value > ALIGN:
                findings.append(Finding(
                    "TL006", "info",
                    f"integer `{arg.name}`={arg.value} is not a multiple of 16; any mask or "
                    "offset derived from it is treated as unaligned.",
                    fix=f"pad `{arg.name}` to {(arg.value + 15) // 16 * 16} if it is a size or "
                        "row pitch",
                    data={"arg": arg.name, "value": arg.value}))
    return findings


def check_memory(args: list[ArgInfo], kernel_name: str = "kernel") -> list[Finding]:
    """VRAM headroom for live arguments (TL010), skipped when no GPU is visible."""
    from ..device import detect
    from .budget import fmt_bytes, tensor_footprint

    spec = detect()
    if spec is None or not spec.total_memory:
        return []
    used, _ = tensor_footprint(args)
    if not used:
        return []
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
    except Exception:
        free, total = None, spec.total_memory
    frac = used / total
    small = total <= 8 << 30
    if frac < (0.55 if small else 0.92) and not (free is not None and free < 384 << 20):
        return []
    sev = "warning" if frac >= 0.85 or (free is not None and free < 384 << 20) else "info"
    free_txt = f", {fmt_bytes(free)} free" if free is not None else ""
    return [Finding(
        "TL010", sev,
        f"`{kernel_name}` tensors occupy {fmt_bytes(used)} of {fmt_bytes(total)} "
        f"({frac:.0%}){free_txt}. On a small GPU the next compile or workspace may OOM.",
        fix="use a smaller dtype, tile the batch, or free unused tensors first",
        data={"tensorBytes": used, "totalMemory": total, "fraction": frac, "freeMemory": free})]


def check(kernel, *args, **kwargs) -> list[Finding]:
    """Check live launch arguments for a ``@triton.jit`` function before launching it.

    >>> for finding in tritonlens.check(add_kernel, x, y, out, n):
    ...     print(finding.message)
    """
    import inspect

    fn = getattr(kernel, "fn", kernel)
    sig = inspect.signature(fn)
    bound = sig.bind_partial(*args, **{k: v for k, v in kwargs.items() if k in sig.parameters})
    constexprs = {p.name for p in getattr(kernel, "params", []) if p.is_constexpr}
    infos = [describe_arg(name, value, name in constexprs)
             for name, value in bound.arguments.items()]
    name = getattr(kernel, "__name__", fn.__name__)
    return check_args(infos, name) + check_memory(infos, name)


def guard(raise_on: str | None = None):
    """Context manager that runs :func:`check` before every Triton launch inside it."""
    import contextlib

    from triton.runtime.jit import JITFunction

    @contextlib.contextmanager
    def _ctx():
        original = JITFunction.run
        seen: set[tuple] = set()

        def run(self, *args, grid, warmup, **kwargs):
            findings = check(self, *args, **kwargs)
            key = (self.__name__, tuple(f.message for f in findings))
            if findings and key not in seen:
                seen.add(key)
                for f in findings:
                    print(f"[tritonlens {f.rule} {f.severity}] {f.message}")
                if raise_on and any(f.severity == raise_on for f in findings):
                    raise ValueError(f"{self.__name__}: {findings[0].message}")
            return original(self, *args, grid=grid, warmup=warmup, **kwargs)

        JITFunction.run = run
        try:
            yield
        finally:
            JITFunction.run = original

    return _ctx()
