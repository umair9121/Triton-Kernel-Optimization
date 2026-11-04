"""What the GPU can actually do: hardware limits, theoretical peaks and measured peaks.

Small GPUs fail differently from datacenter parts: a config tuned for 108 SMs and 164 KB of
shared memory either does not fit (OutOfResources after a slow compile) or leaves most of a
20-SM laptop GPU idle. Every device-aware analysis in tritonlens starts from a
:class:`DeviceSpec`, built from

* a per-architecture table of occupancy limits (from the CUDA occupancy calculator),
* the live device (``torch`` / Triton driver properties) when a GPU is present,
* optional measured peaks (``tritonlens device --calibrate``), cached per GPU, because a laptop
  GPU under its power limit rarely reaches the datasheet numbers.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ArchLimits:
    max_warps_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int = 65536
    max_regs_per_thread: int = 255
    reg_alloc_unit: int = 256  # registers, per warp
    shared_per_sm: int = 102400
    max_shared_per_block: int = 101376
    reserved_shared_per_block: int = 1024
    shared_alloc_unit: int = 128


ARCH_LIMITS: dict[int, ArchLimits] = {
    70: ArchLimits(64, 32, shared_per_sm=98304, max_shared_per_block=98304,
                   reserved_shared_per_block=0, shared_alloc_unit=256),
    75: ArchLimits(32, 16, shared_per_sm=65536, max_shared_per_block=65536,
                   reserved_shared_per_block=0, shared_alloc_unit=256),
    80: ArchLimits(64, 32, shared_per_sm=167936, max_shared_per_block=166912),
    86: ArchLimits(48, 16, shared_per_sm=102400, max_shared_per_block=101376),
    87: ArchLimits(48, 16, shared_per_sm=167936, max_shared_per_block=166912),
    89: ArchLimits(48, 24, shared_per_sm=102400, max_shared_per_block=101376),
    90: ArchLimits(64, 32, shared_per_sm=233472, max_shared_per_block=232448),
}


def limits_for(arch: int) -> ArchLimits:
    if arch in ARCH_LIMITS:
        return ARCH_LIMITS[arch]
    lower = [a for a in ARCH_LIMITS if a <= arch]
    return ARCH_LIMITS[max(lower)] if lower else ARCH_LIMITS[80]


@dataclass
class DeviceSpec:
    name: str
    arch: int
    sm_count: int
    total_memory: int = 0
    l2_bytes: int = 0
    sm_clock_khz: int = 0
    mem_clock_khz: int = 0
    mem_bus_width: int = 0
    measured: dict[str, float] = field(default_factory=dict)  # gbps, fp16_tflops, fp32_tflops
    source: str = "table"

    @property
    def limits(self) -> ArchLimits:
        return limits_for(self.arch)

    @property
    def theoretical_gbps(self) -> float | None:
        if not (self.mem_clock_khz and self.mem_bus_width):
            return None
        # GDDR: the reported memory clock is already the DDR "effective / 2" rate.
        return 2 * self.mem_clock_khz * 1e3 * self.mem_bus_width / 8 / 1e9

    @property
    def bandwidth_gbps(self) -> float | None:
        """Best estimate of attainable DRAM bandwidth (measured if calibrated)."""
        return self.measured.get("gbps") or self.theoretical_gbps

    def peak_tflops(self, dtype: str = "fp16") -> float | None:
        return self.measured.get(f"{dtype}_tflops")

    @property
    def max_resident_warps(self) -> int:
        return self.sm_count * self.limits.max_warps_per_sm

    def to_json(self) -> dict:
        data = asdict(self)
        data["theoretical_gbps"] = self.theoretical_gbps
        data["limits"] = asdict(self.limits)
        return data

    def describe(self) -> str:
        lim = self.limits
        lines = [
            f"{self.name} (sm_{self.arch}, {self.sm_count} SMs, "
            f"{self.total_memory / 2**30:.1f} GiB, L2 {self.l2_bytes / 2**20:.0f} MiB)",
            f"  per SM: {lim.max_warps_per_sm} warps, {lim.max_blocks_per_sm} blocks, "
            f"{lim.regs_per_sm} registers, {lim.shared_per_sm // 1024} KiB shared "
            f"(max {lim.max_shared_per_block // 1024} KiB per block)",
        ]
        if self.theoretical_gbps:
            lines.append(f"  DRAM bandwidth: {self.theoretical_gbps:.0f} GB/s theoretical")
        for key, label in (("gbps", "GB/s copy"), ("fp16_tflops", "TFLOP/s fp16 tensor core"),
                           ("fp32_tflops", "TFLOP/s fp32")):
            if key in self.measured:
                lines.append(f"  measured: {self.measured[key]:.1f} {label}")
        if not self.measured:
            lines.append("  not calibrated: run `tritonlens device --calibrate`")
        return "\n".join(lines)


# Used when analyzing IR without a GPU attached: the IR says sm_XX, the rest is a guess.
KNOWN_DEVICES = {
    "NVIDIA GeForce RTX 3050 Ti Laptop GPU": dict(arch=86, sm_count=20, total_memory=4 << 30,
                                                  l2_bytes=2 << 20, mem_clock_khz=6001000,
                                                  mem_bus_width=128),
}


def _cache_file() -> Path:
    base = os.environ.get("TRITONLENS_CACHE") or os.environ.get("LOCALAPPDATA") or \
        os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tritonlens" / "devices.json"


def load_calibration(name: str) -> dict[str, float]:
    path = _cache_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(name, {})
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(name: str, measured: dict[str, float]) -> Path:
    path = _cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[name] = measured
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def detect(index: int = 0) -> DeviceSpec | None:
    """Describe the live GPU, or ``None`` if CUDA is not available."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(index)
    spec = DeviceSpec(name=props.name, arch=props.major * 10 + props.minor,
                      sm_count=props.multi_processor_count, total_memory=props.total_memory,
                      l2_bytes=getattr(props, "L2_cache_size", 0), source="live")
    try:
        from triton.runtime import driver

        tp = driver.active.utils.get_device_properties(index)
        spec.sm_clock_khz = tp.get("sm_clock_rate", 0)
        spec.mem_clock_khz = tp.get("mem_clock_rate", 0)
        spec.mem_bus_width = tp.get("mem_bus_width", 0)
    except Exception:
        pass
    spec.measured = load_calibration(spec.name)
    return spec


def for_target(target: str, name: str | None = None) -> DeviceSpec:
    """A spec for analysis: the live GPU if it matches ``cuda:XX``, else a table entry."""
    arch = int(target.split(":")[-1]) if target and target.split(":")[-1].isdigit() else 86
    live = detect()
    if live is not None and live.arch == arch and (name is None or name == live.name):
        return live
    if name in KNOWN_DEVICES:
        spec = DeviceSpec(name=name, **KNOWN_DEVICES[name])
        spec.measured = load_calibration(name)
        return spec
    for known, info in KNOWN_DEVICES.items():
        if info["arch"] == arch:
            spec = DeviceSpec(name=known, **info)
            spec.measured = load_calibration(known)
            return spec
    return DeviceSpec(name=f"generic sm_{arch}", arch=arch, sm_count=20)


def calibrate(index: int = 0) -> DeviceSpec:
    """Measure copy bandwidth and matmul throughput (takes a few seconds)."""
    import torch
    from triton.testing import do_bench

    spec = detect(index)
    if spec is None:
        raise RuntimeError("no CUDA device")
    dev = f"cuda:{index}"
    # Leave most of a 4 GB card free: 2 x 256 MB is far beyond any L2.
    nbytes = min(256 << 20, torch.cuda.mem_get_info(index)[0] // 4)

    def bandwidth() -> float:
        src = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
        dst = torch.empty_like(src)
        ms = _median_bench(lambda: dst.copy_(src), do_bench)
        return 2 * nbytes / (ms * 1e-3) / 1e9

    def matmul_tflops(dtype, n: int = 4096) -> float:
        a = torch.randn(n, n, dtype=dtype, device=dev)
        b = torch.randn(n, n, dtype=dtype, device=dev)
        ms = _median_bench(lambda: a @ b, do_bench)
        return 2 * n ** 3 / (ms * 1e-3) / 1e12

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        measured = {"gbps": bandwidth(), "fp16_tflops": matmul_tflops(torch.float16),
                    "fp32_tflops": matmul_tflops(torch.float32)}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.cuda.empty_cache()
    spec.measured = {k: round(v, 2) for k, v in measured.items()}
    save_calibration(spec.name, spec.measured)
    return spec


def _median_bench(fn, do_bench, rounds: int = 5) -> float:
    times = sorted(do_bench(fn, warmup=25, rep=100) for _ in range(rounds))
    return times[len(times) // 2]
