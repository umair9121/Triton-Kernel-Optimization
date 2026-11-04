"""tritonlens: static analysis and visualization of Triton GPU memory layouts."""

from .analysis.runtime_check import check, guard
from .analyze import analyze, analyze_captured
from .capture import capture_script, capturing

__version__ = "0.1.0"

__all__ = ["analyze", "analyze_captured", "capture_script", "capturing", "check", "guard"]
