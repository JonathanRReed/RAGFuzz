"""Storage implementations for ragfuzz."""

from .baseline import BaselineManager
from .run_dir import RunDir

__all__ = ["RunDir", "BaselineManager"]
