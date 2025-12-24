"""Utility modules."""

from typing import Any

from .async_client import close_client, get_async_client

__all__ = ["close_client", "get_async_client"]

VRAMMonitor: Any = None

try:
    from .vram import VRAMMonitor

    __all__.append("VRAMMonitor")
except ImportError:
    pass  # VRAMMonitor remains None
