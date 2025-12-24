"""Utility modules."""

from .async_client import close_client, get_async_client

__all__ = ["close_client", "get_async_client"]

try:
    from .vram import VRAMMonitor

    __all__.append("VRAMMonitor")
except ImportError:
    VRAMMonitor = None
