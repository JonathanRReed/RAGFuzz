"""Utility modules."""

import importlib.util
from typing import Any

from .async_client import close_client, get_async_client, should_trust_env

__all__ = ["close_client", "get_async_client", "should_trust_env"]

VRAMMonitor: Any = None

if importlib.util.find_spec(".vram", __package__):
    from .vram import VRAMMonitor as _VRAMMonitor

    VRAMMonitor = _VRAMMonitor
    __all__.append("VRAMMonitor")
