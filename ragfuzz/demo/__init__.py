"""Demo application package for ragfuzz."""

from .app import create_app, create_demo_app
from .state import DemoState

app = create_app

__all__ = ["DemoState", "app", "create_app", "create_demo_app"]
