"""ragfuzz: Grey-box RAG Auditor and Prompt Fuzzer."""

__version__ = "0.2.0"

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging for ragfuzz.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
    )


logger = logging.getLogger("ragfuzz")
