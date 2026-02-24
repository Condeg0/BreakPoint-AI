import logging
import sys
from pathlib import Path

def get_logger(name: str, artifact_dir: str = "logs") -> logging.Logger:
    """
    Instantiates a production-grade logger.
    Routes output to console and a persistent 'pipeline.log' in the artifact_dir.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times in the same runtime
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    
    # Strict formatting: Timestamp | Severity | Module | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Persistent File Handler
    log_dir = Path(artifact_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Standard Output Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Isolate from root logger to prevent third-party library noise
    logger.propagate = False

    return logger