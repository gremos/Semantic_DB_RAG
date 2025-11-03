"""
Logging configuration for the application.
Sets up colorized console and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import colorlog
from config.settings import Settings


def setup_logging(settings: Settings) -> logging.Logger:
    """
    Setup application logging with console and file handlers.
    
    Args:
        settings: Application settings
        
    Returns:
        Configured root logger
    """
    # Ensure log directory exists
    log_dir = settings.paths.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    console_handler = colorlog.StreamHandler(sys.stdout)

    # Console handler with colors
    console_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for discovery/semantic logs
    from logging.handlers import RotatingFileHandler
    
    # File handler for discovery/semantic logs with rotation
    # Max 10MB per file, keep 5 backup files
    file_handler = RotatingFileHandler(
        log_dir / 'discovery_semantic.log',
        mode='a',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )

    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
