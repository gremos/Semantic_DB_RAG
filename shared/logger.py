#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced logging configuration for the Semantic Database RAG System
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Optional: Try to use colorlog for colored console output
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

def setup_logging(log_level: str = "INFO", log_file: str = None, use_colors: bool = True):
    """
    Setup enhanced logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_colors: Whether to use colored console output
    """
    # Create logs directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"semantic_rag_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with optional colors
    console_handler = logging.StreamHandler(sys.stdout)
    
    if use_colors and HAS_COLORLOG:
        # Colored console output
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s | %(cyan)s%(name)-15s%(reset)s | %(message)s',
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler.setFormatter(color_formatter)
    else:
        # Plain console output
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(name)-15s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    # Log startup message
    logger.info(f"Enhanced Semantic RAG System logging initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module"""
    return logging.getLogger(name)

# Module-specific loggers
def get_discovery_logger() -> logging.Logger:
    """Get logger for database discovery operations"""
    return get_logger("discovery")

def get_semantic_logger() -> logging.Logger:
    """Get logger for semantic analysis operations"""
    return get_logger("semantic")

def get_query_logger() -> logging.Logger:
    """Get logger for query interface operations"""
    return get_logger("query")

def get_config_logger() -> logging.Logger:
    """Get logger for configuration operations"""
    return get_logger("config")

# Context manager for temporary log level changes
class TemporaryLogLevel:
    """Context manager to temporarily change log level"""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)

# Performance logging decorator
def log_performance(logger: logging.Logger = None):
    """Decorator to log function execution time"""
    import time
    from functools import wraps
    
    if logger is None:
        logger = get_logger("performance")
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Error logging helper
def log_error_with_context(logger: logging.Logger, error: Exception, context: dict = None):
    """Log an error with additional context information"""
    error_msg = f"Error: {type(error).__name__}: {str(error)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_msg += f" | Context: {context_str}"
    
    logger.error(error_msg, exc_info=True)

# System health logging
def log_system_health(tables_count: int = 0, 
                     views_count: int = 0, 
                     analysis_success_rate: float = 0.0,
                     cache_status: str = "unknown"):
    """Log system health metrics"""
    health_logger = get_logger("health")
    health_logger.info(f"System Health Check:")
    health_logger.info(f"  Tables discovered: {tables_count}")
    health_logger.info(f"  Views discovered: {views_count}")
    health_logger.info(f"  Analysis success rate: {analysis_success_rate:.2%}")
    health_logger.info(f"  Cache status: {cache_status}")