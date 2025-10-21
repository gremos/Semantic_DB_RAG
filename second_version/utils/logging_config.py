import logging
import sys
from datetime import datetime

def setup_logging(level=logging.INFO):
    """Configure logging for observability."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(
        f'semantic_engine_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

logger = setup_logging()