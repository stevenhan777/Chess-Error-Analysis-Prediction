"""
src/logger.py
Centralised logging configuration for the chess error analysis project.
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_file = os.path.join(
    LOG_DIR,
    datetime.now().strftime("chess_%Y_%m_%d_%H_%M_%S.log"),
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("chess_analysis")
