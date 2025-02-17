"""
This file will contain constants used throughout the project
"""

from pathlib import Path

# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
METADATA_PATH = PARENT_PATH / '../metadata'
RAW24H_PATH = DATA_PATH / 'raw_24h'

# Contants 
NUM_SECS_DAY = 86400  # 60*60*24 
NUM_SECS_HOUR = 3600  # 60*60 