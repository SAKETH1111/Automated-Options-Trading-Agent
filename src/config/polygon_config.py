"""
Polygon.io Configuration
Stores API keys and S3 credentials for Polygon.io services
"""

import os
from typing import Dict, Any

# Polygon.io API Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', 'wWrUjjcksqLDPntXbJb72kiFzAwyqIpY')

# Polygon.io S3 Flat Files Configuration
POLYGON_S3_CONFIG = {
    'endpoint_url': 'https://files.polygon.io',
    'access_key_id': '88976b8e-f103-4d07-959f-4d13ce686dd2',
    'secret_access_key': 'wWrUjjcksqLDPntXbJb72kiFzAwyqIpY',
    'region_name': 'us-east-1',
    'bucket_name': 'flatfile'
}

# Data paths
DATA_PATHS = {
    'flat_files': 'data/flat_files',
    'ml_data': 'data/ml',
    'backtesting': 'data/backtesting'
}

def get_polygon_api_key() -> str:
    """Get Polygon.io API key"""
    return POLYGON_API_KEY

def get_s3_config() -> Dict[str, Any]:
    """Get S3 configuration for Polygon flat files"""
    return POLYGON_S3_CONFIG.copy()

def get_data_path(path_type: str) -> str:
    """Get data path for specific type"""
    return DATA_PATHS.get(path_type, 'data')
