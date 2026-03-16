"""
NextBuy — Shared data loader
Caches the DataFrame at the Python process level using a module-level variable.
This means the 1.2GB CSV is loaded ONCE per EC2 process, shared across all pages.
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

USE_S3        = os.getenv('USE_S3', 'false').lower() == 'true'
S3_BUCKET     = os.getenv('S3_BUCKET', '')
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, '..', 'data')

# Module-level cache — persists for the lifetime of the EC2 process
# All 3 pages import this module, Python only executes it once
_df_cache = None

def get_data() -> pd.DataFrame:
    """
    Returns the cleaned DataFrame.
    Loads from S3 or local on first call, then returns cached copy on all subsequent calls.
    """
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    storage_options = {
        'key':    os.getenv('AWS_ACCESS_KEY_ID'),
        'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'client_kwargs': {'region_name': os.getenv('AWS_REGION', 'eu-west-3')}
    }

    if USE_S3:
        try:
            _df_cache = pd.read_parquet(
                f's3://{S3_BUCKET}/cleaned_data.parquet',
                storage_options=storage_options
            )
        except Exception:
            _df_cache = pd.read_csv(
                f's3://{S3_BUCKET}/cleaned_data.csv',
                storage_options=storage_options
            )
    else:
        parquet_path = os.path.join(DATA_DIR, 'cleaned_data.parquet')
        if os.path.exists(parquet_path):
            _df_cache = pd.read_parquet(parquet_path)
        else:
            _df_cache = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'))

    # Pre-compute is_organic once at load time — not on every page render
    if 'is_organic' not in _df_cache.columns:
        _df_cache['is_organic'] = (
            _df_cache['product_name']
            .str.contains('Organic', case=False, na=False)
            .astype(int)
        )

    return _df_cache