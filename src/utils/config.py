import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel  # type: ignore
from src.utils.models_config.bs_config import BlackScholesSettings
from src.utils.models_config.sabr_config import SABRSettings

NTHREADS = str(8)

os.environ["VECLIB_MAXIMUM_THREADS"] = NTHREADS
os.environ["OMP_NUM_THREADS"] = NTHREADS
os.environ["OPENBLAS_NUM_THREADS"] = NTHREADS
os.environ["MKL_NUM_THREADS"] = NTHREADS

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class PipelineSettings(BaseModel):
    input_csv: str = "binance_dump_with_valid_volumes_arbitrage_free.csv"
    output_csv: Optional[str] = None
    risk_free_rate: float = 0.0
    epsilon: float = 1e-8
    total_hours: int = 2683


class GlobalSettings(BaseModel):
    ppl: PipelineSettings = PipelineSettings()
    bs: BlackScholesSettings = BlackScholesSettings()
    sabr: SABRSettings = SABRSettings()


settings = GlobalSettings()
