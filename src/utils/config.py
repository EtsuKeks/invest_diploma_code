from pydantic import BaseModel  # type: ignore
from pathlib import Path
import os

N_THREADS = "8"

os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"


class PipelineSettings(BaseModel):
    input_csv: str = "binance_dump_with_valid_volumes_arbitrage_free.csv"
    output_csv: str = "binance_dump_with_models_preds_inaccurate.csv"
    risk_free_rate: float = 0.0


class BlackScholesSettings(BaseModel):
    refine_factor: int = 5
    epsilon: float = 1e-8
    radius: float = 0.02
    radius_calibrate_factor: float = 2.0
    sigma_min: float = 0.001
    sigma_max: float = 100.0
    points_initial: int = 10000
    points_calibrate: int = 1000
    max_refines_initial: int = 100
    max_refines_calibrate: int = 10


class GlobalSettings(BaseModel):
    ppl: PipelineSettings = PipelineSettings()
    bs: BlackScholesSettings = BlackScholesSettings()


settings = GlobalSettings()
