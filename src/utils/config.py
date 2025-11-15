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
OUTPUTS_DIR.mkdir(exist_ok=True)


class PipelineSettings(BaseModel):
    input_csv: str = "binance_dump_with_valid_volumes_arbitrage_free.csv"
    output_csv: str = "bs-default_params.csv"
    risk_free_rate: float = 0.0
    epsilon: float = 1e-8


class BlackScholesSettings(BaseModel):
    refine_factor: int = 5
    radius: float = 0.02
    radius_calibrate_factor: float = 2.0
    sigma_min: float = 0.001
    sigma_max: float = 100.0
    points_initial: int = 10000
    points_calibrate: int = 1000
    max_refines_initial: int = 100
    max_refines_calibrate: int = 10


class SABRSettings(BaseModel):
    refine_factor: int = 5
    radius: float = 0.02
    radius_calibrate_factor: float = 2.0
    alpha_min: float = 0.001
    alpha_max: float = 100.0
    rho_min: float = -0.999
    rho_max: float = 0.999
    nu_min: float = 0.001
    nu_max: float = 100.0
    beta: float = 0.5
    points_initial: int = 10
    points_calibrate: int = 10
    max_refines_initial: int = 100
    max_refines_calibrate: int = 10


class GlobalSettings(BaseModel):
    ppl: PipelineSettings = PipelineSettings()
    bs: BlackScholesSettings = BlackScholesSettings()
    sabr: SABRSettings = SABRSettings()


settings = GlobalSettings()
