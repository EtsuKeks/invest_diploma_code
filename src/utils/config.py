import os
from pathlib import Path

from pydantic import BaseModel  # type: ignore

from src.utils.models_config.bs_config import BlackScholesSettings
from src.utils.models_config.sabr_config import SABRSettings
from src.utils.ppl_config import PipelineSettings

NTHREADS = str(8)

os.environ["VECLIB_MAXIMUM_THREADS"] = NTHREADS
os.environ["OMP_NUM_THREADS"] = NTHREADS
os.environ["OPENBLAS_NUM_THREADS"] = NTHREADS
os.environ["MKL_NUM_THREADS"] = NTHREADS

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class GlobalSettings(BaseModel):
    ppl: PipelineSettings = PipelineSettings()
    bs: BlackScholesSettings = BlackScholesSettings()
    sabr: SABRSettings = SABRSettings()


settings = GlobalSettings()
