from src.models.abc.gridsearch_model import GSModelParamDetails, GSModelParams
from typing import List


class BlackSholesSigmaSettings(GSModelParamDetails):
    name: str = "sigma"
    min_value_initial: float = 0.001
    max_value_initial: float = 10.0
    radius_calibrate: float = 0.03
    points_initial: int = 10000
    points_calibrate: int = 10000


class BlackScholesSettings(GSModelParams):
    refine_factor_initial: float = 5.0
    refine_factor_calibrate: float = 2.0
    max_refines_initial: int = 100
    max_refines_calibrate: int = 10
    params_details: List[GSModelParamDetails] = [BlackSholesSigmaSettings()]
