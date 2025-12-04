from typing import List

from src.models.abc.gridsearch_model import GSModelParamDetails, GSModelParams


class SabrAlphaSettings(GSModelParamDetails):
    name: str = "alpha"
    min_value_initial: float = 0.001
    max_value_initial: float = 100.0
    radius_calibrate: float = 0.02
    points_initial: int = 10
    points_calibrate: int = 10


class SabrRhoSettings(GSModelParamDetails):
    name: str = "rho"
    min_value_initial: float = -0.999
    max_value_initial: float = 0.999
    radius_calibrate: float = 0.02
    points_initial: int = 10
    points_calibrate: int = 10


class SabrNuSettings(GSModelParamDetails):
    name: str = "nu"
    min_value_initial: float = 0.001
    max_value_initial: float = 100.0
    radius_calibrate: float = 0.02
    points_initial: int = 10
    points_calibrate: int = 10


class SABRSettings(GSModelParams):
    beta: float = 0.5

    refine_factor_initial: float = 5.0
    refine_factor_calibrate: float = 2.0
    max_refines_initial: int = 100
    max_refines_calibrate: int = 10

    params_details: List[GSModelParamDetails] = [SabrAlphaSettings(), SabrRhoSettings(), SabrNuSettings()]
