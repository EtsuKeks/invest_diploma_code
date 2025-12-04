from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pydantic import BaseModel

from src.models.abc.model import Model
from src.utils.ppl_config import EPSILON


def normalized_sse(close: np.ndarray, preds_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates normal deviation from the actual Close Prices.

    Parameters
    ----------
    close : Actual Close prices of shape (n,)
    preds_matrix : A matrix of shape (m, n) with every row representing predicted prices

    Returns an array of shape(m,) with every value being a normalized deviation of the corresponding m-th
    row of predicted prices from the actual close ones
    """
    diffs = preds_matrix - close[None, :]
    return np.sum(diffs * diffs, axis=1) / np.sum(close * close)


def _cartesian_grid(param_edges: List[np.ndarray]) -> np.ndarray:
    """
    Create cartesian product of parameter edges.

    Parameters
    ----------
    param_edges : List of p ndarrays of length ni for ith param

    Returns an array of shape (m, p), where m = n1 * n2 * n3 * ...
    """
    stacked = np.stack(np.meshgrid(*param_edges, indexing="ij"), axis=-1)
    return stacked.reshape(-1, stacked.shape[-1])


class GSModelParamDetails(BaseModel):
    name: str
    min_value_initial: float
    max_value_initial: float
    radius_calibrate: float
    points_initial: int
    points_calibrate: int


class GSModelParams(BaseModel):
    refine_factor_initial: float
    refine_factor_calibrate: float
    max_refines_initial: int
    max_refines_calibrate: int
    params_details: List[GSModelParamDetails]


class GridSearchModel(Model, ABC):
    """Base class that implements a general incremental grid search for parameter calibration."""

    def __init__(self):
        # stores current best parameters as an array of shape (p,) in the same order as gs_params().params_details
        self._params = None

    @abstractmethod
    def gs_params(self) -> GSModelParams:
        raise NotImplementedError

    @abstractmethod
    def _prices_for_param_grid(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, param_matrix: np.ndarray, r: float
    ) -> np.ndarray:
        """
        Vectorized computation of predicted prices for every row of param_matrix.

        Parameters
        ----------
        S : Underlying Prices of shape (n,)
        K : Strikes of shape (n,)
        T : Time To Maturities of shape (n,)
        is_call : The option type flag - Put or Call of shape (n,)
        param_matrix : An array of shape (m, p), where each row is a parameter vector in the SAME
            ordering as in gs_params().params_details
        r : Current Interest Rate

        Returns a price matrix of shape (m, n)
        """
        raise NotImplementedError

    def _grid_refine(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close: np.ndarray,
        r: float,
        lows: np.ndarray,
        highs: np.ndarray,
        points: np.ndarray,
        refine_factor: float,
        max_refines: int,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
    ) -> np.ndarray:
        """
        Core iterative routine that narrows [lows, highs] using the same logic for both
        initial search and calibration. The callers decide how lows/highs/points are prepared:
          - find_initial_params uses absolute min/max and points_initial
          - calibrate centers lows/highs around current +/- radius and uses points_calibrate

        Returns an array of shape (p,) with best params
        """
        best_params, best_score = None, float("inf")
        for _ in range(max_refines):
            edges = [np.linspace(lows[i], highs[i], points[i]) for i in range(lows.size)]
            param_matrix = _cartesian_grid(edges)
            prices = self._prices_for_param_grid(S, K, T, is_call, param_matrix, r)
            scores = normalized_sse(close, prices)
            idx = int(np.argmin(scores))
            if scores[idx] < best_score - EPSILON:
                best_score, best_params = scores[idx], param_matrix[idx]
                widths = (highs - lows) / refine_factor
                lows = np.maximum(min_bounds, best_params - widths / 2.0)
                highs = np.minimum(max_bounds, best_params + widths / 2.0)
            else:
                break

        return np.array(best_params)

    def find_initial_params(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float
    ) -> None:
        lows = np.array([p.min_value_initial for p in self.gs_params().params_details])
        highs = np.array([p.max_value_initial for p in self.gs_params().params_details])
        points = np.array([p.points_initial for p in self.gs_params().params_details])

        best = self._grid_refine(
            S,
            K,
            T,
            is_call,
            close,
            r,
            lows=lows,
            highs=highs,
            points=points,
            refine_factor=self.gs_params().refine_factor_initial,
            max_refines=self.gs_params().max_refines_initial,
            min_bounds=lows,
            max_bounds=highs,
        )
        self._params = np.array(best)

    def calibrate(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float
    ) -> None:
        radius = np.array([p.radius_calibrate for p in self.gs_params().params_details])
        min_bounds = np.array([p.min_value_initial for p in self.gs_params().params_details])
        max_bounds = np.array([p.max_value_initial for p in self.gs_params().params_details])
        lows, highs = np.maximum(min_bounds, self._params - radius), np.minimum(max_bounds, self._params + radius)

        best = self._grid_refine(
            S,
            K,
            T,
            is_call,
            close,
            r,
            lows=lows,
            highs=highs,
            points=np.array([p.points_calibrate for p in self.gs_params().params_details]),
            refine_factor=self.gs_params().refine_factor_calibrate,
            max_refines=self.gs_params().max_refines_calibrate,
            # Pass min_bounds=lows, max_bounds=highs as is believed that sufficient radius peaked,
            # thus no need in lossing accuracy
            min_bounds=lows,
            max_bounds=highs,
        )
        self._params = np.array(best)

    def price(self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, r: float) -> np.ndarray:
        return self._prices_for_param_grid(S, K, T, is_call, self._params.reshape(1, -1), r)[0]
