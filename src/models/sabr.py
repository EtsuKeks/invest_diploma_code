from typing import Optional

import numpy as np

from src.models.abc.gridsearch_model import GridSearchModel, GSModelParams
from src.models.abc.spot_cautious_model import SpotCautiousModel
from src.models.black_sholes import prices_for_sigmas
from src.utils.config import settings
from scipy.stats import linregress


def _sabr_implied_vol(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    rho: np.ndarray,
    nu: np.ndarray,
    r: float,
) -> np.ndarray:
    """
    Performs vectorized calculation of options' implied volatility used by SABR model. For details,
    please see https://www.researchgate.net/publication/235622441_Managing_Smile_Risk. Here we
    assume zero dividends as cryptocurrency does not provide it normally.

    Parameters
    ----------
    S : Underlying Prices of shape (n,)
    K : Strikes of shape (n,)
    T : Time To Maturities of shape (n,)
    alpha : Alpha parameter of SABR model of shape(m,)
    rho : Rho parameter of SABR model of shape(m,)
    nu : Nu parameter of SABR model of shape(m,)
    beta : Beta parameter of SABR model of shape(m,)

    Returns an array of shape (m, n)
    """
    alpha_b, rho_b, nu_b = alpha[:, None], rho[:, None], nu[:, None]
    S_b, K_b, T_b = S[None, :], K[None, :], T[None, :]
    F_b = S_b * np.exp(r * T_b)
    lnFK = np.log(F_b / K_b)
    Fpow = F_b ** (1.0 - beta)
    FKpow = (F_b * K_b) ** ((1.0 - beta) / 2.0)
    lnFK2 = lnFK * lnFK

    z = (nu_b / alpha_b) * FKpow * lnFK
    xz = np.log((np.sqrt(1.0 - 2.0 * rho_b * z + z * z) + z - rho_b) / (1.0 - rho_b))

    one_minus_beta_sq = (1.0 - beta) * (1.0 - beta)

    A = alpha_b / (
        FKpow
        * (
            1.0
            + (one_minus_beta_sq / 24.0) * lnFK2
            + (one_minus_beta_sq * one_minus_beta_sq / 1920.0) * (lnFK2 * lnFK2)
        )
    )

    correction_T_atm = (
        (one_minus_beta_sq * alpha_b * alpha_b) / (24.0 * (Fpow * Fpow))
        + (rho_b * beta * nu_b * alpha_b) / (4.0 * Fpow)
        + (nu_b * nu_b) * (2.0 - 3.0 * rho_b * rho_b) / 24.0
    )
    sigma_atm = alpha_b / Fpow * (1.0 + correction_T_atm * T_b)

    correction_T = (
        (one_minus_beta_sq * alpha_b * alpha_b) / (24.0 * (FKpow * FKpow))
        + (rho_b * beta * nu_b * alpha_b) / (4.0 * FKpow)
        + (nu_b * nu_b) * (2.0 - 3.0 * rho_b * rho_b) / 24.0
    )

    F_eq_K_mask = np.abs(F_b - K_b) < settings.ppl.epsilon
    ratio = np.where(F_eq_K_mask, 1.0, z / xz)

    return np.where(F_eq_K_mask, sigma_atm, A * ratio * (1.0 + correction_T * T_b))


class SABR(GridSearchModel, SpotCautiousModel):
    def __init__(self, settings):
        super().__init__()
        # stores current best parameters as an array of shape (p,) in the same order as gs_params().params_details
        self._params: Optional[np.ndarray] = None
        self.settings = settings
        self.beta = settings.sabr.beta

    def calibrate_spot_cautious_params(self, S: np.ndarray, r: float) -> None:
        """
        Fits the Beta parameter using historical spot prices.
        
        According to Hagan et al., beta defines the backbone of the volatility surface:
        ln(sigma_atm) ~ ln(alpha) - (1 - beta) * ln(F)
        
        We split the spot history S into chunks to estimate realized volatility and average price
        for each chunk, then perform a log-log regression.
        """
        n_points = S.size
        # 1. Define chunk size to calculate localized volatility
        # Assuming S is high-frequency (e.g. 1s data), we need reasonable windows (e.g., 5-10 mins)
        # We aim for at least 10-20 points for regression.
        n_chunks = 20
        chunk_size = n_points // n_chunks
        
        log_prices = []
        log_vols = []

        # Calculate Log Returns over the whole series
        log_returns = np.diff(np.log(S))

        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size
            
            # Price level for this chunk (Forward approx as Spot for short duration)
            chunk_S = S[start:end]
            avg_price = np.mean(chunk_S)
            
            # Volatility for this chunk (std dev of log returns)
            # We don't need to annualize strictly for the slope calculation, 
            # as scaling factor is a constant shift in log-log space.
            chunk_ret = log_returns[start : end - 1]
            if chunk_ret.size < 2:
                continue
                
            realized_vol = np.std(chunk_ret)

            if realized_vol > settings.ppl.epsilon and avg_price > settings.ppl.epsilon:
                log_prices.append(np.log(avg_price))
                log_vols.append(np.log(realized_vol))

        # 2. Linear Regression: Y = Slope * X + Intercept
        # ln(vol) = -(1 - beta) * ln(F) + ln(alpha)
        # Slope = beta - 1  =>  beta = 1 + Slope
        slope, _, _, _, _ = linregress(log_prices, log_vols)
        
        estimated_beta = 1.0 + slope

        # 3. Clamp Beta
        # Hagan notes beta is typically between 0 (Normal) and 1 (LogNormal) [cite: 423]
        # Though it can technically be outside, for stability we usually clip it.
        self.beta = np.clip(estimated_beta, 0.0, 1.0)

    def gs_params(self) -> GSModelParams:
        return self.settings.sabr

    def _prices_for_param_grid(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, param_matrix: np.ndarray, r: float
    ) -> np.ndarray:
        S_b, K_b, T_b, is_call_b = S[None, :], K[None, :], T[None, :], is_call[None, :]
        alpha, rho, nu = param_matrix[:, 0], param_matrix[:, 1], param_matrix[:, 2]
        sigmas = _sabr_implied_vol(S_b, K_b, T_b, alpha, self.beta, rho, nu, r)
        prices = prices_for_sigmas(sigmas, S_b, K_b, T_b, is_call_b, r)
        return prices
