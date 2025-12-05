from typing import Optional

import numpy as np
from scipy.stats import linregress, norm

from src.models.abc.gridsearch_model import GridSearchModel, GSModelParams
from src.models.black_sholes import prices_for_sigmas
from src.utils.config import settings


def _implied_vol_newton(S, K, T, is_call, price, r, tol=1e-5, max_iter=20):
    """
    Calculates Black-Scholes implied volatility using Newton-Raphson method.
    Optimized for vector inputs.
    """
    sigma = np.full_like(S, 0.5)
    
    valid_mask = (price > 0) & (S > 0) & (K > 0) & (T > 0)
    
    S_v = S[valid_mask]
    K_v = K[valid_mask]
    T_v = T[valid_mask]
    r_v = np.full_like(S_v, r) if np.isscalar(r) else r[valid_mask]
    price_v = price[valid_mask]
    is_call_v = is_call[valid_mask]
    sigma_v = sigma[valid_mask]
    
    sqrt_T = np.sqrt(T_v)
    
    for _ in range(max_iter):
        d1 = (np.log(S_v / K_v) + (r_v + 0.5 * sigma_v**2) * T_v) / (sigma_v * sqrt_T)
        d2 = d1 - sigma_v * sqrt_T
        
        vega = S_v * norm.pdf(d1) * sqrt_T
        
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        
        call_price = S_v * nd1 - K_v * np.exp(-r_v * T_v) * nd2
        put_price = K_v * np.exp(-r_v * T_v) * (1 - nd2) - S_v * (1 - nd1)
        
        model_price = np.where(is_call_v, call_price, put_price)
        
        diff = model_price - price_v
        
        if np.max(np.abs(diff)) < tol:
            break
            
        vega = np.where(vega < 1e-8, 1e-8, vega)
        
        sigma_v = sigma_v - diff / vega
    
    result = np.full_like(S, np.nan)
    result[valid_mask] = sigma_v
    return result


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


class SABR(GridSearchModel):
    def __init__(self, settings):
        super().__init__()
        # stores current best parameters as an array of shape (p,) in the same order as gs_params().params_details
        self._params: Optional[np.ndarray] = None
        self.settings = settings
        self.beta = settings.sabr.beta

    def calibrate_spot_cautious_params(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float
    ) -> None:
        """
        Calibrates the Beta parameter using the correlation between ATM Implied Volatility and Forward Price.
        According to Hagan et al. (3.3), slope of ln(vol) vs ln(F) is -(1 - beta).
        """
        F = S * np.exp(r * T)

        moneyness = np.abs(F / K - 1.0)
        atm_mask = moneyness < 0.05
        
        if np.sum(atm_mask) < 10:
            return

        S_atm = S[atm_mask]
        K_atm = K[atm_mask]
        T_atm = T[atm_mask]
        close_atm = close[atm_mask]
        is_call_atm = is_call[atm_mask]
        F_atm = F[atm_mask]

        implied_vols = _implied_vol_newton(S_atm, K_atm, T_atm, is_call_atm, close_atm, r)

        valid_iv_mask = ~np.isnan(implied_vols) & (implied_vols > 0)
        
        if np.sum(valid_iv_mask) < 5:
            return

        y = np.log(implied_vols[valid_iv_mask])
        x = np.log(F_atm[valid_iv_mask])

        if np.var(x) < settings.ppl.epsilon:
            return

        slope, _, _, _, _ = linregress(x, y)

        estimated_beta = 1.0 + slope

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
