import numpy as np

from src.models.abc.gridsearch_model import GridSearchModel, GSModelParams
from src.models.black_sholes import norm_cdf
from src.utils.config import settings


def _sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Vectorized SABR implied volatility calculation.
    
    Parameters
    ----------
    F, K, T : Arrays of shape (1, n)
    alpha, rho, nu : Arrays of shape (m, 1)
    beta : float
    
    Returns an array of shape (m, n)
    """
    lnFK = np.log(F / K)
    Fpow = F ** (1.0 - beta)
    FKpow = (F * K) ** ((1.0 - beta) / 2.0)
    lnFK2 = lnFK * lnFK

    z = (nu / alpha) * FKpow * lnFK
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

    one_minus_beta_sq = (1.0 - beta) * (1.0 - beta)

    A = alpha / (
        FKpow
        * (
            1.0
            + (one_minus_beta_sq / 24.0) * lnFK2
            + (one_minus_beta_sq * one_minus_beta_sq / 1920.0) * (lnFK2 * lnFK2)
        )
    )

    correction_T_atm = (
        (one_minus_beta_sq * alpha * alpha) / (24.0 * (Fpow * Fpow))
        + (rho * beta * nu * alpha) / (4.0 * Fpow)
        + (nu * nu) * (2.0 - 3.0 * rho * rho) / 24.0
    )
    sigma_atm = alpha / Fpow * (1.0 + correction_T_atm * T)

    correction_T = (
        (one_minus_beta_sq * alpha * alpha) / (24.0 * (FKpow * FKpow))
        + (rho * beta * nu * alpha) / (4.0 * FKpow)
        + (nu * nu) * (2.0 - 3.0 * rho * rho) / 24.0
    )

    F_eq_K_mask = np.abs(F - K) < settings.ppl.epsilon
    ratio = np.where(F_eq_K_mask, 1.0, z / xz)

    return np.where(F_eq_K_mask, sigma_atm, A * ratio * (1.0 + correction_T * T))


def _prices_from_vol_matrix(vol_matrix, S, K, T, is_call, r):
    """
    Prices options using Black-Scholes formula with pre-calculated volatility matrix.
    
    vol_matrix : (m, n)
    S, K, T, is_call : (1, n) inputs broadcasted inside
    """
    sqrtT = np.sqrt(T)

    d1 = (np.log(S / K) + (r + 0.5 * vol_matrix**2) * T) / (vol_matrix * sqrtT)
    d2 = d1 - vol_matrix * sqrtT

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)
    Nnegd1, Nnegd2 = 1.0 - Nd1, 1.0 - Nd2
    disc = np.exp(-r * T)
    
    call = S * Nd1 - K * disc * Nd2
    put = K * disc * Nnegd2 - S * Nnegd1
    
    return np.where(is_call, call, put)


class SABR(GridSearchModel):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.beta = settings.sabr.beta

    def gs_params(self) -> GSModelParams:
        return self.settings.sabr

    def _prices_for_param_grid(
        self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, param_matrix: np.ndarray, r: float
    ) -> np.ndarray:
        S_b, K_b, T_b, is_call_b = S[None, :], K[None, :], T[None, :], is_call[None, :]
        alphas, rhos, nus = param_matrix[:, 0][:, None], param_matrix[:, 1][:, None], param_matrix[:, 2][:, None]
        sigmas = _sabr_implied_vol(S_b, K_b, T_b, alphas, self.beta, rhos, nus)
        prices = _prices_from_vol_matrix(sigmas, S_b, K_b, T_b, is_call_b, r)
        return prices
