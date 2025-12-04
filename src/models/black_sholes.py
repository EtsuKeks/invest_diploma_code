import numpy as np
from scipy.special import erf
from math import sqrt

from src.models.abc.gridsearch_model import GridSearchModel, GSModelParams
from src.utils.config import settings


def norm_cdf(x):
    """Performs vectorized calculation of standard normal cumulative distribution function for each x."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def prices_for_sigmas(S, K, T, is_call, sigmas, r):
    """
    Performs vectorized pricing via Black-Sholes analytical formula. For details,
    please see https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model. Here we
    assume zero dividends as cryptocurrency does not provide it normally.

    Parameters
    ----------
    S : Underlying Prices of shape (n,)
    K : Strikes of shape (n,)
    T : Time To Maturities of shape (n,)
    is_call : The option type flag - Put or Call of shape (n,)
    sigmas : Volatilities of shape (m, n) - where each row repeats certain sigma n times when used
        with Black-Sholes model or sets them independently if used in general case
    r : Current Interest Rate

    Returns an array where every row is a price vector predicted with Black-Sholes formula
    """
    S_b, K_b, T_b, is_call_b = S[None, :], K[None, :], T[None, :], is_call[None, :]
    sqrtT = np.sqrt(T_b)

    S_div_K = S_b / K_b
    S_div_K = S_div_K + np.where(S_div_K > 1, settings.ppl.epsilon, -settings.ppl.epsilon)
    
    # Regardless of which shape was passed - (m, n) or (m,) - such broadcast would work just fine
    # Both d1, d2 are of shape (m, n)
    d1 = (np.log(S_div_K) + (r + 0.5 * (sigmas**2)) * T_b) / (sigmas * sqrtT)
    d2 = d1 - sigmas * sqrtT

    Nd1, Nd2 = norm_cdf(d1) + settings.ppl.epsilon, norm_cdf(d2) + settings.ppl.epsilon
    Nnegd1, Nnegd2 = 1.0 - Nd1, 1.0 - Nd2
    disc = np.exp(-r * T_b)
    call, put = S_b * Nd1 - K_b * disc * Nd2, K_b * disc * Nnegd2 - S_b * Nnegd1
    return np.where(is_call_b, call, put)


class BlackScholes(GridSearchModel):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def gs_params(self) -> GSModelParams:
        return self.settings.bs

    def _prices_for_param_grid(
            self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, param_matrix: np.ndarray, r: float
        ) -> np.ndarray:
        return prices_for_sigmas(S, K, T, is_call, param_matrix[:, 0][:, None], r)
