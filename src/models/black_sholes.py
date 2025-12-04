from math import sqrt

import numpy as np
import pandas as pd
from scipy.special import erf

from src.models.abc.model import Model
from src.utils.config import settings


def norm_cdf(x):
    """
    Performs vectorized calculation of standard normal cumulative distribution
    function for each x.
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normalized_sse(close, preds_matrix):
    """
    Performs normal deviation from the actual Close Prices.

    Parameters
    ----------
    close : ndarray of shape (n,)
        Actual Close prices
    preds_matrix : ndarray of shape (m, n)
        A matrix with every row representing predicted prices

    Returns
    -------
    np.ndarray of shape (m,)
        An array with every value being a normalized deviation of corresponding m-th row
        in predicted prices relative to close
    """
    diffs = preds_matrix - close[None, :]
    return np.sum(diffs * diffs, axis=1) / np.sum(close * close)


def _prices_for_sigmas(S, K, T, is_call, sigmas, r):
    """
    Performs vectorized pricing via Black-Sholes analytical formula. For details,
    please see https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model. Here we
    assume zero dividends as cryptocurrency does not provide it normally.

    Parameters
    ----------
    S : ndarray of shape (n,)
        Underlying Prices
    K : ndarray of shape (n,)
        Strikes
    T : ndarray of shape (n,)
        Time To Maturities
    is_call : ndarray of shape (n,)
        The option type flag - Put or Call
    sigmas : ndarray of shape (m,)
        Sigmas under consideration
    r: float
        Current Interest Rate

    Returns
    -------
    np.ndarray of shape (m, n)
        An array where every row is a price vector predicted with Black-Sholes formula
    """
    S_b, K_b, T_b, is_call_b, sig_b = S[None, :], K[None, :], T[None, :], is_call[None, :], sigmas[:, None]
    sqrtT = np.sqrt(T_b)

    S_div_K = S_b / K_b
    S_div_K = S_div_K + np.where(S_div_K > 1, settings.ppl.epsilon, -settings.ppl.epsilon)
    # Both d1, d2 are of shape (m, n)
    d1 = (np.log(S_div_K) + (r + 0.5 * (sig_b**2)) * T_b) / (sig_b * sqrtT)
    d2 = d1 - sig_b * sqrtT

    Nd1, Nd2 = norm_cdf(d1) + settings.ppl.epsilon, norm_cdf(d2) + settings.ppl.epsilon
    Nnegd1, Nnegd2 = 1.0 - Nd1, 1.0 - Nd2
    disc = np.exp(-r * T_b)
    call, put = S_b * Nd1 - K_b * disc * Nd2, K_b * disc * Nnegd2 - S_b * Nnegd1
    return np.where(is_call_b, call, put)


def _compute_scores_for_sigmas(S, K, T, is_call, close, sigmas, r, max_chunk_elems=300_000_000):
    """
    Calls _prices_for_sigmas subsequently in chunks, guarantying n * m < max_chunk_elems.

    Parameters
    ----------
    S : ndarray of shape (n,)
        Underlying Prices
    K : ndarray of shape (n,)
        Strikes
    T : ndarray of shape (n,)
        Time To Maturities
    is_call : ndarray of shape (n,)
        The option type flag - Put or Call
    close: ndarray of shape (n,)
        Actual Close prices
    sigmas : ndarray of shape (m,)
        Sigmas under consideration

    Returns
    -------
    np.ndarray of shape (m,)
        Normalized SSE calculated for each sigma
    """
    m, m_chunk = sigmas.size, max(1, max_chunk_elems // S.size)
    scores = np.empty(m)
    for start in range(0, m, m_chunk):
        stop = min(m, start + m_chunk)
        prices = _prices_for_sigmas(S, K, T, is_call, sigmas[start:stop], r)
        scores_chunk = normalized_sse(close, prices)
        scores[start:stop] = scores_chunk

    return scores


class BlackScholes(Model):
    def __init__(self, settings):
        self.bs, self.sigma = settings.bs, None

    def price(self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, r: float) -> np.ndarray:
        return _prices_for_sigmas(S, K, T, is_call, np.atleast_1d(self.sigma), r)[0]

    def find_initial_params(self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float) -> None:
        low, high = self.bs.sigma_min, self.bs.sigma_max
        best_sigma, best_score = None, float("inf")

        for _ in range(self.bs.max_refines_initial):
            sigmas = np.linspace(low, high, self.bs.points_initial)
            scores = _compute_scores_for_sigmas(S, K, T, is_call, close, sigmas, r)
            idx = np.argmin(scores)
            if scores[idx] < best_score - settings.ppl.epsilon:
                best_score, best_sigma = scores[idx], sigmas[idx]
            else:
                break

            width = (high - low) / self.bs.refine_factor
            low = max(self.bs.sigma_min, best_sigma - width / 2.0)
            high = min(self.bs.sigma_max, best_sigma + width / 2.0)

        self.sigma = best_sigma

    def calibrate(self, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, close: np.ndarray, r: float) -> None:
        best_sigma = self.sigma
        prices = _prices_for_sigmas(S, K, T, is_call, np.atleast_1d(self.sigma), r)
        best_score = float(normalized_sse(close, prices))
        radius, points = self.bs.radius, self.bs.points_calibrate

        for _ in range(self.bs.max_refines_calibrate):
            left, right = max(self.bs.sigma_min, self.sigma - radius), min(self.bs.sigma_max, self.sigma + radius)
            sigmas = np.linspace(left, right, points)
            scores = _compute_scores_for_sigmas(S, K, T, is_call, close, sigmas)
            idx = int(np.argmin(scores))

            if scores[idx] < best_score - settings.ppl.epsilon:
                best_score, best_sigma = scores[idx], sigmas[idx]
                radius, points = radius * self.bs.radius_calibrate_factor, int(points * self.bs.radius_calibrate_factor)
            else:
                break

        self.sigma = best_sigma
