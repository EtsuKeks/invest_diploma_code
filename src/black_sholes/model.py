from math import sqrt
from scipy.special import erf
import numpy as np
from src.utils.config import settings


def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normalized_sse_vectorized(close, preds_matrix):
    diffs = preds_matrix - close[None, :]
    return np.sum(diffs * diffs, axis=1) / np.sum(close * close)


def _price_matrix_for_sigmas(S, K, T, is_call, sigmas, r):
    S_b, K_b, T_b, is_call_b, sig_b = S[None, :], K[None, :], T[None, :], is_call[None, :], sigmas[:, None]
    sqrtT = np.sqrt(T_b)

    d1 = (np.log(S_b / K_b) + (r + 0.5 * (sig_b**2)) * T_b) / (sig_b * sqrtT)
    d2 = d1 - sig_b * sqrtT

    Nd1, Nd2, Nnegd1, Nnegd2 = _norm_cdf(d1), _norm_cdf(d2), _norm_cdf(-d1), _norm_cdf(-d2)
    disc = np.exp(-r * T_b)
    call, put = S_b * Nd1 - K_b * disc * Nd2, K_b * disc * Nnegd2 - S_b * Nnegd1
    return np.where(is_call_b, call, put)


def compute_scores_for_sigmas(sigmas, S, K, T, is_call, close, r, max_chunk_elems=300_000_000):
    m = sigmas.size
    m_chunk = max(1, max_chunk_elems // S.size)
    scores = np.empty(m, dtype=float)
    for start in range(0, m, m_chunk):
        stop = min(m, start + m_chunk)
        preds_chunk = _price_matrix_for_sigmas(S, K, T, is_call, sigmas[start:stop], r)
        scores_chunk = _normalized_sse_vectorized(close, preds_chunk)
        scores[start:stop] = scores_chunk

    return scores


class BlackScholes:
    def __init__(self):
        self.sigma = None

    def price(self, df):
        S, K, T, is_call = df["underlying_price"].values, df["strike"].values, df["ttm"].values, df["is_call"].values
        return _price_matrix_for_sigmas(S, K, T, is_call, np.atleast_1d(self.sigma), settings.ppl.risk_free_rate)[0]

    def find_initial_params(self, df):
        S, K, T = df["underlying_price"].values, df["strike"].values, df["ttm"].values
        is_call, close = df["is_call"].values, df["close"].values

        low, high = settings.bs.sigma_min, settings.bs.sigma_max
        best_sigma, best_score = None, float("inf")

        for _ in range(settings.bs.max_refines_initial):
            sigmas = np.linspace(low, high, settings.bs.points_initial)
            scores = compute_scores_for_sigmas(sigmas, S, K, T, is_call, close, settings.ppl.risk_free_rate)
            idx = np.argmin(scores)
            if scores[idx] < best_score - settings.bs.epsilon:
                best_score, best_sigma = scores[idx], sigmas[idx]
            else:
                break

            width = (high - low) / settings.bs.refine_factor
            low = max(settings.bs.sigma_min, best_sigma - width / 2.0)
            high = min(settings.bs.sigma_max, best_sigma + width / 2.0)

        self.sigma = best_sigma

    def calibrate(self, df):
        S, K, T = df["underlying_price"].values, df["strike"].values, df["ttm"].values
        is_call, close = df["is_call"].values, df["close"].values

        best_sigma = self.sigma
        prices = _price_matrix_for_sigmas(S, K, T, is_call, np.atleast_1d(self.sigma), settings.ppl.risk_free_rate)
        best_score = _normalized_sse_vectorized(close, prices)[0]
        radius, points = settings.bs.radius, settings.bs.points_calibrate

        for _ in range(settings.bs.max_refines_calibrate):
            left = max(settings.bs.sigma_min, self.sigma - radius)
            right = min(settings.bs.sigma_max, self.sigma + radius)
            sigmas = np.linspace(left, right, points)
            scores = compute_scores_for_sigmas(sigmas, S, K, T, is_call, close, settings.ppl.risk_free_rate)
            idx = int(np.argmin(scores))

            if scores[idx] < best_score - settings.bs.epsilon:
                best_score, best_sigma = scores[idx], sigmas[idx]
                radius = min(settings.bs.sigma_max, radius * settings.bs.radius_calibrate_factor)
                points = int(points * settings.bs.radius_calibrate_factor)
            else:
                break

        self.sigma = best_sigma
