from math import sqrt
from scipy.special import erf
import numpy as np
from src.utils.config import settings

bs = settings.bs

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normalized_sse(close, preds_matrix):
    diffs = preds_matrix - close[None, :]
    return np.sum(diffs * diffs, axis=1) / np.sum(close * close)


def _prices_for_sigmas(S, K, T, is_call, sigmas, r=settings.ppl.risk_free_rate):
    S_b, K_b, T_b, is_call_b, sig_b = S[None, :], K[None, :], T[None, :], is_call[None, :], sigmas[:, None]
    sqrtT = np.sqrt(T_b)

    d1 = (np.log(S_b / K_b) + (r + 0.5 * (sig_b**2)) * T_b) / (sig_b * sqrtT)
    d2 = d1 - sig_b * sqrtT

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)
    Nnegd1, Nnegd2 = 1.0 - Nd1, 1.0 - Nd2
    disc = np.exp(-r * T_b)
    call, put = S_b * Nd1 - K_b * disc * Nd2, K_b * disc * Nnegd2 - S_b * Nnegd1
    return np.where(is_call_b, call, put)


def _compute_scores_for_sigmas(sigmas, S, K, T, is_call, close, max_chunk_elems=300_000_000):
    m = sigmas.size
    m_chunk = max(1, max_chunk_elems // S.size)
    scores = np.empty(m, dtype=float)
    for start in range(0, m, m_chunk):
        stop = min(m, start + m_chunk)
        prices = _prices_for_sigmas(S, K, T, is_call, sigmas[start:stop])
        scores_chunk = normalized_sse(close, prices)
        scores[start:stop] = scores_chunk

    return scores


class BlackScholes:
    def __init__(self):
        self.sigma = None

    def price(self, df):
        S, K, T, is_call = df["underlying_price"].values, df["strike"].values, df["ttm"].values, df["is_call"].values
        return _prices_for_sigmas(S, K, T, is_call, np.atleast_1d(self.sigma))[0]

    def find_initial_params(self, df):
        S, K, T = df["underlying_price"].values, df["strike"].values, df["ttm"].values
        is_call, close = df["is_call"].values, df["close"].values

        low, high = bs.sigma_min, bs.sigma_max
        best_sigma, best_score = None, float("inf")

        for _ in range(bs.max_refines_initial):
            sigmas = np.linspace(low, high, bs.points_initial)
            scores = _compute_scores_for_sigmas(sigmas, S, K, T, is_call, close)
            idx = np.argmin(scores)
            if scores[idx] < best_score - settings.ppl.epsilon:
                best_score, best_sigma = scores[idx], sigmas[idx]
            else:
                break

            width = (high - low) / bs.refine_factor
            low = max(bs.sigma_min, best_sigma - width / 2.0)
            high = min(bs.sigma_max, best_sigma + width / 2.0)

        self.sigma = best_sigma

    def calibrate(self, df):
        S, K, T = df["underlying_price"].values, df["strike"].values, df["ttm"].values
        is_call, close = df["is_call"].values, df["close"].values

        best_sigma = self.sigma
        prices = _prices_for_sigmas(S, K, T, is_call, np.atleast_1d(self.sigma))
        best_score = float(normalized_sse(close, prices))
        radius, points = bs.radius, bs.points_calibrate

        for _ in range(bs.max_refines_calibrate):
            left, right = max(bs.sigma_min, self.sigma - radius), min(bs.sigma_max, self.sigma + radius)
            sigmas = np.linspace(left, right, points)
            scores = _compute_scores_for_sigmas(sigmas, S, K, T, is_call, close)
            idx = int(np.argmin(scores))

            if scores[idx] < best_score - settings.ppl.epsilon:
                best_score, best_sigma = scores[idx], sigmas[idx]
                radius, points = radius * bs.radius_calibrate_factor, int(points * bs.radius_calibrate_factor)
            else:
                break

        self.sigma = best_sigma
