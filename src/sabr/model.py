import numpy as np
from src.utils.config import settings
from src.black_sholes.model import normalized_sse, norm_cdf

sb = settings.sabr


def _sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    lnFK = np.log(F / K)
    Fpow = F ** (1.0 - beta)
    FKpow = (F * K) ** ((1.0 - beta) / 2.0)
    lnFK2 = lnFK * lnFK

    z = (nu / alpha) * FKpow * lnFK
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

    one_minus_beta_sq = (1.0 - beta) * (1.0 - beta)

    A = alpha / (FKpow * (
        1.0 + (one_minus_beta_sq / 24.0) * lnFK2 + (one_minus_beta_sq * one_minus_beta_sq / 1920.0) * (lnFK2 * lnFK2)
    ))

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


def _prices_for_sigma_matrix(sig_matrix, S, K, T, is_call, r=settings.ppl.risk_free_rate):
    S_b, K_b, T_b, is_call_b = S[None, :], K[None, :], T[None, :], is_call[None, :]
    sqrtT = np.sqrt(T_b)

    d1 = (np.log(S_b / K_b) + (r + 0.5 * sig_matrix * sig_matrix) * T_b) / (sig_matrix * sqrtT)
    d2 = d1 - sig_matrix * sqrtT

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)
    Nnegd1, Nnegd2 = 1.0 - Nd1, 1.0 - Nd2
    disc = np.exp(-r * T_b)
    call, put = S_b * Nd1 - K_b * disc * Nd2, K_b * disc * Nnegd2 - S_b * Nnegd1
    return np.where(is_call_b, call, put)


def _compute_scores_for_params(alphas, rhos, nus, F, K, T, is_call, close, beta, max_chunk_elems=300_000_000):
    M, cs = alphas.size, max(1, int((max_chunk_elems / F.size) ** (1.0/3.0)))
    F_row, K_row, T_row = F[None, :], K[None, :], T[None, :]
    scores = np.empty(M ** 3, dtype=float)
    for i in range(0, M, cs):
        di = min(cs, M - i)
        for j in range(0, M, cs):
            dj = min(cs, M - j)
            for k in range(0, M, cs):
                dk = min(cs, M - k)
                A, R, N = np.meshgrid(alphas[i:i+di], rhos[j:j+dj], nus[k:k+dk], indexing='ij')
                alphas_flat, rhos_flat, nus_flat = A.ravel(), R.ravel(), N.ravel()
                sig_matrix = _sabr_implied_vol(
                    F_row, K_row, T_row, alphas_flat[:, None], beta, rhos_flat[:, None], nus_flat[:, None]
                )

                price_matrix = _prices_for_sigma_matrix(sig_matrix, F, K, T, is_call)
                scores_chunk = normalized_sse(close, price_matrix)
                Iidx = np.arange(i, i + di, dtype=np.int64)[:, None, None]
                Jidx = np.arange(j, j + dj, dtype=np.int64)[None, :, None]
                Kidx = np.arange(k, k + dk, dtype=np.int64)[None, None, :]
                linear_idx_block = ((Iidx * (M * M)) + (Jidx * M) + Kidx).ravel()
                scores[linear_idx_block] = scores_chunk

    return scores.reshape((M, M, M))


class SABR:
    def __init__(self):
        self.alpha, self.rho, self.nu = None, None, None
        self.beta = sb.beta

    def price(self, df):
        S, K, T, is_call = df["underlying_price"].values, df["strike"].values, df["ttm"].values, df["is_call"].values
        sigmas = _sabr_implied_vol(S[None, :], K[None, :], T[None, :],
                                       np.atleast_1d(self.alpha)[:, None],
                                       self.beta,
                                       np.atleast_1d(self.rho)[:, None],
                                       np.atleast_1d(self.nu)[:, None])[0, 0, 0]
        return _prices_for_sigma_matrix(sigmas[None, :], S, K, T, is_call)[0, 0, 0]

    def find_initial_params(self, df):
        S, K, T = df["underlying_price"].values, df["strike"].values, df["ttm"].values
        is_call, close = df["is_call"].values, df["close"].values

        a_low, a_high = sb.alpha_min, sb.alpha_max
        r_low, r_high = sb.rho_min, sb.rho_max
        n_low, n_high = sb.nu_min, sb.nu_max
        best_alpha, best_rho, best_nu = self.alpha, self.rho, self.nu
        best_score = float("inf")

        for _ in range(sb.max_refines_initial):
            alphas = np.linspace(a_low, a_high, sb.points_initial)
            rhos = np.linspace(r_low, r_high, sb.points_initial)
            nus = np.linspace(n_low, n_high, sb.points_initial)

            scores = _compute_scores_for_params(alphas, rhos, nus, S, K, T, is_call, close, self.beta)
            i, j, k = np.unravel_index(np.argmin(scores), scores.shape)

            if scores[i, j, k] < best_score - settings.ppl.epsilon:
                best_score, best_alpha, best_rho, best_nu = scores[i, j, k], alphas[i], rhos[j], nus[k]
            else:
                break

            a_width = (a_high - a_low) / sb.refine_factor
            r_width = (r_high - r_low) / sb.refine_factor
            n_width = (n_high - n_low) / sb.refine_factor

            a_low, a_high = max(sb.alpha_min, best_alpha - a_width/2.0), min(sb.alpha_max, best_alpha + a_width/2.0)
            r_low, r_high = max(sb.rho_min, best_rho - r_width/2.0), min(sb.rho_max, best_rho + r_width/2.0)
            n_low, n_high = max(sb.nu_min, best_nu - n_width/2.0), min(sb.nu_max, best_nu + n_width/2.0)

        self.alpha, self.rho, self.nu = best_alpha, best_rho, best_nu

    def calibrate(self, df):
        S, K, T = df["underlying_price"].values, df["strike"].values, df["ttm"].values
        is_call, close = df["is_call"].values, df["close"].values

        best_alpha, best_rho, best_nu = self.alpha, self.rho, self.nu
        alphas, rhos, nus = np.atleast_1d(best_alpha), np.atleast_1d(best_rho), np.atleast_1d(best_nu)
        best_score = _compute_scores_for_params(alphas, rhos, nus, S, K, T, is_call, close, sb.beta)[0, 0, 0]
        radius, points = sb.radius, sb.points_calibrate

        for _ in range(sb.max_refines_calibrate):
            a_left, a_right = max(sb.alpha_min, best_alpha - radius), min(sb.alpha_max, best_alpha + radius)
            r_left, r_right = max(sb.rho_min, best_rho - radius), min(sb.rho_max, best_rho + radius)
            n_left, n_right = max(sb.nu_min, best_nu - radius), min(sb.nu_max, best_nu + radius)

            alphas = np.linspace(a_left, a_right, points)
            rhos = np.linspace(r_left, r_right, points)
            nus = np.linspace(n_left, n_right, points)

            scores = _compute_scores_for_params(alphas, rhos, nus, S, K, T, is_call, close, sb.beta)
            i, j, k = np.unravel_index(np.argmin(scores), scores.shape)

            if scores[i, j, k] < best_score - settings.ppl.epsilon:
                best_score, best_alpha, best_rho, best_nu = scores[i, j, k], alphas[i], rhos[j], nus[k]
                radius, points = radius * sb.radius_calibrate_factor, int(points * sb.radius_calibrate_factor)
            else:
                break

        self.alpha, self.rho, self.nu = best_alpha, best_rho, best_nu
