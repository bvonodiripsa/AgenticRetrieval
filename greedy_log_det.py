"""Greedy log-det diversity selection algorithm."""

import numpy as np


def greedy_log_det_select(vectors: np.ndarray, query_vec: np.ndarray, k: int,
                          eta: float = 0.0, rescale_power: float = 0.0) -> list[int]:
    """Greedily select k indices maximizing log-det(Gram) for diversity."""
    V = vectors.copy()
    if rescale_power > 0:
        sims = V @ query_vec
        for i in range(len(V)):
            V[i] *= (sims[i] ** rescale_power) if sims[i] > 0 else 0
    n = len(V)
    if k >= n:
        return list(range(n))
    if eta == 0.0:
        chosen = []
        R = V.copy()                              # residual vectors
        scores = np.sum(R * R, axis=1)            # ||R[j]||^2
        for _ in range(k):
            best_i = int(np.argmax(scores))
            chosen.append(best_i)
            r_norm = np.sqrt(scores[best_i])
            if r_norm < 1e-12:
                break
            q = R[best_i] / r_norm                # new orthonormal basis vector
            projections = R @ q                    # q^T @ R[j] for all j
            R -= np.outer(projections, q)          # R[j] -= q * (q^T @ R[j])
            scores = np.sum(R * R, axis=1)
            for idx in chosen:
                scores[idx] = -np.inf
    else:
        # eta > 0: Woodbury-based incremental update
        # B starts as (1/eta)*I, z[j] = B @ v[j], score[j] = v[j]^T @ z[j]
        chosen = []
        Z = V / eta                                # z[j] = (1/eta) * v[j]
        scores = np.sum(V * Z, axis=1)             # score[j] = v[j]^T @ z[j]
        for _ in range(k):
            best_i = int(np.argmax(scores))
            chosen.append(best_i)
            u = Z[best_i].copy()
            denom = 1.0 + scores[best_i]
            if abs(denom) < 1e-30:
                break
            # v[j]^T @ u for all j
            vtu = V @ u                            # (n,)
            coeffs = vtu / denom                   # (n,)
            Z -= np.outer(coeffs, u)               # z[j] -= coeff[j] * u
            scores = np.sum(V * Z, axis=1)         # score[j] = v[j]^T @ z[j]
            for idx in chosen:
                scores[idx] = -np.inf
    return chosen
