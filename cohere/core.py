"""Linear algebra and pairwise-coherence scans used by CoHeRe (torch; GPU if available)."""
from __future__ import annotations

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _t(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(DEV, dtype)
    return torch.as_tensor(np.asarray(x), device=DEV, dtype=dtype)


# --------------------------------------------------------------------------- #
# Scatter estimation and whitening
# --------------------------------------------------------------------------- #
def tyler_shape(Xc, iters=500, tol=1e-9, reg=0.0):
    """Tyler's M-estimator of shape for centred data (trace-normalised to d).

    Fixed point  S = (d/n) sum_i x_i x_i^T / (x_i^T S^{-1} x_i).
    Optional ridge `reg` gives the regularised Tyler estimator (Pascal-Chitour-Quek /
    Couillet-McKay) which exists also when d is comparable to n. The unregularised estimator does
    not exist when too many points lie close to a low-dimensional subspace (e.g. a large, tight
    poison cluster in very anisotropic features); the iteration then heads to a singular matrix, and we
    restart it with a small ridge (0.05), for which the fixed point always exists.
    """
    n, d = Xc.shape
    eye = torch.eye(d, device=Xc.device, dtype=Xc.dtype)
    S = eye.clone()
    for _ in range(iters):
        L, info = torch.linalg.cholesky_ex(S)
        if info.item() != 0 or not torch.isfinite(S).all():
            if reg >= 0.05:
                raise RuntimeError("regularised Tyler iteration failed")
            return tyler_shape(Xc, iters=iters, tol=tol, reg=0.05)
        Z = torch.linalg.solve_triangular(L, Xc.T, upper=False)  # L^{-1} x_i
        q = (Z * Z).sum(0).clamp_min(1e-300)                      # x^T S^{-1} x
        S_new = (d / n) * (Xc.T / q) @ Xc
        if reg > 0:
            S_new = (1 - reg) * S_new + reg * eye
        S_new = S_new * (d / torch.trace(S_new))
        if (S_new - S).norm() <= tol * S.norm():
            S = S_new
            break
        S = S_new
    return S


def inv_sqrt_psd(S, floor=1e-10):
    lam, V = torch.linalg.eigh(S)
    lam = lam.clamp_min(floor * lam.max())
    return (V / lam.sqrt()) @ V.T


def whiten(X, method="tyler", center="mean", reg=0.0):
    """Return whitened data Y (n x d) as a float64 torch tensor.

    method: 'tyler' (default), 'sample' or 'none';  center: 'mean' or 'none'.
    """
    X = _t(X)
    Xc = X - X.mean(0, keepdim=True) if center == "mean" else X
    n = Xc.shape[0]
    if method == "none":
        return Xc
    if method == "sample":
        S = Xc.T @ Xc / n
    elif method == "tyler":
        S = tyler_shape(Xc, reg=reg)
    else:
        raise ValueError(method)
    return Xc @ inv_sqrt_psd(S)


def unit_rows(Y):
    return Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-300)


# --------------------------------------------------------------------------- #
# Pairwise cosine scans (blocked; the Gram matrix is never stored)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def row_standardized_topk(U, m=128, block=1024):
    """Global row null: z_ij = (c_ij - med_i) / (1.4826 MAD_i) with med/MAD over all j != i.

    Returns the top-m z-scores per row (descending), their column indices, and (med, mad).
    Under the elliptical null with oracle whitening, med_i -> 0 and 1.4826 MAD_i -> d^{-1/2},
    so z_ij reduces to sqrt(d) c_ij; in general it removes a row-specific location-scale
    nuisance (local density heterogeneity).
    """
    U32 = U.to(torch.float32)
    n = U32.shape[0]
    m = min(m, n - 1)
    Z = torch.empty((n, m), device=U.device)
    I = torch.empty((n, m), device=U.device, dtype=torch.long)
    med = torch.empty(n, device=U.device)
    mad = torch.empty(n, device=U.device)
    for s in range(0, n, block):
        e = min(n, s + block)
        G = U32[s:e] @ U32.T
        ar = torch.arange(e - s, device=U.device)
        G[ar, torch.arange(s, e, device=U.device)] = float("nan")
        md = torch.nanmedian(G, dim=1).values
        md_ = torch.nanmedian((G - md[:, None]).abs(), dim=1).values * 1.4826
        Zb = (G - md[:, None]) / md_[:, None].clamp_min(1e-12)
        Zb[ar, torch.arange(s, e, device=U.device)] = -float("inf")
        v, j = torch.topk(Zb, m, dim=1)
        Z[s:e], I[s:e], med[s:e], mad[s:e] = v, j, md, md_
    return Z, I, med, mad


@torch.no_grad()
def row_local_topk(U, m=64, w_lo=32, w_hi=512, block=1024):
    """Local row null: standardise the top-m cosines of each row by the location/scale of the
    same row's order statistics of ranks w_lo < t <= w_hi (slice [w_lo:w_hi] of the descending
    row, self excluded).

    Robust to mixture (cluster) structure: the window lies inside the row's own natural
    cluster, so a clean point in a dense cluster is compared with its cluster-mates, while
    a poison (k-1 <= w_lo coherent partners) stands out against its former neighbourhood.
    Returns (Z, I) with Z the standardised top-m scores (descending).
    """
    U32 = U.to(torch.float32)
    n = U32.shape[0]
    w_hi = min(w_hi, n - 1)
    Z = torch.empty((n, m), device=U.device)
    I = torch.empty((n, m), device=U.device, dtype=torch.long)
    for s in range(0, n, block):
        e = min(n, s + block)
        G = U32[s:e] @ U32.T
        ar = torch.arange(e - s, device=U.device)
        G[ar, torch.arange(s, e, device=U.device)] = -2.0
        v, j = torch.topk(G, w_hi, dim=1)
        win = v[:, w_lo:w_hi]
        loc = win.median(dim=1).values
        sc = (win - loc[:, None]).abs().median(dim=1).values * 1.4826
        Z[s:e] = (v[:, :m] - loc[:, None]) / sc[:, None].clamp_min(1e-9)
        I[s:e] = j[:, :m]
    return Z, I


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
def bh(p, q):
    """Benjamini-Hochberg rejections at level q (boolean mask)."""
    p = np.asarray(p)
    n = p.size
    order = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = np.nonzero(p[order] <= thr)[0]
    rej = np.zeros(n, dtype=bool)
    if ok.size:
        rej[order[: ok.max() + 1]] = True
    return rej
