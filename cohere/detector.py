"""CoHeRe: coherence-based, heterogeneity-robust detection of a constant number of poisons.

Pipeline (method section of the paper; pseudocode in its Supplementary Material):
  1. centre and restrict to the numerically non-degenerate principal subspace;
  2. robust whitening by Tyler's shape matrix (ridge-regularised when d_eff/n > 0.2), or by a
     trusted clean set when one is given (trusted-clean mode);
  3. projection to the unit sphere (invariance to per-sample scale);
  4. row-wise empirical null at two scales, z_ij = (c_ij - m_i) / s_i: global (median/MAD of the
     whole row, top L=128 kept) and local (median/MAD of the partners of ranks 33..512, i.e. slice
     [32:512] of the sorted row; top 64 kept);
  5. pooled empirical tail of the standardised scores, per scale -> pair p-values (below the pooling
     threshold, the larger of the Gaussian and the pooled tail; all pair p-values floored at
     1/(10(N-1)) in row_hc);
  6. row-wise higher criticism of the pair p-values after discarding each row's best partner;
     generalised-Pareto row p-values; the two scales merged by the Bonferroni minimum;
  7. second pass: BH at level 0.5 on the first-pass coherence p-values gives the liberal set; if it
     is non-empty and has fewer than n/4 rows, re-whiten without it and repeat 3-6;
  8. spectral (QUE) and marginal (robust Mahalanobis) scores on the final whitened, non-sphered
     vectors, generalised-Pareto calibrated and merged with the coherence p-value by the Bonferroni
     minimum (out["p"], used for Benjamini-Hochberg); the ranking out["score"] uses robust
     median/MAD p-values of the same components.
In trusted-clean mode the ranking uses the family-evidence rule of the paper (_trusted_selection).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats

from . import core


def effective_subspace(X, rel=1e-8):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=core.DEV)
    Xc = Xt - Xt.mean(0, keepdim=True)
    _, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    keep = s ** 2 > rel * (s ** 2).max()
    return Xc @ Vh[keep].T


def empirical_tail(Z):
    """Upper-tail function of standardised pair scores, estimated by pooling the top-m scores of
    every row.  Returns a callable z -> P0(Z >= z) (per pair).

    Pooling is valid for z above the smallest pooled top-m value of *every* row, i.e. above
    z_min = max_i Z[i, -1]; below that the larger of the Gaussian tail and the pooled tail is
    returned (Gaussian-tail guard).
    """
    n = Z.shape[0]
    N = n - 1
    flat = np.sort(Z.ravel())[::-1]
    z_floor = float(Z[:, -1].max())
    # P0(Z >= z) ~ #{(i,j): z_ij >= z} / (n N)
    def sf(z):
        z = np.asarray(z)
        idx = np.searchsorted(-flat, -z, side="right")
        emp = np.maximum(idx, 1) / (n * N)
        gau = stats.norm.sf(z)
        return np.where(z >= z_floor, emp, np.maximum(gau, emp))
    return sf


def row_hc(p_sorted, N, alpha0=0.1):
    """Row-wise higher criticism of sorted pair p-values (one row per sample), with the p-values
    floored at 1/(10 N) and ranks restricted to t <= alpha0 N."""
    m = p_sorted.shape[1]
    t = np.arange(1, m + 1)[None, :]
    p = np.clip(p_sorted, 1.0 / (10 * N), 1 - 1e-12)
    hc = np.sqrt(N) * (t / N - p) / np.sqrt(p * (1 - p))
    hc[:, t[0] > alpha0 * N] = -np.inf
    return hc.max(1)


def que_scores(Y, alpha=4.0):
    """QUE spectral score (Dong-Hopkins-Li 2019) on robustly whitened, *non-sphered* data.
    NB: on Tyler-sphered data the scatter is isotropic by construction (Tyler's fixed-point
    equation), so spectral statistics must use the whitened vectors before sphering."""
    Y = Y / Y.norm(dim=1).mean() * math.sqrt(Y.shape[1])
    S = Y.T @ Y / Y.shape[0]
    lam, V = torch.linalg.eigh(S)
    w = torch.exp(alpha * (lam - 1) / (lam.max() - 1).clamp_min(1e-6))
    Q = (V * w) @ V.T / w.sum()
    return ((Y @ Q) * Y).sum(1).cpu().numpy()


def robust_pvalues(T, log=False):
    """Contamination-robust upper-tail p-values: Gaussian tail of the median/MAD z-score
    (breakdown 1/2), optionally on the log scale for positive quadratic-form scores.
    Used for *ranking* in the combination; GPD p-values are used for BH."""
    T = np.asarray(T, dtype=np.float64)
    if log:
        T = np.log(np.maximum(T, 1e-12 * max(np.max(T), 1e-12)))
    med = np.median(T)
    mad = np.median(np.abs(T - med)) * 1.4826 + 1e-12
    return stats.norm.sf((T - med) / mad)


def bonferroni_min(ps):
    """Bonferroni minimum p-value: valid under arbitrary dependence and, unlike the Cauchy
    combination, insensitive to uninformative components with p close to 1."""
    P = np.stack([np.clip(p, 1e-300, 1.0) for p in ps])
    return np.minimum(1.0, P.shape[0] * P.min(0))


def gpd_tail_pvalues(T, q_lo=0.90, q_hi=0.995):
    """Empirical-null p-values for row statistics T via a generalised Pareto tail.

    The bulk of rows is null (k << n), so we fit a GPD to exceedances of T over its
    q_lo-quantile, using only exceedances below the q_hi-quantile to limit the influence
    of the (at most k) non-null rows, then extrapolate:
        p_i = min{(1 - q_lo) * GPD_sf(T_i - u), rank_i / (n + 1)}   for T_i > u,
        p_i = rank_i / (n + 1)                                         otherwise,
    with rank_i the rank of T_i in decreasing order.
    """
    T = np.asarray(T, dtype=np.float64)
    n = T.size
    u = np.quantile(T, q_lo)
    hi = np.quantile(T, q_hi)
    exc = T[(T > u) & (T <= hi)] - u
    ranks = stats.rankdata(-T) / (n + 1)
    try:
        c, loc, scale = stats.genpareto.fit(exc, floc=0.0)
        # truncated fit correction is ignored; the (q_lo, q_hi] window is the bulk of the tail
        tail = (1 - q_lo) * stats.genpareto.sf(T - u, c, loc=0.0, scale=scale)
    except Exception:  # pragma: no cover
        tail = ranks
    return np.where(T > u, np.minimum(tail, ranks), ranks)


@torch.no_grad()
def conditional_pair_pvalues(Z, I, UR, n_bins=40, sample=2_000_000, seed=0, block=2048):
    """Pair p-values of victim coherence *conditional on reference similarity*.

    Null model: victim coherence of clean pairs is explained by their similarity in an
    independent reference representation (pretrained encoder / base model), i.e.
    z_ij | c^R_ij has a law depending only on c^R_ij.  For each reference bin b,
        P0(z >= t | b) ~ #{pooled top-m pairs in b with z >= t} / #{all pairs in b}.
    Returns p (n x m) aligned with Z, and the reference cosines of the top-m partners.
    """
    n, m = Z.shape
    UR = UR.to(torch.float32)
    Z = Z.to(torch.float32)
    g = torch.Generator(device=UR.device).manual_seed(seed)
    a = torch.randint(0, n, (sample,), device=UR.device, generator=g)
    b = torch.randint(0, n, (sample,), device=UR.device, generator=g)
    ok = a != b
    a, b = a[ok], b[ok]
    cr_s = torch.cat([(UR[a[c:c + 65536]] * UR[b[c:c + 65536]]).sum(1) for c in range(0, a.numel(), 65536)])
    qs = torch.tensor(np.concatenate([np.linspace(0, 0.98, n_bins - 5), [0.99, 0.999, 0.9999, 0.99999, 1.0]]),
                      device=UR.device, dtype=torch.float32)
    edges = torch.quantile(cr_s[:min(cr_s.numel(), 16_000_000)], qs)
    edges[0], edges[-1] = -1.01, 1.01
    nb = edges.numel() - 1
    # denominators: number of (ordered) pairs per reference bin, over all pairs
    tot = torch.zeros(nb, device=UR.device, dtype=torch.float64)
    for s in range(0, n, block):
        e = min(n, s + block)
        C = UR[s:e] @ UR.T
        C[torch.arange(e - s), torch.arange(s, e)] = 2.0  # excluded (lands beyond last edge)
        bb = torch.bucketize(C.flatten(), edges[1:-1])
        bb = bb[C.flatten() < 1.5]
        tot += torch.bincount(bb, minlength=nb).double()
    # reference cosines of the victim top-m partners
    CR = torch.empty_like(Z)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        CR[s:e] = (UR[s:e, None, :] * UR[I[s:e]]).sum(-1)
    bins = torch.bucketize(CR, edges[1:-1])
    zf, bf = Z.flatten(), bins.flatten()
    p = torch.empty_like(zf, dtype=torch.float64)
    for t in range(nb):
        sel = bf == t
        if sel.sum() == 0:
            continue
        zt = zf[sel]
        order = torch.argsort(zt, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(1, order.numel() + 1, device=zt.device)
        p[sel] = rank.double() / tot[t].clamp_min(1.0)
    return p.view(n, m).clamp(max=1.0).cpu().numpy(), CR


def _whiten_from(Xr, fit_mask, reg):
    """Tyler shape fitted on rows `fit_mask`, applied to all rows (leave-candidates-out)."""
    S = core.tyler_shape(Xr[fit_mask], reg=reg)
    return Xr @ core.inv_sqrt_psd(S)


def _hc_from_Z(Z, I, ref_U, skip_top, N, store=None, name=None):
    Zn = Z.cpu().numpy()
    if ref_U is None:
        sf = empirical_tail(Zn)
        P = sf(Zn)
        if store is not None:
            store[name] = sf
    else:
        P, _ = conditional_pair_pvalues(Z, I, ref_U)
        P = np.sort(P, axis=1)
    return row_hc(P[:, skip_top:], N - skip_top)


@torch.no_grad()
def _query_rows_local(Uq, U, m=64, w_lo=32, w_hi=512, block=1024):
    """Local-null standardised top-m cosines of external query rows Uq against the data U."""
    Uq32, U32 = Uq.to(torch.float32), U.to(torch.float32)
    w_hi = min(w_hi, U32.shape[0])
    out = []
    for s in range(0, Uq32.shape[0], block):
        G = Uq32[s:s + block] @ U32.T
        v, _ = torch.topk(G, w_hi, dim=1)
        win = v[:, w_lo:w_hi]
        loc = win.median(1).values
        sc = (win - loc[:, None]).abs().median(1).values * 1.4826
        out.append(((v[:, :m] - loc[:, None]) / sc[:, None].clamp_min(1e-9)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def _query_rows_global(Uq, U, m=128, block=1024):
    Uq32, U32 = Uq.to(torch.float32), U.to(torch.float32)
    out = []
    for s in range(0, Uq32.shape[0], block):
        G = Uq32[s:s + block] @ U32.T
        md = G.median(1).values
        md_ = (G - md[:, None]).abs().median(1).values * 1.4826
        out.append(torch.topk((G - md[:, None]) / md_[:, None].clamp_min(1e-12), m, dim=1).values.cpu())
    return torch.cat(out).numpy()


def _coherence_pass(Y, m, ref_U, skip_top, null="multi"):
    """Row-wise HC under a two-level empirical null.

    null = 'global' : row median/MAD over all entries (Efron-type);
           'local'  : row location/scale from the partners of ranks 33..512 (robust to clusters);
           'multi'  : both, merged by the Bonferroni minimum (robust over the unknown k).
    """
    U = core.unit_rows(Y)
    n = U.shape[0]
    N = n - 1
    parts, tails = {}, {}
    if null in ("global", "multi"):
        Z, I, _, _ = core.row_standardized_topk(U, m=m)
        parts["global"] = _hc_from_Z(Z, I, ref_U, skip_top, N, tails, "global")
    if null in ("local", "multi"):
        Z, I = core.row_local_topk(U, m=min(m, 64))
        parts["local"] = _hc_from_Z(Z, I, ref_U, skip_top, N, tails, "local")
    ps = {k: gpd_tail_pvalues(v) for k, v in parts.items()}
    p_coh = ps[next(iter(ps))] if len(ps) == 1 else bonferroni_min(list(ps.values()))
    out = {"hc_g": -np.log(np.clip(p_coh, 1e-300, 1)), "p_coh": p_coh}
    for k, v in parts.items():
        out[f"hc_g_{k}"] = v
    out["_tails"], out["_U"] = tails, U
    return out


def _trusted_selection(out, Y, Yc, skip_top, m):
    """Family selection in trusted-clean mode (family-evidence rule of the paper): score the trusted
    clean points with the same statistics, compute for each family f (coherence global/local,
    spectral, marginal) the standardised excess E_f = (N_f - mu) / sqrt(mu), mu = n / (m + 1), of
    data points beyond the trusted maximum, and rank by the Bonferroni minimum of the robust
    p-values of the families with E_f >= max E_f / 2 (all families, i.e. score_min, if
    max E_f <= 3)."""
    U, Uc = out["_U"], core.unit_rows(Yc)
    fams = {}
    # coherence (both scales), spectral (QUE), marginal
    for name, fn, mm in [("global", _query_rows_global, m), ("local", _query_rows_local, 64)]:
        if name in out["_tails"]:
            Zq = fn(Uc, U, mm)
            Pq = np.sort(out["_tails"][name](Zq), axis=1)
            fams[f"coh_{name}"] = (out[f"hc_g_{name}"], row_hc(Pq[:, skip_top:], U.shape[0] - skip_top))
    Ys = Y / Y.norm(dim=1).mean() * math.sqrt(Y.shape[1])
    Yq = Yc / Y.norm(dim=1).mean() * math.sqrt(Y.shape[1])
    S = Ys.T @ Ys / Ys.shape[0]
    lam, V = torch.linalg.eigh(S)
    w = torch.exp(4.0 * (lam - 1) / (lam.max() - 1).clamp_min(1e-6))
    Q = (V * w) @ V.T / w.sum()
    fams["spec"] = (out["que"], ((Yq @ Q) * Yq).sum(1).cpu().numpy())
    fams["marg"] = (out["marg"], (Yc ** 2).sum(1).cpu().numpy())
    n, mc = len(out["que"]), len(Yc)
    ev = {}
    for f, (T, Tc) in fams.items():
        excess = int((T > np.max(Tc)).sum())
        mu = n / (mc + 1)
        ev[f] = (excess - mu) / math.sqrt(mu)
    best = max(ev, key=ev.get)
    res = {"family_evidence": ev, "family": best}
    if ev[best] > 3.0:
        # keep every family whose evidence is within a factor 2 of the strongest one
        keep = [f for f in fams if ev[f] >= 0.5 * ev[best]]
        res["family"] = "+".join(keep)
        ps = []
        for f in keep:
            T = fams[f][0]
            if f.startswith("coh"):
                ps.append(robust_pvalues(np.log1p(np.maximum(T - np.min(T), 0.0))))
            else:
                ps.append(robust_pvalues(T, log=True))
        res["score_sel"] = -np.log(np.clip(bonferroni_min(ps), 1e-300, 1))
    return res


def cohere(X, m=128, whiten="tyler", with_spectral=True, return_all=False, skip_top=1, ref=None,
           two_pass=True, liberal_q=0.5, null="multi", clean=None):
    """CoHeRe detector (method section of the paper; pseudocode in its Supplementary Material).
    Returns per-sample scores (higher = more suspicious).

    X:     (n, d) features of the inspected samples (one class).
    m:     number of top standardised coherences kept per row at the global scale (L = 128 in the
           paper); the local scale keeps min(m, 64).
    clean: optional (n_c, d) features of a trusted clean set (same feature map). If given, the
           whitening is estimated on it (Theorem 9(c)), which removes self-whitening absorption
           in the dense regime; the second pass is then unnecessary. Use n_c >> d.
    ref:   optional (n, d_ref) features of the same samples in an independent reference
           representation; pair p-values are then conditional on reference similarity
           (optional reference conditioning, Supplementary Material; not used in the paper).
    return_all: if True, return a dict with the component statistics; out["p"] are the p-values
           for Benjamini-Hochberg (cohere.core.bh(out["p"], q)).
    """
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=core.DEV)
    mu = Xt.mean(0, keepdim=True)
    Xc = Xt - mu
    _, s_, Vh = torch.linalg.svd(Xc, full_matrices=False)
    keep = s_ ** 2 > 1e-8 * (s_ ** 2).max()
    B = Vh[keep].T
    Xr = Xc @ B
    n, d = Xr.shape
    reg = 0.0 if d / n < 0.2 else min(0.5, d / n)
    ref_U = None if ref is None else core.unit_rows(effective_subspace(ref))
    if clean is not None:
        Ct = torch.as_tensor(np.asarray(clean), dtype=torch.float64, device=core.DEV)
        Cr = (Ct - Ct.mean(0, keepdim=True)) @ B
        Cq = (Ct - mu) @ B                      # trusted points in the data's coordinates
        regc = 0.0 if d / Cr.shape[0] < 0.2 else min(0.5, d / Cr.shape[0])
        Wt = core.inv_sqrt_psd(core.tyler_shape(Cr, reg=regc))
        Y1 = Xr @ Wt
        two_pass = False
    else:
        # pass 1: self-whitened (all rows)
        Y1 = core.whiten(Xr, method=whiten, center="none", reg=reg)
    out = _coherence_pass(Y1, m, ref_U, skip_top, null)
    out["pass1_p_coh"] = out["p_coh"]
    Y = Y1
    if two_pass:
        lib = core.bh(out["p_coh"], liberal_q)
        if 0 < lib.sum() < n // 4:
            Y = _whiten_from(Xr, torch.as_tensor(~lib, device=Xr.device), reg)
            out.update(_coherence_pass(Y, m, ref_U, skip_top, null))
        out["n_liberal"] = int(lib.sum())
    if with_spectral:
        # spectral (QUE) and marginal (robust Mahalanobis = squared norm after Tyler whitening)
        # components on the (re-)whitened, non-sphered vectors; all calibrated by GPD tails.
        q = que_scores(Y)
        mg = (Y ** 2).sum(1).cpu().numpy()
        out["que"], out["marg"] = q, mg
        out["p_spec"] = gpd_tail_pvalues(q)
        out["p_marg"] = gpd_tail_pvalues(mg)
        # BH / FDR: GPD-calibrated components
        out["p"] = bonferroni_min([out["p_coh"], out["p_spec"], out["p_marg"]])
        # ranking: contamination-robust components (valid up to 50% poisons)
        coh_stats = [out[k] for k in sorted(out) if k.startswith("hc_g_")] or [out["hc_g"]]
        coh_ps = [robust_pvalues(np.log1p(np.maximum(h - np.min(h), 0.0))) for h in coh_stats]
        out["p_rank"] = bonferroni_min(coh_ps + [robust_pvalues(q, log=True), robust_pvalues(mg, log=True)])
    else:
        out["p"] = out["p_coh"]
    pr = out.get("p_rank", out["p"])
    out["score_min"] = -np.log(np.clip(pr, 1e-300, 1)) + 1e-6 * out["hc_g"]
    if clean is not None and with_spectral and ref is None:
        out.update(_trusted_selection(out, Y, Cq @ Wt, skip_top, m))
    out.pop("_tails", None)
    out.pop("_U", None)
    out["score"] = out["score_sel"] if "score_sel" in out else out["score_min"]
    return out if return_all else out["score"]
