"""Synthetic check of the constant-k picture on the elliptical coherent-contamination model (ECM).

Clean features x_i = R_i Sigma^{1/2} theta_i with heavy-tailed (log-normal) radii R_i and an anisotropic
Sigma; k poisons share the trigger displacement Sigma^{1/2} a w. With a^2 = r sqrt(d) the coherence SNR of
Definition 2 is about r. At constant k the spectral threshold sqrt(n)/k grows with n, whereas coherence
needs only r of order sqrt(2 log n): for r between the two, spectral signatures fail and CoHeRe succeeds.

  python examples/demo_synthetic.py --n 8000 --k 5 --r 10
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohere import cohere  # noqa: E402


def sample_ecm(n, d, k, r, sigma_r=0.5, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.geomspace(10.0, 0.5, d)                      # anisotropic spectrum
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sh = (Q * np.sqrt(lam)) @ Q.T                         # Sigma^{1/2}
    theta = rng.standard_normal((n, d))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    R = math.sqrt(d) * np.exp(sigma_r * rng.standard_normal(n) - sigma_r ** 2)
    X = (R[:, None] * theta) @ Sh
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    poison = np.zeros(n, dtype=bool)
    poison[rng.choice(n, k, replace=False)] = True
    X[poison] += math.sqrt(r * math.sqrt(d)) * (Sh @ w)  # a^2 = r sqrt(d)
    return X, poison


def spectral_signature(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[0]) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r", type=float, default=10.0)
    args = ap.parse_args()
    X, lab = sample_ecm(args.n, args.d, args.k, args.r)
    print(f"n = {args.n}, d = {args.d}, k = {args.k}, r = {args.r}; "
          f"spectral threshold sqrt(n)/k = {math.sqrt(args.n) / args.k:.1f}, "
          f"coherence threshold sqrt(2 log n) = {math.sqrt(2 * math.log(args.n)):.1f}")
    print(f"spectral signature AUROC {roc_auc_score(lab, spectral_signature(X)):.3f}")
    print(f"CoHeRe             AUROC {roc_auc_score(lab, cohere(X)):.3f}")


if __name__ == "__main__":
    main()
