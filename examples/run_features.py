"""Run CoHeRe on the features of one inspected class and report AUROC, TPR at 1% FPR and the BH set.

Input: an .npz file with
  X          (n, d)   features of the inspected samples (e.g. penultimate layer of the target class),
  is_poison  (n,)     ground-truth poison mask (only used for evaluation),
  clean      (n_c, d) optional trusted clean features of the same class (trusted-clean mode).

Example (one CIFAR-10 run of the paper: ResNet-18, Blend trigger, k = 50 poisons, seed 0):
  python examples/run_features.py data/cifar10_blend_k50_seed0.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohere import bh, cohere, core  # noqa: E402
from cohere.detector import effective_subspace, que_scores  # noqa: E402


def tpr_at_fpr(lab, score, fpr=0.01):
    thr = np.quantile(score[~lab], 1 - fpr)
    return float((score[lab] > thr).mean())


def report(name, lab, out, q=None):
    s = out["score"]
    line = f"{name:<24} AUROC {roc_auc_score(lab, s):.4f}  TPR@1%FPR {tpr_at_fpr(lab, s):.3f}"
    if q is not None:
        flagged = bh(out["p"], q)
        line += f"  BH(q={q}): {flagged.sum()} flagged, {int((flagged & lab).sum())} of them poisons"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--q", type=float, default=0.1, help="BH level for the flagged set")
    args = ap.parse_args()
    f = np.load(args.npz)
    X, lab = f["X"].astype(np.float64), f["is_poison"].astype(bool)
    print(f"n = {len(X)}, d = {X.shape[1]}, poisons = {lab.sum()}")
    t0 = time.time()
    report("CoHeRe (self-contained)", lab, cohere(X, return_all=True), args.q)
    print(f"  ({time.time() - t0:.1f} s)")
    # baseline for comparison: QUE after Tyler whitening fitted on the same (poisoned) samples
    que = que_scores(core.whiten(effective_subspace(X), method="tyler", center="none"))
    report("QUE (self-whitened)", lab, {"score": que})
    if "clean" in f.files:
        C = f["clean"].astype(np.float64)
        out = cohere(X, clean=C, return_all=True)
        report(f"CoHeRe[T] ({len(C)} clean)", lab, out)
        fam = out["family"] if "score_sel" in out else "all (no family has evidence above 3)"
        print(f"  families used: {fam}")


if __name__ == "__main__":
    main()
