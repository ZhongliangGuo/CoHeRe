# CoHeRe

Reference implementation of **CoHeRe**, the detector of the paper *Sharp Phase Transitions for Backdoor Detection at Constant Poison Count*.

Data-poisoning backdoors need a nearly constant number of poisoned samples, so a defender must find a few poisons among ever more training samples. CoHeRe inspects the features of one class (for example the penultimate-layer features of the target class) and ranks the samples mainly by the pairwise coherence of their scale-free directions, using a two-level empirical null and two-pass whitening; a spectral (QUE) and a marginal score are merged with it by a Bonferroni minimum, so that it tracks the envelope of the three detector families of the paper. It needs neither clean data nor the number of poisons; a trusted clean set, when available, enables the trusted-clean mode.

## Installation

```bash
pip install -r requirements.txt
```

A GPU is used when available (PyTorch); the detector also runs on the CPU. Tested with Python 3.12, PyTorch 2.13, NumPy 2.4, SciPy 1.18 and scikit-learn 1.9.

There is no installable package: run the examples and your own scripts from the repository root, or add it to the path (`export PYTHONPATH=/path/to/CoHeRe`). The import name `cohere` is also used by the unrelated Cohere SDK on PyPI; if that SDK is installed, make sure this repository comes first on the path.

## Usage

```python
import numpy as np
from cohere import cohere, bh

X = ...                                  # (n, d) features of the inspected class
scores = cohere(X)                       # higher = more suspicious

out = cohere(X, return_all=True)
flagged = bh(out["p"], 0.1)              # Benjamini-Hochberg set at level 0.1

C = ...                                  # (n_c, d) trusted clean features, n_c >> d
scores_t = cohere(X, clean=C)            # trusted-clean mode
```

All experiments of the paper use the defaults: L = 128 top standardized coherences per row at the global scale (`m=128`) and 64 at the local scale; a local null window of ranks 33 to 512; pooled pair tails with a Gaussian-tail guard below the pooling threshold, and all pair p-values floored at 1/(10(N-1)); row-wise higher criticism after skipping each row's best partner, over ranks up to min(L-1, 0.1(N-1)); generalized-Pareto row p-values fitted to the exceedances between the 0.90 and 0.995 quantiles; a second pass at liberal BH level 0.5, used only if the liberal set is non-empty and has fewer than n/4 rows; and Tyler whitening with ridge `min(0.5, d_eff/n)` when `d_eff/n > 0.2` (d_eff: dimension of the non-degenerate principal subspace). Benjamini-Hochberg uses the generalized-Pareto calibrated p-values (`out["p"]`); the ranking (`out["score"]`) uses robust median/MAD versions of the same components.

## Examples

**Real features.** `data/cifar10_blend_k50_seed0.npz` holds the 512-dimensional ResNet-18 features of the 5050 target-class training samples of one CIFAR-10 run of the paper (Blend trigger, k = 50 poisons, seed 0), the poison mask, and the 1000 clean test features of the same class. The features are computed from CIFAR-10 (A. Krizhevsky, *Learning multiple layers of features from tiny images*, 2009).

```bash
python examples/run_features.py data/cifar10_blend_k50_seed0.npz
```

```
n = 5050, d = 512, poisons = 50
CoHeRe (self-contained)  AUROC 0.9999  TPR@1%FPR 1.000  BH(q=0.1): 42 flagged, 34 of them poisons
  (1.6 s)
QUE (self-whitened)      AUROC 0.5589  TPR@1%FPR 0.020
CoHeRe[T] (1000 clean)   AUROC 1.0000  TPR@1%FPR 1.000
  families used: coh_global+coh_local+spec
```

The running time varies with the hardware. The QUE line is a baseline: with the whitening fitted on the poisoned samples themselves, it reaches only AUROC 0.56, because self-whitening absorbs the poison spike (Proposition 3 of the paper).

**Synthetic data.** On the elliptical coherent-contamination model with heavy-tailed radii, spectral signatures fail below the threshold sqrt(n)/k while CoHeRe needs a trigger strength of order sqrt(2 log n):

```bash
python examples/demo_synthetic.py --n 8000 --k 5 --r 10
```

```
n = 8000, d = 256, k = 5, r = 10.0; spectral threshold sqrt(n)/k = 17.9, coherence threshold sqrt(2 log n) = 4.2
spectral signature AUROC 0.467
CoHeRe             AUROC 0.999
```

## Results

[RESULTS.md](RESULTS.md) lists the per-attack AUROC and TPR at 1% FPR of every detector on CIFAR-10 (ResNet-18 and VGG-16-BN) and GTSRB, and the per-attack cleanse-and-retrain results.

## Running time

On one RTX 3090 with d = 512, CoHeRe takes 1.4, 2.3, 4.2 and 11.2 s for n = 5000, 10^4, 2 x 10^4 and 5 x 10^4. The Gram matrix is never stored: rows are processed in blocks of 1024.
