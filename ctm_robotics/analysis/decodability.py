"""
analysis/decodability.py

Linear-decodability R²_CV protocol — the unified interpretability metric
for the CTM-spacecraft paper (spec 2026-05-12-ctm-spacecraft-ral.md §5.1).

Given an arbitrary representation z (CTM sync, LSTM h_t, or anything else)
and a set of hidden physical variables, fit ridge regression with 5-fold CV
and report R²_CV per hidden variable. Optionally controls for visible
observation dimensions and reports ΔR² (the marginal information about
the hidden variable carried by z beyond what visible obs already provide).

Used identically at all three layers (gym, IsaacLab, hardware) so numbers
are directly comparable.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


_DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _r2_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, alphas=_DEFAULT_ALPHAS):
    """5-fold CV ridge regression; returns mean R² over folds and best α per fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_r2 = []
    best_alphas = []
    for train_idx, test_idx in kf.split(X):
        model = RidgeCV(alphas=alphas, scoring="r2", cv=3)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_r2.append(r2_score(y[test_idx], pred))
        best_alphas.append(model.alpha_)
    return float(np.mean(fold_r2)), float(np.std(fold_r2)), best_alphas


def linear_decodability(
    z: np.ndarray,
    hidden_vars: dict,
    visible_vars: np.ndarray | None = None,
    n_splits: int = 5,
    alphas: tuple = _DEFAULT_ALPHAS,
) -> dict:
    """Linear decodability of hidden variables from a representation.

    Args:
        z:            (T, D_z) representation matrix
        hidden_vars:  {name: (T,) array of ground-truth hidden variable}
        visible_vars: optional (T, D_vis) matrix of visible observations.
                      If provided, ΔR² is computed as the marginal information
                      in z over visible-only baseline.
        n_splits:     number of folds for cross-validation (default 5)
        alphas:       ridge regularization grid

    Returns:
        {name: {"r2_cv":              float,  -- R²_CV(z -> v_k)
                "r2_cv_std":          float,  -- std across folds
                "r2_cv_visible":      float,  -- R²_CV(visible -> v_k)  (or None)
                "r2_cv_visible_std":  float,  -- (or None)
                "delta_r2":           float,  -- r2_cv - r2_cv_visible (or None)
                "best_alphas":        list,   -- best alpha per fold
               }}
    """
    z = np.asarray(z)
    T, D = z.shape
    if T < 2 * D:
        raise ValueError(
            f"too few samples: T={T} < 2*D={2*D}. Increase rollout length."
        )

    out = {}
    for name, v_k in hidden_vars.items():
        v_k = np.asarray(v_k).reshape(-1)
        if len(v_k) != T:
            raise ValueError(f"target '{name}' length {len(v_k)} != z length {T}")

        if visible_vars is not None:
            X_full = np.concatenate([z, visible_vars], axis=1)
        else:
            X_full = z
        r2_cv, r2_cv_std, best_alphas = _r2_cv(X_full, v_k, n_splits, alphas)

        if visible_vars is not None:
            r2_visible, r2_visible_std, _ = _r2_cv(visible_vars, v_k, n_splits, alphas)
            delta = r2_cv - r2_visible
        else:
            r2_visible = None
            r2_visible_std = None
            delta = None

        out[name] = {
            "r2_cv":             r2_cv,
            "r2_cv_std":         r2_cv_std,
            "r2_cv_visible":     r2_visible,
            "r2_cv_visible_std": r2_visible_std,
            "delta_r2":          delta,
            "best_alphas":       best_alphas,
        }
    return out
