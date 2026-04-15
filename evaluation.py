import numpy as np
import pandas as pd
import warnings

from numpy.random import default_rng
from scipy import stats 
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, BayesianRidge

def evaluate_imputation_error(X_true, imputed_dict: dict, missing_mask):
    """
    Returns:
      * rmse_missing     : RMSE over truly missing entries
      * mae_missing      : MAE  over truly missing entries
      * rmse_cols_mean   : mean of per-column RMSEs (over their missing entries)
      * bias_missing     : mean(X_imp - X_true) over all missing entries
      * abs_bias_missing : abs(mean(X_imp - X_true)) over all missing entries
      * bias_cols_mean   : mean of per-column biases over missing entries
    """
    X_true = np.asarray(X_true, dtype=float)
    if X_true.ndim != 2:
        raise ValueError("X_true must be 2D.")
    n, p = X_true.shape

    missing_mask = np.asarray(missing_mask, dtype=bool)
    if missing_mask.shape != X_true.shape:
        raise ValueError("missing_mask must match X_true shape.")

    rows = []
    for name, X_imp in imputed_dict.items():
        X_imp = np.asarray(X_imp, dtype=float)
        if X_imp.shape != X_true.shape:
            raise ValueError(f"Imputed result '{name}' has shape {X_imp.shape}, expected {X_true.shape}.")

        valid_mask = missing_mask & np.isfinite(X_imp) & np.isfinite(X_true)

        if valid_mask.any():
            diff_valid = (X_imp - X_true)[valid_mask]
            rmse_missing = float(np.sqrt(np.mean(diff_valid**2)))
            mae_missing  = float(np.mean(np.abs(diff_valid)))
            bias_missing = float(np.mean(diff_valid))
            abs_bias_missing = float(abs(bias_missing))

            # Per-column metrics over their missing entries
            rmse_cols = []
            bias_cols = []
            for j in range(p):
                col_mask = valid_mask[:, j]
                if col_mask.any():
                    dcol = (X_imp[:, j] - X_true[:, j])[col_mask]
                    rmse_cols.append(np.sqrt(np.mean(dcol**2)))
                    bias_cols.append(np.mean(dcol))
            rmse_cols_mean = float(np.mean(rmse_cols)) if rmse_cols else np.nan
            bias_cols_mean = float(np.mean(bias_cols)) if bias_cols else np.nan
        else:
            rmse_missing = mae_missing = bias_missing = abs_bias_missing = rmse_cols_mean = bias_cols_mean = np.nan

        rows.append({
            "method": name,
            "rmse_missing": rmse_missing,
            "mae_missing": mae_missing,
            "bias_missing": bias_missing,
            "abs_bias_missing": abs_bias_missing,
            "rmse_cols_mean": rmse_cols_mean,
            "bias_cols_mean": bias_cols_mean,
        })

    return pd.DataFrame(rows)




def evaluate_param_recovery_regression(
    imputed_dict: dict,
    y: np.ndarray,
    beta_true: np.ndarray | None = None,        # shape (p,)
    intercept_true: float | None = None,        # optional
    ridge_alpha: float | None = None,           # if set, uses Ridge(alpha); else OLS
    *,
    X_full: np.ndarray | None = None,           # optional baseline (no missingness)
    baseline_name: str = "NoMissing",
) -> pd.DataFrame:
    """
    Fit y ~ X for each imputed X and compare estimated params to truth.
    Returns: method, coef_RMSE, coef_Bias, intercept_err.
    If X_full is provided, appends a baseline row labeled `baseline_name`.
    """
    y = np.asarray(y, float).ravel()
    rows = []

    def _fit_and_metrics(Xmat: np.ndarray, method_label: str):
        Xmat = np.asarray(Xmat, float)
        if Xmat.ndim != 2:
            raise ValueError(f"[{method_label}] X must be 2D, got shape {Xmat.shape}")
        _, p = Xmat.shape

        model = LinearRegression() if ridge_alpha is None else Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        model.fit(Xmat, y)

        beta_hat = model.coef_.reshape(-1)
        b0_hat   = float(model.intercept_)

        # Defaults if truth not provided
        coef_RMSE = coef_Bias = np.nan
        intercept_err = np.nan

        if beta_true is not None:
            bt = np.asarray(beta_true, float).reshape(-1)
            if bt.size != p:
                raise ValueError(f"[{method_label}] beta_true length {bt.size} != p {p}")
            diff = beta_hat - bt
            coef_RMSE = float(np.sqrt(np.mean(diff**2)))
            coef_Bias = float(np.mean(diff))

        if intercept_true is not None:
            intercept_err = float(b0_hat - float(intercept_true))

        return dict(
            method=method_label,
            coef_RMSE=coef_RMSE,
            coef_Bias=coef_Bias,
            intercept_err=intercept_err,
        )

    for method, X_imp in imputed_dict.items():
        rows.append(_fit_and_metrics(X_imp, method_label=method))

    if X_full is not None:
        rows.append(_fit_and_metrics(X_full, method_label=baseline_name))

    cols = ["method", "coef_RMSE", "coef_Bias", "intercept_err"]
    return pd.DataFrame(rows, columns=cols)



from numpy import ndarray
from typing import Optional
from typing import Optional
def evaluate_predictive_downstream(
    imputed_dict: dict, X_true, y,
    *,
    compute_importance: bool = False,
    importance_scoring: Optional[str] = None,  # None => auto ('r2' vs 'roc_auc')
    importance_repeats: int = 20,
    importance_random_state: int = 0,
    feature_names: Optional[list[str]] = None,
):
    """
    For each imputed X, fit on TRAIN and evaluate on TEST:
      - LinearRegression if y is continuous
      - LogisticRegression if y is binary {0,1}

    Returns:
      metrics_df
      importance_df (or None)
    """
    import numpy as np
    import pandas as pd

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        r2_score, mean_squared_error,
        accuracy_score, roc_auc_score, average_precision_score,
        log_loss, brier_score_loss, f1_score, balanced_accuracy_score
    )
    from sklearn.inspection import permutation_importance

    X_true = np.asarray(X_true, float)
    y = np.asarray(y, float).ravel()

    # detect task: binary iff values subset of {0,1} and both classes present (on observed y)
    y_fin = y[np.isfinite(y)]
    uniq = np.unique(np.round(y_fin))
    is_binary = (uniq.size == 2) and set(uniq.tolist()).issubset({0.0, 1.0})

    if importance_scoring is None:
        importance_scoring = "roc_auc" if is_binary else "r2"

    metrics_rows, imp_rows = [], []

    # ----------------------------
    # Build ONE shared train/test split (same for all methods)
    # ----------------------------
    keep = np.isfinite(y)
    idx = np.where(keep)[0]

    # deterministic split based on importance_random_state (already exists)
    rng = np.random.default_rng(int(importance_random_state))

    eval_mode = "holdout"
    if idx.size < 4:
        # too few points -> fallback to in-sample (but report it)
        eval_mode = "insample_small_n"
        train_idx = idx
        test_idx = idx
    else:
        n_train = max(2, int(np.floor(0.7 * idx.size)))
        train_idx = rng.choice(idx, size=n_train, replace=False)
        test_idx = np.setdiff1d(idx, train_idx, assume_unique=False)
        if test_idx.size < 2:
            # degenerate split -> fallback
            eval_mode = "insample_degenerate_split"
            train_idx = idx
            test_idx = idx

    # For classification: ensure both classes in train; else fallback to all observed
    if is_binary and idx.size >= 2:
        y01_all = (y[idx] > 0).astype(int)
        if np.unique(y01_all).size < 2:
            # no class variation at all
            eval_mode = "insample_one_class"
            train_idx = idx
            test_idx = idx
        else:
            y01_train = (y[train_idx] > 0).astype(int)
            if np.unique(y01_train).size < 2:
                eval_mode = "insample_train_one_class"
                train_idx = idx
                test_idx = idx

    n_train_used = int(train_idx.size)
    n_test_used = int(test_idx.size)

    # ----------------------------
    # Evaluate each method
    # ----------------------------
    for method, X_imp in imputed_dict.items():
        try:
            X_imp = np.asarray(X_imp, float)

            # shape checks
            if X_imp.ndim != 2:
                raise ValueError(f"{method}: X_imp must be 2D, got {X_imp.shape}")
            if X_imp.shape[0] != y.shape[0]:
                raise ValueError(f"{method}: n_rows mismatch X={X_imp.shape[0]} vs y={y.shape[0]}")

            # neutral fill for any non-finite entries in X_imp
            if not np.isfinite(X_imp).all():
                X_imp = X_imp.copy()
                col_means = np.nanmean(np.where(np.isfinite(X_imp), X_imp, np.nan), axis=0)
                col_means = np.where(np.isfinite(col_means), col_means, 0.0)
                bad = ~np.isfinite(X_imp)
                if bad.any():
                    X_imp[bad] = np.take(col_means, np.nonzero(bad)[1])

            X_train = X_imp[train_idx]
            y_train = y[train_idx]
            X_test  = X_imp[test_idx]
            y_test  = y[test_idx]

            # Feature names
            feats = (
                feature_names
                if (feature_names and len(feature_names) == X_imp.shape[1])
                else [f"x{j+1}" for j in range(X_imp.shape[1])]
            )

            if is_binary:
                ytr = (y_train > 0).astype(int)
                yte = (y_test > 0).astype(int)

                clf = LogisticRegression(max_iter=1000, fit_intercept=True, solver="lbfgs")
                clf.fit(X_train, ytr)

                yhat = clf.predict(X_test)
                p = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else \
                    1.0 / (1.0 + np.exp(-clf.decision_function(X_test)))

                acc = float(accuracy_score(yte, yhat))
                try:    auc = float(roc_auc_score(yte, p))
                except: auc = np.nan
                try:    ap  = float(average_precision_score(yte, p))
                except: ap  = np.nan
                try:    ll  = float(log_loss(yte, p, labels=[0, 1]))
                except: ll  = np.nan
                try:    brier = float(brier_score_loss(yte, p))
                except: brier = np.nan
                f1  = float(f1_score(yte, yhat)) if np.unique(yte).size == 2 else np.nan
                bal = float(balanced_accuracy_score(yte, yhat)) if np.unique(yte).size == 2 else np.nan
                prob_bias = float(np.mean(p - yte)) if yte.size else np.nan

                metrics_rows.append(dict(
                    method=str(method),
                    task="classification",
                    accuracy=acc, roc_auc=auc, avg_precision=ap,
                    log_loss=ll, brier=brier, f1=f1, balanced_acc=bal,
                    prob_bias=prob_bias,
                    n_train=n_train_used, n_test=n_test_used,
                    eval_mode=eval_mode,
                ))

                if compute_importance:
                    try:
                        pi = permutation_importance(
                            estimator=clf, X=X_test, y=yte,
                            scoring=importance_scoring,
                            n_repeats=importance_repeats,
                            random_state=int(importance_random_state),
                        )
                        for j, fname in enumerate(feats):
                            imp_rows.append(dict(
                                method=str(method), feature=fname,
                                imp_mean=float(pi.importances_mean[j]),
                                imp_std=float(pi.importances_std[j]),
                                imp_type="permutation_test",
                                n_train=n_train_used, n_test=n_test_used,
                                eval_mode=eval_mode,
                            ))
                    except Exception as e:
                        import warnings
                        warnings.warn(f"[{method}] permutation_importance failed: {e}")

            else:
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X_train, y_train)
                yhat = reg.predict(X_test)

                r2  = float(r2_score(y_test, yhat)) if y_test.size >= 2 else np.nan
                mse = float(mean_squared_error(y_test, yhat)) if y_test.size >= 1 else np.nan
                pred_rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
                pred_bias = float(np.mean(yhat - y_test)) if y_test.size else np.nan

                metrics_rows.append(dict(
                    method=str(method),
                    task="regression",
                    r2=r2, mse=mse, pred_rmse=pred_rmse, pred_bias=pred_bias,
                    n_train=n_train_used, n_test=n_test_used,
                    eval_mode=eval_mode,
                ))

                if compute_importance:
                    try:
                        pi = permutation_importance(
                            estimator=reg, X=X_test, y=y_test,
                            scoring=importance_scoring,
                            n_repeats=importance_repeats,
                            random_state=int(importance_random_state),
                        )
                        for j, fname in enumerate(feats):
                            imp_rows.append(dict(
                                method=str(method), feature=fname,
                                imp_mean=float(pi.importances_mean[j]),
                                imp_std=float(pi.importances_std[j]),
                                imp_type="permutation_test",
                                n_train=n_train_used, n_test=n_test_used,
                                eval_mode=eval_mode,
                            ))
                    except Exception as e:
                        import warnings
                        warnings.warn(f"[{method}] permutation_importance failed: {e}")

        except Exception as e:
            import warnings
            warnings.warn(f"[{method}] predictive eval failed: {e}")

    metrics_df = pd.DataFrame(metrics_rows)
    importance_df = pd.DataFrame(imp_rows) if (compute_importance and len(imp_rows) > 0) else None
    return metrics_df, importance_df




_LABEL_MAP = {
    "mean": "Mean",
    "Mean": "Mean",
    "median": "Median",
    "Median": "Median",
    "knn": "kNN (k=3)",
    "knn3": "kNN (k=3)",
    "kNN (k=3)": "kNN (k=3)",
    "KNN (k=3)": "kNN (k=3)",
}
def _canon_label(name: str) -> str:
    return _LABEL_MAP.get(str(name), str(name))


def _baseline_coef_rmse(X: np.ndarray, y: np.ndarray, beta_true: np.ndarray, *, ridge_alpha: float | None = None) -> float:
    """
    Fit the complete-data regression (X, y) and return RMSE(beta_hat vs beta_true).
    Uses Ridge(alpha=ridge_alpha) if provided, else OLS.
    """
    from sklearn.linear_model import Ridge, LinearRegression

    if ridge_alpha is not None:
        reg = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
    else:
        reg = LinearRegression(fit_intercept=False)

    reg.fit(X, y)
    beta_hat = np.asarray(getattr(reg, "coef_", None), dtype=float).ravel()
    err = beta_hat - np.asarray(beta_true, dtype=float).ravel()
    return float(np.sqrt(np.mean(err ** 2)))


def evaluate_param_recovery_regression_safe(
    imputed_dict: dict[str, "np.ndarray"],
    *,
    y: "np.ndarray",
    beta_true: "np.ndarray",
    intercept_true: float = 0.0,
    ridge_alpha: float | None = None,
) -> "pd.DataFrame":
    """
    Computes per-method parameter recovery with robust fallbacks.
    - Safely canonicalizes method names (if _canon_label exists; otherwise uses raw name).
    - Validates shapes and finiteness.
    - If the requested fit fails or yields non-finite metrics, retries with Ridge(alpha=1e-3).

    Returns columns: ['method','coef_RMSE','coef_Bias','intercept_err']
    """
    import numpy as np, pandas as pd, warnings
    from sklearn.linear_model import Ridge, LinearRegression

    y = np.asarray(y, dtype=float).ravel()
    beta_true = np.asarray(beta_true, dtype=float).ravel()
    if not (np.isfinite(y).all() and np.isfinite(beta_true).all()):
        raise ValueError("y and beta_true must be finite.")

    # --- safe canonicalizer ---
    try:
        mapper = _canon_label  # may be dict/callable/undefined
    except NameError:
        mapper = None

    def canon(name: str) -> str:
        if mapper is None:
            return str(name)
        try:
            if isinstance(mapper, dict):
                v = mapper.get(name, name)
            else:
                v = mapper(name)
            if v is None:
                return str(name)
            try:
                import numpy as _np
                if isinstance(v, float) and _np.isnan(v):
                    return str(name)
            except Exception:
                pass
            return str(v)
        except Exception:
            return str(name)

    rows = []
    for raw_name, X_imp in imputed_dict.items():
        name = canon(str(raw_name))
        X_imp = np.asarray(X_imp, dtype=float)

        # --- basic validity checks ---
        if X_imp.ndim != 2:
            warnings.warn(f"[{name}] X_imp is not 2D (shape={X_imp.shape}); skipping.")
            continue
        if X_imp.shape[0] != y.shape[0]:
            warnings.warn(f"[{name}] n_rows mismatch: X={X_imp.shape[0]} vs y={y.shape[0]}; skipping.")
            continue
        if X_imp.shape[1] != beta_true.shape[0]:
            warnings.warn(f"[{name}] n_cols mismatch: X={X_imp.shape[1]} vs beta_true={beta_true.shape[0]}; skipping.")
            continue

        if not np.isfinite(X_imp).all():
            warnings.warn(f"[{name}] X_imp has non-finite values; attempting to fill.")
            X_imp = X_imp.copy()
            col_means = np.nanmean(np.where(np.isfinite(X_imp), X_imp, np.nan), axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            bad = ~np.isfinite(X_imp)
            if bad.any():
                X_imp[bad] = np.take(col_means, np.nonzero(bad)[1])

        def _fit_and_metrics(use_ridge: bool, alpha_val: float | None = None):
            if use_ridge:
                reg = Ridge(alpha=float(alpha_val if alpha_val is not None else 1e-3),
                            fit_intercept=False)
            else:
                reg = LinearRegression(fit_intercept=False)
            reg.fit(X_imp, y)
            beta_hat = np.asarray(reg.coef_, dtype=float).ravel()
            bt = beta_true
            rmse = float(np.sqrt(np.mean((beta_hat - bt) ** 2)))
            bias = float(np.mean(beta_hat - bt))
            int_err = float(abs(intercept_true - 0.0))
            return rmse, bias, int_err

        try:
            if ridge_alpha is None:
                rmse, bias, int_err = _fit_and_metrics(use_ridge=False)
            else:
                rmse, bias, int_err = _fit_and_metrics(use_ridge=True, alpha_val=ridge_alpha)
            if not (np.isfinite(rmse) and np.isfinite(bias)):
                raise ValueError("non-finite metrics")
        except Exception as e:
            warnings.warn(f"[{name}] primary fit failed ({e}); retrying with Ridge(alpha=1e-3).")
            rmse, bias, int_err = _fit_and_metrics(use_ridge=True, alpha_val=1e-3)

        rows.append({"method": name, "coef_RMSE": rmse, "coef_Bias": bias, "intercept_err": int_err})

    import pandas as pd
    return pd.DataFrame(rows)


def evaluate_param_recovery_safe(
    imputed_dict: dict[str, ndarray],
    *,
    y: ndarray,
    beta_true: ndarray,
    intercept_true: float = 0.0,
    ridge_alpha: Optional[float] = None,
    fit_intercept: Optional[bool] = None,
) -> pd.DataFrame:

    import warnings

    y = np.asarray(y, float).ravel()
    beta_true = np.asarray(beta_true, float).ravel()
    if not (np.isfinite(y).all() and np.isfinite(beta_true).all()):
        raise ValueError("y and beta_true must be finite.")

    y_fin = y[np.isfinite(y)]
    u = np.unique(np.round(y_fin))
    is_binary = (u.size == 2) and set(u.tolist()).issubset({0.0, 1.0})

    if fit_intercept is None:
        fit_intercept_local = True if is_binary else False
    else:
        fit_intercept_local = bool(fit_intercept)

    rows = []
    for raw_name, X_imp in imputed_dict.items():
        name = str(raw_name)
        X_imp = np.asarray(X_imp, float)

        if X_imp.ndim != 2:
            warnings.warn(f"[{name}] X_imp is not 2D (shape={X_imp.shape}); skipping.")
            continue
        if X_imp.shape[0] != y.shape[0]:
            warnings.warn(f"[{name}] n_rows mismatch: X={X_imp.shape[0]} vs y={y.shape[0]}; skipping.")
            continue
        if X_imp.shape[1] != beta_true.shape[0]:
            warnings.warn(f"[{name}] n_cols mismatch: X={X_imp.shape[1]} vs beta_true={beta_true.shape[0]}; skipping.")
            continue

        if not np.isfinite(X_imp).all():
            warnings.warn(f"[{name}] X_imp has non-finite values; attempting to fill.")
            X_imp = X_imp.copy()
            col_means = np.nanmean(np.where(np.isfinite(X_imp), X_imp, np.nan), axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            bad = ~np.isfinite(X_imp)
            if bad.any():
                X_imp[bad] = np.take(col_means, np.nonzero(bad)[1])

        keep = np.isfinite(y)
        X_use = X_imp[keep]
        y_use = y[keep]

        def _fit_and_metrics():
            if is_binary:
                C = 1.0 / float(ridge_alpha) if (ridge_alpha is not None and ridge_alpha > 0) else 1.0
                mdl = LogisticRegression(
                    penalty="l2", C=C, max_iter=1000, fit_intercept=fit_intercept_local, solver="lbfgs"
                )
                mdl.fit(X_use, (y_use > 0).astype(int))
                beta_hat = np.asarray(mdl.coef_, float).ravel()
                b0_hat = float(mdl.intercept_[0]) if fit_intercept_local else 0.0
                task = "classification"
            else:
                if ridge_alpha is not None:
                    mdl = Ridge(alpha=float(ridge_alpha), fit_intercept=fit_intercept_local)
                else:
                    mdl = LinearRegression(fit_intercept=fit_intercept_local)
                mdl.fit(X_use, y_use)
                beta_hat = np.asarray(mdl.coef_, float).ravel()
                b0_hat = float(mdl.intercept_) if fit_intercept_local else 0.0
                task = "regression"

            rmse = float(np.sqrt(np.mean((beta_hat - beta_true) ** 2)))
            bias = float(np.mean(beta_hat - beta_true))
            int_err = float(abs(b0_hat - float(intercept_true)))
            return task, rmse, bias, int_err

        try:
            task, rmse, bias, int_err = _fit_and_metrics()
            if not (np.isfinite(rmse) and np.isfinite(bias)):
                raise ValueError("non-finite metrics")
        except Exception as e:
            warnings.warn(f"[{name}] primary fit failed ({e}); retrying with stronger L2.")
            if is_binary:
                ridge_alpha = (ridge_alpha if ridge_alpha is not None else 1.0) * 10.0
            else:
                ridge_alpha = (ridge_alpha if ridge_alpha is not None else 1e-3) * 10.0
            task, rmse, bias, int_err = _fit_and_metrics()

        rows.append({
            "method": name, "task": task,
            "coef_RMSE": rmse, "coef_Bias": bias, "intercept_err": int_err
        })

    return pd.DataFrame(rows)



