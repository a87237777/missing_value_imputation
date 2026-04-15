import numpy as np
import pandas as pd
import warnings

from numpy.random import default_rng
from scipy import stats 


#Data Simulation
def simulate_data(
    n: int = 1000,
    p: int = 8,
    rho: float = 0.5,
    snr: float = 5.0,
    seed: int = 0,
    *,
    k_nonzero: int | float | None = None,   # None = dense beta; if 0<k<1 => fraction of p
    heavy_tail: bool = False,
    heteroskedastic: bool = False,
    binary: bool = False,                   # when True, y is binary (logistic model)
    prevalence: float = 0.5,                # target P(y=1) for binary case
    beta_scale: float = 1.0,                # scales beta in binary case (controls difficulty)
):
    """
    Simulate correlated Gaussian features and a response.

    If binary=False:
        Linear model with noise calibrated to target *sample* SNR.
    If binary=True:
        Logistic model with intercept chosen so mean(sigmoid(X beta + b0)) ~= prevalence.

    Returns
    -------
    X : (n, p) ndarray
    y : (n,)   ndarray  (float for regression; {0,1} ints for binary)
    beta : (p,) ndarray
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if n < 1 or p < 1:
        raise ValueError("n and p must be >= 1.")
    if p >= 2 and not (-1.0/(p-1) < rho < 1.0):
        raise ValueError(f"rho invalid for p={p}; need -1/(p-1) < rho < 1.")
    if not binary and snr <= 0:
        raise ValueError("snr must be > 0 for the continuous (non-binary) case.")
    if binary:
        if not (0.0 < prevalence < 1.0):
            raise ValueError("prevalence must be in (0,1) for the binary case.")
        if beta_scale <= 0:
            raise ValueError("beta_scale must be > 0 for the binary case.")

    rng = default_rng(seed)


    Z = rng.standard_normal((n, p))
    if p == 1 or abs(rho) < 1e-12:
        X = Z
    else:
        g = rng.standard_normal((n, 1))
        X = np.sqrt(1 - rho) * Z + np.sqrt(rho) * g

    # --- Sparse or dense beta ---
    if k_nonzero is None:
        beta = rng.normal(0.0, 1.0, p)
    else:
        k = int(round(k_nonzero * p)) if (isinstance(k_nonzero, float) and 0 < k_nonzero < 1) else int(k_nonzero)
        k = max(1, min(k, p))
        beta = np.zeros(p)
        idx = rng.choice(p, size=k, replace=False)
        beta[idx] = rng.normal(0.0, 1.0, k)

    # --- Continuous (Gaussian) path with target SNR ---
    if not binary:
        xb = X @ beta
        var_signal = float(np.var(xb, ddof=0))
        base_noise_var = var_signal / snr if var_signal > 1e-15 else 1.0 / snr

        if heavy_tail:
            # Student-t(df=5), rescaled to target variance
            df = 5.0
            eps = rng.standard_t(df, n)
            eps *= np.sqrt(base_noise_var * (df - 2) / df / (np.var(eps, ddof=0) or 1.0))
        else:
            eps = rng.normal(0.0, np.sqrt(base_noise_var), n)

        if heteroskedastic:
            w = rng.normal(0.0, 1.0, p)
            h = X @ w
            h = (h - h.mean()) / (h.std() if h.std() >= 1e-12 else 1.0)
            scales = 1.0 + 0.5 * np.abs(h)      # mild heteroskedasticity
            eps *= scales
            # re-normalize to keep variance ≈ base_noise_var
            eps *= np.sqrt(base_noise_var / (np.var(eps, ddof=0) or base_noise_var))

        y = (X @ beta) + eps
        return X, y, beta

    # --- Binary (logistic) path ---
    beta = beta_scale * beta
    xb = X @ beta

    def _sigmoid(u):
        out = np.empty_like(u, dtype=float)
        pos = u >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
        e = np.exp(u[neg])
        out[neg] = e / (1.0 + e)
        return out

    def mean_prob(b0):
        return _sigmoid(xb + b0).mean()

    lo, hi = -20.0, 20.0
    mlo, mhi = mean_prob(lo), mean_prob(hi)
    widen = 0
    while (mlo > prevalence or mhi < prevalence) and widen < 6:
        lo -= 20.0
        hi += 20.0
        mlo, mhi = mean_prob(lo), mean_prob(hi)
        widen += 1
    if not (mlo <= prevalence <= mhi):
        raise RuntimeError("Failed to bracket prevalence; consider adjusting beta_scale or prevalence.")

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mm = mean_prob(mid)
        if mm < prevalence:
            lo = mid
        else:
            hi = mid
    b0 = 0.5 * (lo + hi)

    p_true = _sigmoid(xb + b0)
    y = rng.binomial(1, p_true).astype(int)

    # Keep return signature identical to your original
    return X, y, beta


#nonlinear simulation
def simulate_data_nonlinear(
    n: int = 1000,
    p: int = 8,
    rho: float = 0.5,
    snr: float = 5.0,
    seed: int = 0,
    *,
    kind: str = "additive",      # 'additive' or 'interaction'
    k_active: int | None = None  # features used by f(X); default ~ p/2
):
    """
    Nonlinear simulator with two options:
      - additive:      y = sum_j g_j(X_j) + ε
      - interaction:   y = sum_{(a,b)} c_{ab} X_a X_b + ε
    SNR is calibrated at the sample level: Var(signal)/Var(ε) ≈ snr.
    """
    if n < 1 or p < 1:
        raise ValueError("n and p must be >= 1.")
    if snr <= 0:
        raise ValueError("snr must be > 0.")
    if p >= 2 and not (-1.0/(p-1) < rho < 1.0):
        raise ValueError(f"rho invalid for p={p}; need -1/(p-1) < rho < 1.")

    rng = default_rng(seed)
    k_active = max(1, p // 2) if k_active is None else int(k_active)
    if not (1 <= k_active <= p):
        raise ValueError("k_active must be in [1, p].")

    # --- Equicorrelated Gaussian X ---
    cov = np.eye(p) if p == 1 else (1 - rho) * np.eye(p) + rho * np.ones((p, p))
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(p))
    X = rng.standard_normal((n, p)) @ L.T

    active = rng.choice(p, size=k_active, replace=False)

    # --- Build nonlinear signal f(X) ---
    if kind == "additive":
        f = np.zeros(n)
        # smooth per-feature transforms; centered and scaled
        for j in active:
            phase = rng.normal()  # small random phase to diversify shapes
            gj = np.tanh(0.7 * X[:, j]) + 0.5 * np.sin(2.0 * X[:, j] + phase)
            gj -= gj.mean()
            f += gj
        f /= np.sqrt(k_active)  # stabilize variance across k_active

    elif kind == "interaction":
        if k_active < 2:
            raise ValueError("interaction kind needs k_active >= 2.")
        idx = list(active)
        pairs = [(a, b) for i, a in enumerate(idx) for b in idx[i+1:]]
        num_pairs = len(pairs)
        coeffs = rng.normal(0, 1, num_pairs)
        f = np.zeros(n)
        for c, (a, b) in zip(coeffs, pairs):
            f += c * (X[:, a] * X[:, b])
        f /= np.sqrt(num_pairs)  # stabilize variance
        f -= f.mean()
    else:
        raise ValueError("kind must be 'additive' or 'interaction'.")

    # --- Calibrate noise to hit target SNR ---
    var_signal = float(np.var(f, ddof=0))
    if var_signal < 1e-14:
        warnings.warn("Signal variance ~0; adding jitter.")
        f = f + 1e-6 * rng.standard_normal(n)
        var_signal = float(np.var(f, ddof=0))
    noise_std = np.sqrt(var_signal / snr)
    eps = rng.normal(0.0, noise_std, n)

    y = f + eps
    meta = dict(
        kind=kind,
        k_active=k_active,
        active=active,
        rho=rho,
        seed=seed,
        snr_target=snr,
        snr_empirical=float(np.var(f) / np.var(eps)) if np.var(eps) > 0 else np.inf,
    )
    return X, y, meta




def make_missing(
    X,
    mechanism: str = "MCAR",
    prop: float = 0.3,
    strength: float = 2.0,
    seed: int = 1,
    driver_col: int | None = None,
    y=None,
    prop_y: float | None = None,
    mech_y: str = "MCAR",
):
    import numpy as np
    from numpy.random import default_rng

    def _sigmoid(u):
        u = np.asarray(u, dtype=float)
        out = np.empty_like(u)
        pos = u >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
        e = np.exp(u[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def _calibrate_shift(scores, target_prop, iters=60):
        lo, hi = -50.0, 50.0
        for _ in range(8):
            mlo = _sigmoid(scores - lo).mean()
            mhi = _sigmoid(scores - hi).mean()
            if mlo >= target_prop >= mhi:
                break
            lo -= 50.0
            hi += 50.0

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            mm = _sigmoid(scores - mid).mean()
            if mm > target_prop:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _standardize_1d(v):
        v = np.asarray(v, dtype=float)
        mu = np.mean(v)
        sd = np.std(v)
        sd_safe = sd if np.isfinite(sd) and sd >= 1e-12 else 1.0
        return (v - mu) / sd_safe

    if not (0 <= prop < 1):
        raise ValueError("prop must be in [0, 1).")
    if prop_y is not None and not (0 <= prop_y < 1):
        raise ValueError("prop_y must be in [0, 1).")

    rng = default_rng(seed)

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError("X must have positive dimensions")

    mech = str(mechanism).upper()
    if mech == "MAR" and p == 1:
        mech = "MNAR"
    if mech == "SLIGHTLY_MAR" and p == 1:
        raise ValueError("SLIGHTLY_MAR requires p >= 2.")

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd_safe = np.where(np.isfinite(sd) & (sd >= 1e-12), sd, 1.0)
    Z = (X - mu) / sd_safe

    mask_X = np.zeros((n, p), dtype=bool)

    if mech == "MCAR":
        mask_X = rng.random((n, p)) < prop

    elif mech == "SLIGHTLY_MAR":
        if driver_col is None:
            driver_col = int(rng.integers(0, p))
        if not (0 <= driver_col < p):
            raise ValueError(f"driver_col must be in [0, {p - 1}]")

        scores = strength * Z[:, driver_col]
        shift = _calibrate_shift(scores, prop)
        probs = _sigmoid(scores - shift)

        for j in range(p):
            if j == driver_col:
                mask_X[:, j] = rng.random(n) < prop
            else:
                pj = np.clip(probs + rng.normal(0, 1e-3, size=n), 0.0, 1.0)
                mask_X[:, j] = rng.random(n) < pj

    elif mech in {"MAR", "MNAR"}:
        for j in range(p):
            if mech == "MAR":
                k = int(rng.integers(0, p - 1))
                if k >= j:
                    k += 1
                driver = Z[:, k]
            else:
                driver = Z[:, j]

            scores = strength * driver
            shift = _calibrate_shift(scores, prop)
            probs = _sigmoid(scores - shift)
            mask_X[:, j] = rng.random(n) < probs

    else:
        raise ValueError(f"Unknown mechanism '{mechanism}'")

    for j in range(p):
        if mask_X[:, j].all():
            keep = max(1, int(np.ceil((1 - prop) * n)))
            idx_keep = rng.choice(n, size=keep, replace=False)
            mask_X[:, j] = True
            mask_X[idx_keep, j] = False

    rows_all_missing = np.where(mask_X.all(axis=1))[0]
    if rows_all_missing.size:
        unmask_cols = rng.integers(0, p, size=rows_all_missing.size)
        mask_X[rows_all_missing, unmask_cols] = False

    X_miss = X.copy()
    X_miss[mask_X] = np.nan

    y_miss, mask_y = None, None
    if y is not None and prop_y is not None and prop_y > 0:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != n:
            raise ValueError("y must have same number of rows as X")

        mech_y_u = str(mech_y).upper()

        if mech_y_u == "MCAR":
            mask_y = rng.random(n) < prop_y

        elif mech_y_u == "MNAR":
            ys = _standardize_1d(y)

            y_obs = y[np.isfinite(y)]
            y_unique = np.unique(y_obs)

            if y_unique.size <= 2 and np.all(np.isin(y_unique, [0.0, 1.0])):
                X_std = (X - np.mean(X, axis=0)) / np.where(
                    np.std(X, axis=0) >= 1e-12,
                    np.std(X, axis=0),
                    1.0,
                )

                if p == 1:
                    x_driver = X_std[:, 0]
                else:
                    w = rng.normal(size=p)
                    w = w / (np.linalg.norm(w) + 1e-12)
                    x_driver = X_std @ w

                x_driver = _standardize_1d(x_driver)

                alpha = 0.7
                scores = strength * (alpha * ys + (1.0 - alpha) * x_driver)
            else:
                scores = strength * ys

            shift = _calibrate_shift(scores, prop_y)
            probs = _sigmoid(scores - shift)
            mask_y = rng.random(n) < probs

        else:
            raise ValueError("mech_y must be 'MCAR' or 'MNAR'")

        if mask_y.all():
            keep = max(1, int(np.ceil((1 - prop_y) * n)))
            idx_keep = rng.choice(n, size=keep, replace=False)
            mask_y[:] = True
            mask_y[idx_keep] = False

        y_miss = y.copy()
        y_miss[mask_y] = np.nan

    return X_miss, mask_X, y_miss, mask_y


def compare_include_y_once(
    X, y,
    make_missing_fn=make_missing,
    make_missing_kwargs: dict | None = None,
    random_state: int = 0,
):
    """
    Run one comparison:
      1) Create X_miss via make_missing_fn(**make_missing_kwargs)
      2) Impute WITHOUT y
      3) Impute WITH y (append y to design for imputation only)
      4) Return two evaluation DataFrames: imputation errors & downstream predictions
    """
    make_missing_kwargs = make_missing_kwargs or {}
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()

    X_miss, mask_X, _, _ = make_missing_fn(X, y=y, **make_missing_kwargs)

    # Without y
    imp_wo = impute_all(X_miss, random_state=random_state, use_mice=False)

    # With y
    imp_w  = impute_all_include_y(X_miss, y=y, include_y=True, random_state=random_state, use_mice=False)

    # Evaluate
    err_wo = evaluate_imputation_error(X_true=X, imputed_dict=imp_wo, missing_mask=mask_X)
    err_w  = evaluate_imputation_error(X_true=X, imputed_dict=imp_w,  missing_mask=mask_X)
    err_wo["scenario"] = "Impute w/o y"
    err_w["scenario"]  = "Impute with y"
    err = pd.concat([err_wo, err_w], ignore_index=True)

    pred_wo = evaluate_predictive_downstream(imp_wo, X_true=X, y=y)
    pred_w  = evaluate_predictive_downstream(imp_w,  X_true=X, y=y)
    pred_wo["scenario"] = "Impute w/o y"
    pred_w["scenario"]  = "Impute with y"
    pred = pd.concat([pred_wo, pred_w], ignore_index=True)

    return {"imputation_error": err, "predictive": pred}