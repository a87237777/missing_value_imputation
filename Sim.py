def simulate_data(n=1000, p=8, rho=0.5, snr=5.0, seed=0):
    
    """
    Simulate correlated Gaussian X and linear y with controllable SNR.

    n : Number of samples.
    p : Number of features.
    rho : Equicorrelation parameter; valid range is (-1/(p-1), 1) for p>=2.
    snr : Target signal-to-noise ratio: Var(X @ beta) / noise_var.
        
    """
    if n < 1 or p < 1:
        raise ValueError("n and p must be >= 1.")
    if snr <= 0:
        raise ValueError("snr must be > 0.")
    if p >= 2 and not (-1.0/(p-1) < rho < 1.0):
        raise ValueError(f"rho invalid for p={p}; need -1/(p-1) < rho < 1.")

    rng = default_rng(seed)

    # Equicorrelated covariance (or 1x1 if p==1)
    cov = np.eye(p) if p == 1 else (1 - rho) * np.eye(p) + rho * np.ones((p, p))
    L = np.linalg.cholesky(cov)

    Z = rng.standard_normal((n, p))
    X = Z @ L.T

    beta = rng.normal(0.0, 1.0, p)

    # Use population signal variance for tighter SNR
    var_signal = float(beta @ cov @ beta)
    noise_var = max(var_signal / snr, 1e-12)

    y = X @ beta + rng.normal(0.0, np.sqrt(noise_var), n)
    return X, y, beta




def _zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)



def make_missing(
    X,
    mechanism="MCAR",
    prop=0.3,
    strength=2.0,
    seed=1,
    driver_col=None,  # for SLIGHTLY_MAR; None => choose randomly
    y=None,
    prop_y=None,      # fraction of missing in y (None => no missingness in y)
    mech_y="MCAR",    # mechanism for y: {"MCAR","MNAR"}
):
    """
    Create missingness mask(s) and return arrays with np.nan at missing entries.

    Mechanisms for X
    ----------------
    - MCAR: uniform random per column with rate `prop`
    - MAR: for each column j, missingness depends on a (random) *other* column k≠j
    - MNAR: for each column j, missingness depends on its own (latent) true value
    - SLIGHTLY_MAR: all non-driver columns depend on the same driver col; driver itself is MCAR

    Extra for y
    -----------
    If y is given, missingness can also be induced in y:
      * prop_y: fraction of missing outcomes
      * mech_y:
          - "MCAR": uniform random missingness in y
          - "MNAR": probability of missing depends on y itself (logistic function)

    Returns
    -------
    X_miss : float array with NaNs at missing entries
    mask_X : bool array, True where missing in X
    y_miss : float array (if y provided), with NaNs at missing entries
    mask_y : bool array, True where missing in y
    """
    if not (0 <= prop < 1):
        raise ValueError("prop must be in [0, 1).")
    if prop_y is not None and not (0 <= prop_y < 1):
        raise ValueError("prop_y must be in [0, 1).")

    rng = default_rng(seed)
    X = np.asarray(X, dtype=float, order="C")
    n, p = X.shape

    mech = mechanism.upper()
    if mech == "MAR" and p == 1:
        mech = "MNAR"
    if mech == "SLIGHTLY_MAR" and p == 1:
        raise ValueError("SLIGHTLY_MAR requires p >= 2.")

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = sd.copy()
    sd_safe[sd_safe < 1e-12] = np.inf  # (x - mu)/inf == 0
    Z = (X - mu) / sd_safe

    def _sigmoid(u):
        # numerically stable logistic
        return np.where(u >= 0, 1.0 / (1.0 + np.exp(-u)), np.exp(u) / (1.0 + np.exp(u)))

    mask_X = np.zeros_like(X, dtype=bool)

    if mech == "SLIGHTLY_MAR":
        if driver_col is None:
            driver_col = int(rng.integers(0, p))
        driver = Z[:, driver_col]
        prob_all = _sigmoid(strength * driver)
        t = np.quantile(prob_all, 1 - prop)
        driven_mask = prob_all > t

        for j in range(p):
            if j == driver_col:
                mask_X[:, j] = rng.random(n) < prop
            else:
                mask_X[:, j] = driven_mask

            if mask_X[:, j].all():
                keep = rng.choice(n, size=max(1, int(np.ceil((1 - prop) * n))), replace=False)
                mask_X[:, j] = True
                mask_X[keep, j] = False

    else:
        for j in range(p):
            if mech == "MCAR":
                mask_j = rng.random(n) < prop
            elif mech == "MAR":
                k = int(rng.integers(0, p - 1))
                if k >= j:
                    k += 1
                driver = Z[:, k]
                prob = _sigmoid(strength * driver)
                t = np.quantile(prob, 1 - prop)
                mask_j = prob > t
            elif mech == "MNAR":
                driver = Z[:, j]
                prob = _sigmoid(strength * driver)
                t = np.quantile(prob, 1 - prop)
                mask_j = prob > t
            else:
                raise ValueError(f"Unknown mechanism '{mechanism}'")

            if mask_j.all():
                keep = rng.choice(n, size=max(1, int(np.ceil((1 - prop) * n))), replace=False)
                mask_j[:] = True
                mask_j[keep] = False

            mask_X[:, j] = mask_j

    rows_all_missing = np.where(mask_X.all(axis=1))[0]
    if rows_all_missing.size:
        unmask_cols = rng.integers(0, p, size=rows_all_missing.size)
        mask_X[rows_all_missing, unmask_cols] = False

    X_miss = X.copy()
    X_miss[mask_X] = np.nan

    y_miss, mask_y = None, None
    if y is not None and prop_y is not None and prop_y > 0:
        y = np.asarray(y, dtype=float)
        if y.shape[0] != n:
            raise ValueError("y must have same number of rows as X")

        mech_y_u = mech_y.upper()
        if mech_y_u == "MCAR":
            mask_y = rng.random(n) < prop_y
        elif mech_y_u == "MNAR":
            y_sd = float(y.std())
            driver_y = (y - y.mean()) / (y_sd if y_sd >= 1e-12 else np.inf)
            prob_y = _sigmoid(strength * driver_y)
            t_y = np.quantile(prob_y, 1 - prop_y)
            mask_y = prob_y > t_y
        else:
            raise ValueError("mech_y must be 'MCAR' or 'MNAR'")

        y_miss = y.copy()
        y_miss[mask_y] = np.nan
    else:
        mask_y = None
        y_miss = None

    return X_miss, mask_X, y_miss, mask_y



