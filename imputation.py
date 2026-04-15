def impute_all(
    X_miss,
    random_state: int = 0,
    # --- MICE (R) only ---
    use_mice: bool = True,
    mice_m: int = 5,
    mice_max_iter: int = 20,
    mice_estimator=None,
    mice_method: str = "pmm",         # e.g. "pmm", "norm", "norm.nob", "rf"
    # --- MissForest (R) ---
    use_missforest: bool = False,
    missforest_max_iter: int = 10,
    # --- Deep learning ---
    use_vae: bool = False,
    use_gan: bool = False,
    vae_kwargs: dict | None = None,
    gan_kwargs: dict | None = None,
    # --- KNN ----
    knn_neighbors: int = 3,
    # --- SimpleNN imputer ---
    use_nn: bool = False,
    nn_kwargs: dict | None = None,
):
    """
    Impute X_miss

    IMPORTANT (parallel safety):
      - MissForest and MICE are run via external Rscript subprocesses.

    Returns a dict of completed matrices (only those actually run):
      - "Mean"
      - "Median"
      - "kNN (k=knn_neighbors)"
      - "Iterative (RandomForest)"
      - "MissForest (R)"                 (if use_missforest=True)
      - "MICE (R; m=..., method=...)"    (if use_mice=True; pooled average of m completes)
      - "VAE" / "GAN (GAIN)"             (if enabled and available)
      - "SimpleNN"                       (if use_nn=True and PyTorch available)

    Prereqs for R methods:
      - A working `Rscript` on PATH (or set env var RSCRIPT=/path/to/Rscript)
      - R packages installed in that R:
          missForest, mice
    """
    import os
    import warnings
    import numpy as np
    import subprocess
    import tempfile
    import pandas as pd

    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.exceptions import ConvergenceWarning


    def _rscript() -> str:
        return os.environ.get("RSCRIPT", "Rscript")

    def _abs_runner(rel_path: str) -> str:
        return os.path.abspath(rel_path)

    def _missforest_R_external(X: np.ndarray, max_iter: int, seed: int) -> np.ndarray:
        """
        Parallel-safe MissForest: run via external Rscript (NOT rpy2).
        """
        X = np.asarray(X, dtype=float, order="C")

        # guard: rows all nan
        row_all_nan = np.isnan(X).all(axis=1)
        if row_all_nan.any():
            col_means = np.nanmean(X, axis=0)
            X = X.copy()
            X[row_all_nan, :] = np.where(np.isfinite(col_means), col_means, 0.0)

        runner_path = _abs_runner("./missforest_runner.R")
        if not os.path.exists(runner_path):
            raise RuntimeError(f"MissForest runner not found: {runner_path}")

        with tempfile.TemporaryDirectory() as td:
            in_csv = os.path.join(td, "in.csv")
            out_csv = os.path.join(td, "out.csv")

            pd.DataFrame(X).to_csv(in_csv, index=False)

            cmd = [_rscript(), runner_path, in_csv, out_csv, str(int(max_iter)), str(int(seed or 0))]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "MissForest via Rscript failed.\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}\n"
                )

            X_imp = pd.read_csv(out_csv).to_numpy(dtype=float)
            return np.asarray(X_imp, dtype=float, order="C")

    def _mice_R_external(X: np.ndarray, m: int, maxit: int, seed: int, method: str) -> list[np.ndarray]:
        """
        MICE: run via external Rscript.
        Returns list of m completed (n x p) numpy arrays.
        """
        X = np.asarray(X, dtype=float, order="C")

        # guards: avoid all-NA cols/rows
        col_all_nan = np.isnan(X).all(axis=0)
        if col_all_nan.any():
            warnings.warn(f"{int(col_all_nan.sum())} column(s) all-NaN; pre-filling 0.0 before R::mice.")
            X = X.copy()
            X[:, col_all_nan] = 0.0

        row_all_nan = np.isnan(X).all(axis=1)
        if row_all_nan.any():
            warnings.warn(f"{int(row_all_nan.sum())} row(s) all-NaN; mean-filling those rows before R::mice.")
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            X = X.copy()
            X[row_all_nan, :] = col_means

        runner_path = _abs_runner("./mice_runner.R")
        if not os.path.exists(runner_path):
            raise RuntimeError(f"MICE runner not found: {runner_path}")

        with tempfile.TemporaryDirectory() as td:
            in_csv = os.path.join(td, "in.csv")
            out_prefix = os.path.join(td, "out")

            pd.DataFrame(X).to_csv(in_csv, index=False)

            cmd = [
                _rscript(), runner_path,
                in_csv, out_prefix,
                str(int(m)), str(int(maxit)),
                str(int(seed or 0)), str(method),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "MICE via Rscript failed.\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}\n"
                )

            completed = []
            for k in range(1, int(m) + 1):
                path_k = f"{out_prefix}_{k}.csv"
                if not os.path.exists(path_k):
                    raise RuntimeError(f"MICE output missing: {path_k}")
                Xk = pd.read_csv(path_k).to_numpy(dtype=float)
                completed.append(np.asarray(Xk, dtype=float, order="C"))

            return completed


    X_miss = np.asarray(X_miss, dtype=float)
    out: dict[str, np.ndarray] = {}

    # Treat ±inf as missing
    finite_mask = np.isfinite(X_miss)
    if not finite_mask[~np.isnan(X_miss)].all():
        warnings.warn("Non-finite values (±inf) found; treating them as missing.")
        X_miss = X_miss.copy()
        X_miss[~finite_mask] = np.nan

    n, p = X_miss.shape
    obs_mask = ~np.isnan(X_miss)

    def _restore_observed(X_hat: np.ndarray) -> np.ndarray:
        X_hat = np.asarray(X_hat, dtype=float)
        X_hat[obs_mask] = X_miss[obs_mask]
        return X_hat

    # ---------- Baselines ----------
    out["Mean"] = _restore_observed(SimpleImputer(strategy="mean").fit_transform(X_miss))
    out["Median"] = _restore_observed(SimpleImputer(strategy="median").fit_transform(X_miss))

    # ----------- kNN ---------
    def _safe_k(k, n_rows):
        return int(max(1, min(int(k), max(1, n_rows - 1))))

    col_all_nan = np.isnan(X_miss).all(axis=0)
    has_all_nan = bool(col_all_nan.any())

    k_eff = _safe_k(knn_neighbors, n)
    if not has_all_nan:
        X_knn = KNNImputer(n_neighbors=k_eff, weights="uniform").fit_transform(X_miss)
    else:
        obs_cols = ~col_all_nan
        X_knn = np.zeros_like(X_miss, dtype=float)
        if obs_cols.any():
            X_knn[:, obs_cols] = KNNImputer(n_neighbors=k_eff, weights="uniform").fit_transform(X_miss[:, obs_cols])
    out[f"kNN (k={k_eff})"] = _restore_observed(X_knn)

    # ---------- IterativeImputer with RandomForest ----------
    rf_imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
            n_jobs=1,
            criterion="squared_error",
        ),
        random_state=random_state,
        max_iter=10,
        initial_strategy="mean",
        sample_posterior=False,
        skip_complete=True,
    )

    if not has_all_nan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            X_rf = rf_imputer.fit_transform(X_miss)
    else:
        obs_cols = ~col_all_nan
        X_rf = np.zeros_like(X_miss, dtype=float)
        if obs_cols.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                X_rf[:, obs_cols] = rf_imputer.fit_transform(X_miss[:, obs_cols])
    out["Iterative (RandomForest)"] = _restore_observed(X_rf)

    # ---------- MissForest (R, external) ----------
    if use_missforest:
        try:
            if not has_all_nan:
                X_mf = _missforest_R_external(X_miss, max_iter=missforest_max_iter, seed=random_state)
            else:
                obs_cols = ~col_all_nan
                X_mf = np.zeros_like(X_miss, dtype=float)
                if obs_cols.any():
                    X_mf[:, obs_cols] = _missforest_R_external(
                        X_miss[:, obs_cols], max_iter=missforest_max_iter, seed=random_state
                    )
            out["MissForest (R)"] = _restore_observed(X_mf)
        except Exception as e:
            warnings.warn(f"MissForest (R) failed/unavailable: {e}")

    # ---------- MICE ----------
    if use_mice:
        if not has_all_nan:
            comps = _mice_R_external(
                X_miss,
                m=int(mice_m),
                maxit=int(mice_max_iter),
                seed=int(random_state if random_state is not None else 0),
                method=str(mice_method),
            )
            X_mice = np.mean(np.stack(comps, axis=0), axis=0)
        else:
            obs_cols = ~col_all_nan
            X_mice = np.zeros_like(X_miss, dtype=float)
            if obs_cols.any():
                comps = _mice_R_external(
                    X_miss[:, obs_cols],
                    m=int(mice_m),
                    maxit=int(mice_max_iter),
                    seed=int(random_state if random_state is not None else 0),
                    method=str(mice_method),
                )
                X_mice[:, obs_cols] = np.mean(np.stack(comps, axis=0), axis=0)

        out[f"MICE (R; m={mice_m})"] = _restore_observed(X_mice)

    # ---------- deep methods ----------
    if use_vae:
        try:
            from VAE import impute_vae
            X_vae = impute_vae(X_miss, random_state=random_state, **(vae_kwargs or {}))
            out["VAE"] = _restore_observed(X_vae)
        except Exception as e:
            warnings.warn(f"Skipping VAE imputation: {e}")

    if use_gan:
        try:
            from GAN import impute_gain
            X_gain = impute_gain(X_miss, random_state=random_state, **(gan_kwargs or {}))
            out["GAN (GAIN)"] = _restore_observed(X_gain)
        except Exception as e:
            warnings.warn(f"Skipping GAN (GAIN) imputation: {e}")

    if use_nn:
        try:
            from NN import impute_nn
            X_nn = impute_nn(X_miss, random_state=random_state, **(nn_kwargs or {}))
            out["SimpleNN"] = _restore_observed(X_nn)
        except Exception as e:
            warnings.warn(f"Skipping SimpleNN imputation: {e}")

    return out






def impute_all_include_y(
    X_miss,
    y=None,
    include_y: bool = False,
    random_state: int = 0,
    scale_y_for_knn: bool = True,   # standardize y when used as auxiliary feature

    # --- MICE (R)  ---
    use_mice: bool = True,
    mice_m: int = 5,
    mice_max_iter: int = 20,
    mice_estimator=None,
    mice_method: str = "pmm", 

    use_missforest: bool = False,
    missforest_max_iter: int = 10,

    use_vae: bool = False,
    use_gan: bool = False,
    vae_kwargs: dict | None = None,
    gan_kwargs: dict | None = None,

    knn_neighbors: int = 3,

    use_nn: bool = False,
    nn_kwargs: dict | None = None,
):
    """
    Works with `impute_all`.
    Returns: dict {method_name: X_imputed (n, p)}
    """
    import numpy as np

    X_miss = np.asarray(X_miss, float)
    n = X_miss.shape[0]

    if not include_y or y is None:
        return impute_all(
            X_miss,
            random_state=random_state,
            # R MICE
            use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
            mice_estimator=mice_estimator, mice_method=mice_method,
            # MissForest (R)
            use_missforest=use_missforest, missforest_max_iter=missforest_max_iter,
            # Deep
            use_vae=use_vae, use_gan=use_gan,
            vae_kwargs=(vae_kwargs or {}), gan_kwargs=(gan_kwargs or {}),
            # kNN
            knn_neighbors=knn_neighbors,
            # SimpleNN
            use_nn=use_nn, nn_kwargs=(nn_kwargs or {}),
        )

    y = np.asarray(y, float).reshape(-1)
    if y.shape[0] != n:
        raise ValueError("y must have the same number of rows as X_miss")

    # Treat y as fully observed auxiliary feature: fill any NaNs safely.
    if np.isnan(y).any():
        import warnings
        warnings.warn("y contains NaNs; filling with its mean before augmentation.")
        m = np.nanmean(y)
        if not np.isfinite(m):  # all-NaN edge case
            m = 0.0
        y = np.where(np.isnan(y), m, y)

    if scale_y_for_knn:
        mu, sd = float(np.mean(y)), float(np.std(y))
        y_aux = (y - mu) / (sd if sd >= 1e-12 else 1.0)
    else:
        y_aux = y.copy()

    X_aug = np.column_stack([X_miss, y_aux])

    res_aug = impute_all(
        X_aug,
        random_state=random_state,
        # R MICE
        use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
        mice_estimator=mice_estimator, mice_method=mice_method,
        # MissForest (R)
        use_missforest=use_missforest, missforest_max_iter=missforest_max_iter,
        # Deep
        use_vae=use_vae, use_gan=use_gan,
        vae_kwargs=(vae_kwargs or {}), gan_kwargs=(gan_kwargs or {}),
        # kNN
        knn_neighbors=knn_neighbors,
        # SimpleNN
        use_nn=use_nn, nn_kwargs=(nn_kwargs or {}),
    )

    out = {name: arr[:, :-1] for name, arr in res_aug.items()}
    return out




def impute_all_drop_ymissingness(
    X_miss,
    y,
    *,
    random_state: int = 0,
    use_mice: bool = True,
    mice_m: int = 5,
    mice_max_iter: int = 20,
    mice_estimator=None, 
    mice_method: str = "pmm",
    use_missforest: bool = False,
    missforest_max_iter: int = 10,
    use_vae: bool = False,
    use_gan: bool = False,
    vae_kwargs: dict | None = None,
    gan_kwargs: dict | None = None,
    knn_neighbors: int = 3,
    use_nn: bool = False,
    nn_kwargs: dict | None = None,
):
    """
    Drop rows where y is missing, then impute X on the remaining rows.

    Returns:
      imputed_dict_sub : dict {method_name: X_imputed_kept (n_kept, p)}
      y_kept           : ndarray (n_kept,)   -- y with no missingness
      kept_idx         : ndarray (n_kept,)   -- original row indices kept
    """
    import numpy as np

    X_miss = np.asarray(X_miss, float)
    y = np.asarray(y, float).reshape(-1)

    if X_miss.shape[0] != y.shape[0]:
        raise ValueError("y must have the same number of rows as X_miss")

    keep = np.isfinite(y)
    if (~keep).all():
        raise ValueError("All rows have missing y; nothing to impute after dropping.")

    kept_idx = np.nonzero(keep)[0].astype(int)
    X_sub = X_miss[keep, :]
    y_kept = y[keep]

    imputed_dict_sub = impute_all(
        X_sub,
        random_state=random_state,
        use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
        mice_estimator=mice_estimator, mice_method=mice_method,
        use_missforest=use_missforest, missforest_max_iter=missforest_max_iter,
        use_vae=use_vae, use_gan=use_gan,
        vae_kwargs=(vae_kwargs or {}), gan_kwargs=(gan_kwargs or {}),
        knn_neighbors=knn_neighbors,
        use_nn=use_nn, nn_kwargs=(nn_kwargs or {}),
    )
    return imputed_dict_sub, y_kept, kept_idx



def impute_all_include_y_drop_ymissingness(
    X_miss,
    y,
    *,
    random_state: int = 0,
    use_mice: bool = True,
    mice_m: int = 5,
    mice_max_iter: int = 20,
    mice_estimator=None,
    mice_method: str = "pmm",
    use_missforest: bool = False,
    missforest_max_iter: int = 10,
    use_vae: bool = False,
    use_gan: bool = False,
    vae_kwargs: dict | None = None,
    gan_kwargs: dict | None = None,
    knn_neighbors: int = 3,
    use_nn: bool = False,
    nn_kwargs: dict | None = None,
):
    """
    Drop rows where y is missing, then impute X on the remaining rows,
    using y (observed) as an auxiliary feature during imputation.

    Returns:
      imputed_dict_sub : dict {method_name: X_imputed_kept (n_kept, p)}
      y_kept           : ndarray (n_kept,)
      kept_idx         : ndarray (n_kept,)
    """
    import numpy as np

    X_miss = np.asarray(X_miss, float)
    y = np.asarray(y, float).reshape(-1)

    if X_miss.shape[0] != y.shape[0]:
        raise ValueError("y must have the same number of rows as X_miss")

    keep = np.isfinite(y)
    if (~keep).all():
        raise ValueError("All rows have missing y; nothing to impute after dropping.")

    kept_idx = np.nonzero(keep)[0].astype(int)
    X_sub = X_miss[keep, :]
    y_kept = y[keep]

    imputed_dict_sub = impute_all_include_y(
        X_sub,
        y=y_kept,
        include_y=True,
        random_state=random_state,
        # R MICE
        use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
        mice_estimator=mice_estimator, mice_method=mice_method,
        # MissForest (R)
        use_missforest=use_missforest, missforest_max_iter=missforest_max_iter,
        # Deep
        use_vae=use_vae, use_gan=use_gan,
        vae_kwargs=(vae_kwargs or {}), gan_kwargs=(gan_kwargs or {}),
        # kNN + SimpleNN
        knn_neighbors=knn_neighbors,
        use_nn=use_nn, nn_kwargs=(nn_kwargs or {}),
    )
    return imputed_dict_sub, y_kept, kept_idx
