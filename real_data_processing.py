import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from itertools import combinations
from imputation import impute_all, impute_all_include_y



def missingness_summary(X: pd.DataFrame):
    """Missingness summary for a feature DataFrame (mixed dtypes OK)."""
    miss = X.isna()

    overall = pd.Series({
        "n_rows": X.shape[0],
        "n_cols": X.shape[1],
        "n_missing": int(miss.values.sum()),
        "pct_missing": float(miss.values.mean() * 100),
        "rows_with_any_missing": int(miss.any(axis=1).sum()),
        "cols_with_any_missing": int(miss.any(axis=0).sum()),
    })

    miss_by_col = pd.DataFrame({
        "dtype": X.dtypes.astype(str),
        "n_missing": miss.sum(axis=0).astype(int),
        "pct_missing": (miss.mean(axis=0) * 100).astype(float),
    }).sort_values(["pct_missing", "n_missing"], ascending=False)


    row_missing_k = miss.sum(axis=1)
    row_hist = (
        row_missing_k.value_counts()
        .sort_index()
        .to_frame("n_rows")
    )
    row_hist["pct_rows"] = row_hist["n_rows"] / X.shape[0] * 100

    return overall, miss_by_col, row_hist


def eval_predictive_xy_test(
    X_train_miss, Y_train_miss,
    X_test_miss,  Y_test_miss,
    *,
    random_state=10,
    use_mice=True, mice_m=5, mice_max_iter=20, mice_method="pmm",
    use_missforest=False, missforest_max_iter=10,
    knn_neighbors=3,
    use_vae=False, use_gan=False, use_nn=False,
    vae_kwargs=None, gan_kwargs=None, nn_kwargs=None,
    model_type="logreg",
    ridge_alpha=1.0,
    standardize=True,
    rf_kwargs=None,
    gbrt_kwargs=None,
    mlp_kwargs=None,
    require_y_test_observed="any",
    min_train_obs=20,
    fill_x_with_train_means=True,
    k_missing=1,
    threshold=0.5,
    verbose=True,
):
    import numpy as np
    import pandas as pd

    from imputation import impute_all, impute_all_include_y

    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    def _to_float(a):
        if isinstance(a, pd.DataFrame):
            a = a.to_numpy()
        return np.asarray(a, dtype=float)

    def _check_binary_1d(y):
        y = _to_float(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D. Got shape {y.shape}")
        obs = np.isfinite(y)
        if obs.any():
            vals = np.unique(y[obs])
            if not np.all(np.isin(vals, [0.0, 1.0])):
                raise ValueError("Y must be binary {0,1} (NaN allowed).")
        return y

    def _fill_nans(X_tr, X_te):
        X_tr_df = pd.DataFrame(X_tr, columns=X_cols)
        X_te_df = pd.DataFrame(X_te, columns=X_cols)
        tr_means = X_tr_df.mean()
        if fill_x_with_train_means:
            X_tr_df = X_tr_df.fillna(tr_means)
            X_te_df = X_te_df.fillna(tr_means)
        else:
            X_tr_df = X_tr_df.fillna(tr_means)
            X_te_df = X_te_df.fillna(X_te_df.mean())
        return X_tr_df.to_numpy(), X_te_df.to_numpy()

    def _make_model():
        mt = str(model_type).lower()

        if mt in ("logreg", "logistic"):
            C = 1.0 / max(float(ridge_alpha), 1e-12)
            base = LogisticRegression(C=C, solver="liblinear", max_iter=2000)
            return make_pipeline(StandardScaler(), base) if standardize else base

        if mt == "rf":
            rf = dict(
                n_estimators=400,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=0,
            )
            rf.update(rf_kwargs or {})
            return RandomForestClassifier(**rf)

        if mt == "gbrt":
            gb = dict(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=0,
            )
            gb.update(gbrt_kwargs or {})
            return GradientBoostingClassifier(**gb)

        if mt == "mlp":
            mlp = dict(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                max_iter=300,
                random_state=0,
            )
            mlp.update(mlp_kwargs or {})
            base = MLPClassifier(**mlp)
            return make_pipeline(StandardScaler(), base) if standardize else base

        raise ValueError(f"Unknown model_type={model_type}")

    def _eligible_test_mask(y_test, X_test_miss):
        y_ok = np.isfinite(y_test)
        if str(require_y_test_observed).lower() == "all":
            if not y_ok.all():
                raise ValueError("require_y_test_observed='all' but some Y_test are missing.")
            y_ok = np.ones_like(y_ok, dtype=bool)

        x_missing_ct = np.isnan(X_test_miss).sum(axis=1)
        x_ok = x_missing_ct >= int(k_missing)
        return y_ok & x_ok

    X_train_miss = _to_float(X_train_miss)
    X_test_miss  = _to_float(X_test_miss)
    y_train = _check_binary_1d(Y_train_miss)
    y_test  = _check_binary_1d(Y_test_miss)

    n_tr, p = X_train_miss.shape
    n_te, p2 = X_test_miss.shape
    if p != p2:
        raise ValueError("Train/test X must have same number of columns")

    X_cols = [f"X{j}" for j in range(p)]

    eligible_test_rows = _eligible_test_mask(y_test, X_test_miss)

    if verbose:
        print(f"Scoring rows: {eligible_test_rows.sum()} / {len(y_test)}")

    imputed_train = impute_all(
        X_train_miss, random_state=random_state,
        use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
        mice_method=mice_method, use_missforest=use_missforest,
        missforest_max_iter=missforest_max_iter, knn_neighbors=knn_neighbors,
        use_vae=use_vae, use_gan=use_gan, use_nn=use_nn,
        vae_kwargs=vae_kwargs, gan_kwargs=gan_kwargs, nn_kwargs=nn_kwargs
    )

    imputed_test = impute_all(
        X_test_miss, random_state=random_state,
        use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
        mice_method=mice_method, use_missforest=use_missforest,
        missforest_max_iter=missforest_max_iter, knn_neighbors=knn_neighbors,
        use_vae=use_vae, use_gan=use_gan, use_nn=use_nn,
        vae_kwargs=vae_kwargs, gan_kwargs=gan_kwargs, nn_kwargs=nn_kwargs
    )

    imputed_y_train = impute_all_include_y(
        X_train_miss, y_train, include_y=True, random_state=random_state,
        use_mice=use_mice, mice_m=mice_m, mice_max_iter=mice_max_iter,
        mice_method=mice_method, use_missforest=use_missforest,
        missforest_max_iter=missforest_max_iter, knn_neighbors=knn_neighbors,
        use_vae=use_vae, use_gan=use_gan, use_nn=use_nn,
        vae_kwargs=vae_kwargs, gan_kwargs=gan_kwargs, nn_kwargs=nn_kwargs
    )

    imputed_y_test = imputed_test

    def _score_family(family, train_dict, test_dict):
        rows = []

        obs_tr = np.isfinite(y_train)
        if obs_tr.sum() < min_train_obs:
            return pd.DataFrame()

        y_tr = y_train[obs_tr].astype(int)
        if np.unique(y_tr).size < 2:
            return pd.DataFrame()

        obs_te = eligible_test_rows
        y_true = y_test[obs_te].astype(int)
        if y_true.size == 0:
            return pd.DataFrame()

        for method in sorted(set(train_dict) & set(test_dict)):
            X_tr, X_te = train_dict[method], test_dict[method]

            if X_tr.shape != X_train_miss.shape:
                raise ValueError(
                    f"{method}: train imputed shape {X_tr.shape} != expected {X_train_miss.shape}"
                )
            if X_te.shape != X_test_miss.shape:
                raise ValueError(
                    f"{method}: test imputed shape {X_te.shape} != expected {X_test_miss.shape}"
                )

            X_tr, X_te = _fill_nans(X_tr, X_te)

            clf = _make_model()
            clf.fit(X_tr[obs_tr], y_tr)

            prob = np.clip(clf.predict_proba(X_te[obs_te])[:, 1], 1e-12, 1 - 1e-12)

            brier = float(np.mean((prob - y_true) ** 2))
            rmse = float(np.sqrt(brier))
            bias = float(np.mean(prob - y_true))

            pred = (prob >= threshold).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())

            sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
            specificity = tn / (tn + fp) if (tn + fp) else np.nan

            rows.append({
                "family": family,
                "method": method,
                "model_type": model_type,
                "threshold": threshold,
                "auroc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
                "auprc": average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
                "brier": brier,
                "rmse": rmse,
                "bias": bias,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "n_test_scored": int(len(y_true)),
                "pos_rate_test": float(y_true.mean()),
            })

        return pd.DataFrame(rows)

    df1 = _score_family("Without y", imputed_train, imputed_test)
    df2 = _score_family("With y", imputed_y_train, imputed_y_test)

    out = pd.concat([df1, df2], ignore_index=True)
    if out.empty:
        return out

    return (
        out.sort_values(["family", "auprc", "auroc"], ascending=[True, False, False])
           .reset_index(drop=True)
    )

    

def plot_auroc_auprc(
    df,
    *,
    families=("Without y", "With y"),
    sort_by="auprc",
    zoom_margin=0.02,
    offset=0.14,
    point_size=50,
    line_alpha=0.25,
    save_path=None,
    dpi=300
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    required_cols = {"family", "method", "auroc", "auprc"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"`df` must contain columns {sorted(required_cols)}. "
            f"Missing columns: {sorted(missing_cols)}"
        )

    wide = df.pivot(index="method", columns="family", values=["auroc", "auprc"])
    methods = wide.index.tolist()

    if sort_by in ("auroc", "auprc"):
        if (sort_by, families[0]) in wide.columns and (sort_by, families[1]) in wide.columns:
            delta = wide[(sort_by, families[1])] - wide[(sort_by, families[0])]
            methods = delta.sort_values(ascending=False).index.tolist()

    x = np.arange(len(methods))

    def _make_metric_save_path(base_path, metric):
        if base_path is None:
            return None
        root, ext = os.path.splitext(base_path)
        if ext == "":
            ext = ".png"
        return f"{root}_{metric}{ext}"

    def _draw_panel(metric, panel_title, y_label, metric_save_path=None):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        values = []
        for fam in families:
            if (metric, fam) in wide.columns:
                values.append(wide.loc[methods, (metric, fam)].to_numpy())

        if values:
            all_values = np.concatenate([v[np.isfinite(v)] for v in values if v.size])
        else:
            all_values = np.array([0.0, 1.0], dtype=float)

        if all_values.size == 0:
            all_values = np.array([0.0, 1.0], dtype=float)

        lower = float(np.nanmin(all_values) - zoom_margin)
        upper = float(np.nanmax(all_values) + zoom_margin)
        lower = max(lower, 0.0)
        upper = min(upper, 1.0)

        if upper - lower < 0.05:
            midpoint = (upper + lower) / 2
            lower = max(midpoint - 0.03, 0.0)
            upper = min(midpoint + 0.03, 1.0)

        y_first = (
            wide.loc[methods, (metric, families[0])].to_numpy()
            if (metric, families[0]) in wide.columns else None
        )
        y_second = (
            wide.loc[methods, (metric, families[1])].to_numpy()
            if (metric, families[1]) in wide.columns else None
        )

        if y_first is not None and y_second is not None:
            valid = np.isfinite(y_first) & np.isfinite(y_second)
            for i in np.where(valid)[0]:
                ax.plot(
                    [x[i] - offset / 2, x[i] + offset / 2],
                    [y_first[i], y_second[i]],
                    linewidth=1.5,
                    alpha=line_alpha,
                )

        for i, fam in enumerate(families):
            if (metric, fam) not in wide.columns:
                continue

            y = wide.loc[methods, (metric, fam)].to_numpy()
            valid = np.isfinite(y)

            if i == 0:
                ax.scatter(
                    x[valid] + (i - 0.5) * offset,
                    y[valid],
                    marker="o",
                    s=point_size,
                    facecolors="none",
                    edgecolors="#1f77b4",
                    linewidths=1.8,
                    alpha=1,
                    label=fam,
                )
            else:
                ax.scatter(
                    x[valid] + (i - 0.5) * offset,
                    y[valid],
                    marker="v",
                    s=point_size,
                    facecolors="none",
                    edgecolors="#2ca02c",
                    linewidths=1.8,
                    alpha=1,
                    label=fam,
                )

        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)

        ax.set_title(panel_title)
        ax.set_ylabel(y_label)
        ax.set_ylim(lower, upper)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.legend(title="Analysis family", frameon=False)

        plt.tight_layout()

        if metric_save_path is not None:
            plt.savefig(metric_save_path, dpi=dpi, bbox_inches="tight")

        plt.show()

    _draw_panel(
        metric="auprc",
        panel_title="Area Under the Precision-Recall Curve",
        y_label="AUPRC",
        metric_save_path=_make_metric_save_path(save_path, "auprc"),
    )

    _draw_panel(
        metric="auroc",
        panel_title="Area Under the ROC Curve",
        y_label="AUROC",
        metric_save_path=_make_metric_save_path(save_path, "auroc"),
    )




def plot_brier_bias(
    df,
    *,
    families=("Without y", "With y"),
    sort_by="brier",
    zoom_margin=0.02,
    offset=0.14,
    point_size=70,
    line_alpha=0.25,
    save_path=None,
    dpi=300
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    required_cols = {"family", "method", "brier", "bias"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"`df` must contain columns {sorted(required_cols)}. "
            f"Missing columns: {sorted(missing_cols)}"
        )

    wide = df.pivot(index="method", columns="family", values=["brier", "bias"])
    methods = wide.index.tolist()

    if sort_by in ("brier", "bias"):
        if (sort_by, families[0]) in wide.columns and (sort_by, families[1]) in wide.columns:
            delta = wide[(sort_by, families[1])] - wide[(sort_by, families[0])]
            methods = delta.sort_values(
                ascending=True if sort_by == "brier" else False
            ).index.tolist()

    x = np.arange(len(methods))

    def _collect_values(metric):
        vals = []
        for fam in families:
            if (metric, fam) in wide.columns:
                vals.append(wide.loc[methods, (metric, fam)].to_numpy())

        if not vals:
            return np.array([0.0, 1.0], dtype=float)

        combined = np.concatenate([arr[np.isfinite(arr)] for arr in vals if arr.size])
        return combined if combined.size else np.array([0.0, 1.0], dtype=float)

    def _make_metric_save_path(base_path, metric):
        if base_path is None:
            return None
        root, ext = os.path.splitext(base_path)
        if ext == "":
            ext = ".png"
        return f"{root}_{metric}{ext}"

    def _draw_panel(metric, panel_title, y_label, center_at_zero=False, force_nonnegative=False, metric_save_path=None):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        all_values = _collect_values(metric)

        lower = float(np.nanmin(all_values) - zoom_margin)
        upper = float(np.nanmax(all_values) + zoom_margin)

        if force_nonnegative:
            lower = max(lower, 0.0)

        if center_at_zero and np.isfinite(lower) and np.isfinite(upper):
            bound = max(abs(lower), abs(upper))
            lower, upper = -bound, bound
            if (upper - lower) < 0.05:
                lower, upper = -0.03, 0.03

        if np.isfinite(lower) and np.isfinite(upper) and (upper - lower) < 0.05:
            midpoint = (upper + lower) / 2
            lower = midpoint - 0.03
            upper = midpoint + 0.03
            if force_nonnegative:
                lower = max(lower, 0.0)

        y_first = (
            wide.loc[methods, (metric, families[0])].to_numpy()
            if (metric, families[0]) in wide.columns else None
        )
        y_second = (
            wide.loc[methods, (metric, families[1])].to_numpy()
            if (metric, families[1]) in wide.columns else None
        )

        if y_first is not None and y_second is not None:
            valid = np.isfinite(y_first) & np.isfinite(y_second)
            for i in np.where(valid)[0]:
                ax.plot(
                    [x[i] - offset / 2, x[i] + offset / 2],
                    [y_first[i], y_second[i]],
                    linewidth=1.5,
                    alpha=line_alpha,
                )

        for i, fam in enumerate(families):
            if (metric, fam) not in wide.columns:
                continue

            y = wide.loc[methods, (metric, fam)].to_numpy()
            valid = np.isfinite(y)

            if i == 0:
                ax.scatter(
                    x[valid] + (i - 0.5) * offset,
                    y[valid],
                    marker="o",
                    s=point_size,
                    facecolors="none",
                    edgecolors="#1f77b4",
                    linewidths=1.8,
                    alpha=1,
                    label=fam,
                )
            else:
                ax.scatter(
                    x[valid] + (i - 0.5) * offset,
                    y[valid],
                    marker="v",
                    s=point_size,
                    facecolors="none",
                    edgecolors="#2ca02c",
                    linewidths=1.8,
                    alpha=1,
                    label=fam,
                )

        if center_at_zero:
            ax.axhline(0.0, linewidth=1.0, alpha=0.4)

        ax.set_title(panel_title)
        ax.set_ylabel(y_label)
        ax.set_ylim(lower, upper)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.legend(title="Analysis family", frameon=False)

        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)

        plt.tight_layout()

        if metric_save_path is not None:
            plt.savefig(metric_save_path, dpi=dpi, bbox_inches="tight")

        plt.show()

    _draw_panel(
        metric="brier",
        panel_title="Brier Score",
        y_label="Brier score",
        center_at_zero=False,
        force_nonnegative=True,
        metric_save_path=_make_metric_save_path(save_path, "brier"),
    )

    _draw_panel(
        metric="bias",
        panel_title="Estimation Bias",
        y_label=r"Bias",
        center_at_zero=True,
        force_nonnegative=False,
        metric_save_path=_make_metric_save_path(save_path, "bias"),
    )