import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#Parameter recovery
def plot_param_recovery_mech_means(
    df: pd.DataFrame,
    metric: str = "coef_RMSE",
    *,
    mech_col: str = "mech",
    method_col: str = "method",
    scenario_col: str = "scenario",
    baseline_method: str = "NoMissing",
    baseline_value: float | None = None,
    include_baseline_line: bool = True,
    figsize=(10, 8),
    title: str | None = None,
    xlabel: str | None = None,
    mech_markers: dict[str, str] | None = None,
    mech_colors: dict[str, str] | None = None,
):
    """
    Point plot of mean(metric) across replications, grouped by missingness mechanism.
    Y-axis: methods/scenarios, X-axis: chosen metric.

    Methods are grouped together on the y-axis, for example:
        Mean w/o y
        Mean with y
        Mean drop-y w/o y
        Mean drop-y with y
        MICE w/o y
        MICE with y
        ...

    Handles scenarios such as:
      - Impute w/o y
      - Impute with y
      - Drop y-miss → impute w/o y
      - Drop y-miss → impute + y
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    required = {metric, mech_col, method_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    def _canon_mech(x):
        s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        if s in {"SLIGHTLY_MAR", "SLIGHTLYMAR", "SLIGHT_MAR"}:
            return "SLIGHTLY_MAR"
        return s

    df[mech_col] = df[mech_col].map(_canon_mech)

    def _canon_scenario(x):
        s = str(x).strip().lower()
        s = s.replace("→", "->")
        s = " ".join(s.split())

        if s in {"impute w/o y", "impute without y", "without y", "w/o y"}:
            return "Impute w/o y"
        if s in {"impute with y", "with y", "impute + y"}:
            return "Impute with y"
        if s in {
            "drop y-miss -> impute w/o y",
            "drop y-missingness -> impute w/o y",
            "drop y miss -> impute w/o y",
        }:
            return "Drop y-miss → impute w/o y"
        if s in {
            "drop y-miss -> impute + y",
            "drop y-missingness -> impute + y",
            "drop y miss -> impute + y",
            "drop y-miss -> impute with y",
        }:
            return "Drop y-miss → impute + y"

        return str(x)

    if scenario_col in df.columns:
        df[scenario_col] = df[scenario_col].map(_canon_scenario)
    else:
        df[scenario_col] = ""

    def _display_method(row):
        m = str(row[method_col]).strip()
        scen = str(row.get(scenario_col, "")).strip()

        if m == baseline_method:
            if scen == "Drop y-miss → impute w/o y":
                return 'No Missings ("truth", drop y) w/o y'
            if scen == "Drop y-miss → impute + y":
                return 'No Missings ("truth", drop y) + y'
            if scen == "Impute with y":
                return 'No Missings ("truth") with y'
            return 'No Missings ("truth") w/o y'

        if m == "Complete Case":
            if scen == "Drop y-miss → impute w/o y":
                return "Complete Case (drop y) w/o y"
            if scen == "Drop y-miss → impute + y":
                return "Complete Case (drop y) + y"
            if scen == "Impute with y":
                return "Complete Case with y"
            return "Complete Case w/o y"

        if m == "Drop Missingness":
            return "Drop Missingness"

        if scen == "Impute with y":
            return f"{m} with y"
        if scen == "Impute w/o y":
            return f"{m} w/o y"
        if scen == "Drop y-miss → impute w/o y":
            return f"{m} drop-y w/o y"
        if scen == "Drop y-miss → impute + y":
            return f"{m} drop-y with y"

        return m

    df["display_method"] = df.apply(_display_method, axis=1)

    agg = (
        df[[mech_col, "display_method", metric]]
        .dropna(subset=[metric, mech_col, "display_method"])
        .groupby([mech_col, "display_method"], as_index=False)[metric]
        .mean()
    )
    if agg.empty:
        raise ValueError("No non-NaN values for the chosen metric; nothing to plot.")


    def _split_display_name(name: str):
        s = str(name)

        special_map = {
            'No Missings ("truth") w/o y': ('No Missings ("truth")', 1),
            'No Missings ("truth") with y': ('No Missings ("truth")', 2),
            'No Missings ("truth", drop y) w/o y': ('No Missings ("truth")', 3),
            'No Missings ("truth", drop y) + y': ('No Missings ("truth")', 4),

            "Complete Case w/o y": ("Complete Case", 1),
            "Complete Case with y": ("Complete Case", 2),
            "Complete Case (drop y) w/o y": ("Complete Case", 3),
            "Complete Case (drop y) + y": ("Complete Case", 4),

            "Drop Missingness": ("Drop Missingness", 0),
        }

        if s in special_map:
            return special_map[s]

        if s.endswith(" w/o y"):
            return (s[:-6], 1)
        if s.endswith(" with y"):
            return (s[:-7], 2)
        if " drop-y w/o y" in s:
            return (s.replace(" drop-y w/o y", ""), 3)
        if " drop-y with y" in s:
            return (s.replace(" drop-y with y", ""), 4)

        return (s, 99)

    base_method_order = [
        'No Missings ("truth")',
        "Complete Case",
        "Drop Missingness",
    ]

    discovered_base_methods = []
    for name in agg["display_method"].unique():
        base, _ = _split_display_name(name)
        if base not in discovered_base_methods and base not in base_method_order:
            discovered_base_methods.append(base)

    full_base_order = base_method_order + sorted(discovered_base_methods)

    def _method_rank(name):
        base, scen_rank = _split_display_name(name)
        try:
            base_rank = full_base_order.index(base)
        except ValueError:
            base_rank = 999
        return (base_rank, scen_rank, str(name))

    method_order = sorted(agg["display_method"].unique(), key=_method_rank)

    preferred_mech_order = ["MCAR", "SLIGHTLY_MAR", "MAR", "MNAR"]
    mech_unique = list(agg[mech_col].unique())
    mech_order = [m for m in preferred_mech_order if m in mech_unique] + [
        m for m in mech_unique if m not in preferred_mech_order
    ]


    if mech_markers is None:
        mech_markers = {
            "MCAR": "o",
            "SLIGHTLY_MAR": "+",
            "MAR": "v",
            "MNAR": "x",
        }
    if mech_colors is None:
        mech_colors = {
            "MCAR": "#1f77b4",
            "SLIGHTLY_MAR": "#ff7f0e",
            "MAR": "#2ca02c",
            "MNAR": "#d62728",
        }

    def _marker_for(mech):
        return mech_markers.get(mech, "o")

    def _color_for(mech):
        return mech_colors.get(mech, "black")

    point_size = 28
    edge_lw = 1.0
    legend_ms = 5
    legend_edge_lw = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(method_order))

    for y in y_pos:
        ax.axhline(y, linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)

    for mech in mech_order:
        mdf = agg[agg[mech_col] == mech]
        mk = _marker_for(mech)
        col = _color_for(mech)

        for i, method_name in enumerate(method_order):
            row = mdf[mdf["display_method"] == method_name]
            if row.empty:
                continue

            x = row[metric].iloc[0]
            if pd.isna(x):
                continue

            if mk in {"+", "x"}:
                ax.scatter(
                    float(x),
                    y_pos[i],
                    marker=mk,
                    color=col,
                    linewidths=edge_lw,
                    s=point_size,
                    zorder=3,
                )
            else:
                ax.scatter(
                    float(x),
                    y_pos[i],
                    marker=mk,
                    facecolors="none",
                    edgecolors=col,
                    linewidths=edge_lw,
                    s=point_size,
                    zorder=3,
                )


    if include_baseline_line:
        if baseline_value is None:
            base_rows = agg.loc[
                agg["display_method"].str.contains(
                    r'^No Missings \("truth"\)', regex=True, na=False
                ),
                metric,
            ].dropna()

            if base_rows.empty:
                base_df = df.loc[df[method_col] == baseline_method, metric].dropna()
                if base_df.empty:
                    raise ValueError("No baseline rows found and no baseline_value provided.")
                baseline_value = float(np.median(base_df.to_numpy()))
            else:
                baseline_value = float(np.median(base_rows.to_numpy()))

        if baseline_value is not None and np.isfinite(baseline_value):
            ax.axvline(
                float(baseline_value),
                linestyle="--",
                linewidth=0.8,
                color="gray",
                zorder=1,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_order)
    ax.invert_yaxis()

    vals = agg[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size:
        is_bias = "bias" in metric.lower()
        if is_bias:
            vmax = float(np.max(np.abs(vals)))
            if vmax > 0:
                ax.set_xlim(-1.1 * vmax, 1.1 * vmax)
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            pad = 0.03 * max(1.0, vmax - vmin)
            left = min(0.0, vmin - pad) if vmin >= 0 else vmin - pad
            ax.set_xlim(left, vmax + pad)

    # x-axis label
    if xlabel is not None:
        xlab = xlabel
    else:
        label_map = {
            "coef_Bias": "Bias",
            "coef_bias": "Bias",
            "coef_rmse": "RMSE",
            "coef_RMSE": "RMSE",
        }
        xlab = label_map.get(metric, metric)

    ax.set_xlabel(xlab)
    ax.set_title(title or f"Parameter Recovery ({xlab})")
    ax.grid(False)

    legend_elems = []
    for mech in mech_order:
        mk = _marker_for(mech)
        col = _color_for(mech)

        if mk in {"+", "x"}:
            legend_elems.append(
                Line2D(
                    [0], [0],
                    marker=mk,
                    linestyle="",
                    color=col,
                    label=str(mech),
                    markersize=legend_ms,
                )
            )
        else:
            legend_elems.append(
                Line2D(
                    [0], [0],
                    marker=mk,
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor=col,
                    markeredgewidth=legend_edge_lw,
                    label=str(mech),
                    markersize=legend_ms,
                )
            )

    ax.legend(
        handles=legend_elems,
        title="Mechanism",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    return fig, ax



#Prediction
def plot_pred_mech_means(
    df: pd.DataFrame,
    metric: str = "pred_rmse",
    *,
    mech_col: str = "mech",
    method_col: str = "method",
    scenario_col: str = "scenario",
    baseline_method: str = "NoMissing",
    baseline_value: float | None = None,
    include_baseline_line: bool = True,
    figsize=(10, 8),
    title: str | None = None,
    xlabel: str | None = None,
    mech_markers: dict[str, str] | None = None,
    mech_colors: dict[str, str] | None = None,
):
    """
    Point plot of mean(metric) across replications, grouped by missingness mechanism.
    Y-axis: methods/scenarios, X-axis: predictive metric.

    Handles scenarios such as:
      - Impute w/o y
      - Impute with y
      - Drop y-miss → impute w/o y
      - Drop y-miss → impute + y
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    required = {metric, mech_col, method_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    def _canon_mech(x):
        s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        if s in {"SLIGHTLY_MAR", "SLIGHTLYMAR", "SLIGHT_MAR"}:
            return "SLIGHTLY_MAR"
        return s

    df[mech_col] = df[mech_col].map(_canon_mech)

    def _canon_scenario(x):
        s = str(x).strip().lower()
        s = s.replace("→", "->")
        s = " ".join(s.split())

        if s in {"impute w/o y", "impute without y", "without y", "w/o y"}:
            return "Impute w/o y"
        if s in {"impute with y", "with y", "impute + y"}:
            return "Impute with y"
        if s in {
            "drop y-miss -> impute w/o y",
            "drop y-missingness -> impute w/o y",
            "drop y miss -> impute w/o y",
        }:
            return "Drop y-miss → impute w/o y"
        if s in {
            "drop y-miss -> impute + y",
            "drop y-missingness -> impute + y",
            "drop y miss -> impute + y",
            "drop y-miss -> impute with y",
        }:
            return "Drop y-miss → impute + y"

        return str(x)

    if scenario_col in df.columns:
        df[scenario_col] = df[scenario_col].map(_canon_scenario)
    else:
        df[scenario_col] = ""

    def _display_method(row):
        m = str(row[method_col]).strip()
        scen = str(row.get(scenario_col, "")).strip()

        if m == baseline_method:
            if scen == "Drop y-miss → impute w/o y":
                return 'No Missings ("truth", drop y-miss) w/o y'
            if scen == "Drop y-miss → impute + y":
                return 'No Missings ("truth", drop y-miss) with y'
            if scen == "Impute with y":
                return 'No Missings ("truth") with y'
            return 'No Missings ("truth") w/o y'

        if m == "Complete Case":
            if scen == "Drop y-miss → impute w/o y":
                return "Complete Case (drop y-miss) w/o y"
            if scen == "Drop y-miss → impute + y":
                return "Complete Case (drop y-miss) with y"
            if scen == "Impute with y":
                return "Complete Case with y"
            return "Complete Case w/o y"

        if m == "Drop Missingness":
            return "Drop Missingness"

        if scen == "Impute with y":
            return f"{m} with y"
        if scen == "Impute w/o y":
            return f"{m} w/o y"
        if scen == "Drop y-miss → impute w/o y":
            return f"{m} drop-y w/o y"
        if scen == "Drop y-miss → impute + y":
            return f"{m} drop-y with y"

        return m

    df["display_method"] = df.apply(_display_method, axis=1)


    agg = (
        df[[mech_col, "display_method", metric]]
        .dropna(subset=[mech_col, "display_method", metric])
        .groupby([mech_col, "display_method"], as_index=False)[metric]
        .mean()
    )
    if agg.empty:
        raise ValueError("No non-NaN values for the chosen metric; nothing to plot.")


    def _split_display_name(name: str):
        s = str(name)

        special_map = {
            'No Missings ("truth") w/o y': ('No Missings ("truth")', 1),
            'No Missings ("truth") with y': ('No Missings ("truth")', 2),
            'No Missings ("truth", drop y-miss) w/o y': ('No Missings ("truth")', 3),
            'No Missings ("truth", drop y-miss) with y': ('No Missings ("truth")', 4),

            "Complete Case w/o y": ("Complete Case", 1),
            "Complete Case with y": ("Complete Case", 2),
            "Complete Case (drop y-miss) w/o y": ("Complete Case", 3),
            "Complete Case (drop y-miss) with y": ("Complete Case", 4),

            "Drop Missingness": ("Drop Missingness", 0),
        }
        if s in special_map:
            return special_map[s]

        if s.endswith(" w/o y"):
            return (s[:-6], 1)
        if s.endswith(" with y"):
            return (s[:-7], 2)
        if " drop-y w/o y" in s:
            return (s.replace(" drop-y w/o y", ""), 3)
        if " drop-y with y" in s:
            return (s.replace(" drop-y with y", ""), 4)

        return (s, 99)

    base_method_order = [
        'No Missings ("truth")',
        "Complete Case",
        "Drop Missingness",
    ]

    discovered_base_methods = []
    for name in agg["display_method"].unique():
        base, _ = _split_display_name(name)
        if base not in discovered_base_methods and base not in base_method_order:
            discovered_base_methods.append(base)

    full_base_order = base_method_order + sorted(discovered_base_methods)

    def _method_rank(name):
        base, scen_rank = _split_display_name(name)
        try:
            base_rank = full_base_order.index(base)
        except ValueError:
            base_rank = 999
        return (base_rank, scen_rank, str(name))

    method_order = sorted(agg["display_method"].unique(), key=_method_rank)


    preferred_mech_order = ["MCAR", "SLIGHTLY_MAR", "MAR", "MNAR"]
    mech_unique = list(agg[mech_col].unique())
    mech_order = [m for m in preferred_mech_order if m in mech_unique] + [
        m for m in mech_unique if m not in preferred_mech_order
    ]


    if mech_markers is None:
        mech_markers = {
            "MCAR": "o",
            "SLIGHTLY_MAR": "+",
            "MAR": "v",
            "MNAR": "x",
        }
    if mech_colors is None:
        mech_colors = {
            "MCAR": "#1f77b4",
            "SLIGHTLY_MAR": "#ff7f0e",
            "MAR": "#2ca02c",
            "MNAR": "#d62728",
        }

    def _marker_for(mech):
        return mech_markers.get(mech, "o")

    def _color_for(mech):
        return mech_colors.get(mech, "black")

    point_size = 28
    edge_lw = 1.0
    legend_ms = 5
    legend_edge_lw = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(method_order))

    for y in y_pos:
        ax.axhline(y, linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)

    for mech in mech_order:
        mdf = agg[agg[mech_col] == mech]
        mk = _marker_for(mech)
        col = _color_for(mech)

        for i, method_name in enumerate(method_order):
            row = mdf[mdf["display_method"] == method_name]
            if row.empty:
                continue

            x = row[metric].iloc[0]
            if pd.isna(x):
                continue

            if mk in {"+", "x"}:
                ax.scatter(
                    float(x),
                    y_pos[i],
                    marker=mk,
                    color=col,
                    linewidths=edge_lw,
                    s=point_size,
                    zorder=3,
                )
            else:
                ax.scatter(
                    float(x),
                    y_pos[i],
                    marker=mk,
                    facecolors="none",
                    edgecolors=col,
                    linewidths=edge_lw,
                    s=point_size,
                    zorder=3,
                )


    if include_baseline_line:
        if baseline_value is None:
            base_rows = agg.loc[
                agg["display_method"].str.contains(
                    r'^No Missings \("truth"', regex=True, na=False
                ),
                metric,
            ].dropna()

            if base_rows.empty:
                base_df = df.loc[df[method_col] == baseline_method, metric].dropna()
                if base_df.empty:
                    raise ValueError("No baseline rows found and no baseline_value provided.")
                baseline_value = float(np.median(base_df.to_numpy()))
            else:
                baseline_value = float(np.median(base_rows.to_numpy()))

        if baseline_value is not None and np.isfinite(baseline_value):
            ax.axvline(
                float(baseline_value),
                linestyle="--",
                linewidth=0.8,
                color="gray",
                zorder=1,
            )


    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_order)
    ax.invert_yaxis()

    vals = agg[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size:
        is_bias = "bias" in metric.lower()
        if is_bias:
            vmax = float(np.max(np.abs(vals)))
            if vmax > 0:
                ax.set_xlim(-1.1 * vmax, 1.1 * vmax)
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            pad = 0.03 * max(1.0, vmax - vmin)
            left = min(0.0, vmin - pad) if vmin >= 0 else vmin - pad
            ax.set_xlim(left, vmax + pad)

    # x-axis label
    if xlabel is not None:
        xlab = xlabel
    else:
        label_map = {
            "pred_rmse": "RMSE",
            "mse": "MSE",
            "pred_bias": "Bias",
        }
        xlab = label_map.get(metric.lower(), metric)

    ax.set_xlabel(xlab)
    ax.set_title(title or f"{xlab} by method and mechanism")
    ax.grid(False)

    legend_elems = []
    for mech in mech_order:
        mk = _marker_for(mech)
        col = _color_for(mech)

        if mk in {"+", "x"}:
            legend_elems.append(
                Line2D(
                    [0], [0],
                    marker=mk,
                    linestyle="",
                    color=col,
                    label=str(mech),
                    markersize=legend_ms,
                )
            )
        else:
            legend_elems.append(
                Line2D(
                    [0], [0],
                    marker=mk,
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor=col,
                    markeredgewidth=legend_edge_lw,
                    label=str(mech),
                    markersize=legend_ms,
                )
            )

    ax.legend(
        handles=legend_elems,
        title="Mechanism",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    return fig, ax




#classification
def plot_pred_accuracy_mech_means(
    df: pd.DataFrame,
    *,
    mech_col: str = "mech",
    method_col: str = "method",
    scenario_col: str = "scenario",
    unit_col: str = "rep",              # kept for API compatibility (not used)
    baseline_name: str = "NoMissing",
    title: str = "Classification Accuracy by Method and Mechanism",
    include_baseline_line: bool = True,
    figsize=(6, 4),
    mech_markers: dict[str, str] | None = None,
    mech_colors: dict[str, str] | None = None,
    accuracy_col: str = "accuracy",
):
    """
    Point plot of mean accuracy across replications, grouped by missingness mechanism.
    Y-axis: methods (display labels), X-axis: accuracy.
    """
    required = {accuracy_col, mech_col, method_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    def _canon_mech(x):
        s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        if s in {"SLIGHTLY_MAR", "SLIGHTLYMAR", "SLIGHT_MAR"}:
            return "SLIGHTLY_MAR"
        return s

    df[mech_col] = df[mech_col].map(_canon_mech)

    def _display_method(row):
        m = row[method_col]
        if m == baseline_name:
            return 'No Missings ("truth")'
        if m == "Drop Missingness":
            return "Drop Missingness"

        scen = str(row.get(scenario_col, "")).lower()
        if "with y" in scen:
            return f"{m} w y"
        if ("w/o y" in scen) or ("without y" in scen) or ("no y" in scen):
            return f"{m} w/o y"
        return str(m)

    df["display_method"] = df.apply(_display_method, axis=1)

    agg = (
        df[[mech_col, "display_method", accuracy_col]]
        .dropna(subset=[mech_col, "display_method", accuracy_col])
        .groupby([mech_col, "display_method"], as_index=False)[accuracy_col]
        .mean()
    )
    if agg.empty:
        raise ValueError("No non-NaN values for accuracy; nothing to plot.")

    method_order = list(agg["display_method"].unique())
    if 'No Missings ("truth")' in method_order:
        method_order = ['No Missings ("truth")'] + [m for m in method_order if m != 'No Missings ("truth")']

    preferred_mech_order = ["MCAR", "SLIGHTLY_MAR", "MAR", "MNAR"]
    mech_unique = list(agg[mech_col].unique())
    mech_order = [m for m in preferred_mech_order if m in mech_unique] + [
        m for m in mech_unique if m not in preferred_mech_order
    ]

    if mech_markers is None:
        mech_markers = {"MCAR": "o", "SLIGHTLY_MAR": "+", "MAR": "v", "MNAR": "x"}
    if mech_colors is None:
        mech_colors = {"MCAR": "#1f77b4", "SLIGHTLY_MAR": "#ff7f0e", "MAR": "#2ca02c", "MNAR": "#d62728"}

    def _marker_for(mech): return mech_markers.get(mech, "o")
    def _color_for(mech):  return mech_colors.get(mech, "black")

    point_size = 28
    edge_lw = 1.0
    legend_ms = 5
    legend_edge_lw = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(method_order))

    for y in y_pos:
        ax.axhline(y, linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)

    for mech in mech_order:
        mdf = agg[agg[mech_col] == mech]
        mk = _marker_for(mech)
        col = _color_for(mech)

        for i, method_name in enumerate(method_order):
            row = mdf[mdf["display_method"] == method_name]
            if row.empty:
                continue
            x = row[accuracy_col].iloc[0]
            if pd.isna(x):
                continue

            if mk in {"+", "x"}:
                ax.scatter(
                    float(x), y_pos[i],
                    marker=mk, color=col, linewidths=edge_lw,
                    s=point_size, zorder=3
                )
            else:
                ax.scatter(
                    float(x), y_pos[i],
                    marker=mk, facecolors="none", edgecolors=col,
                    linewidths=edge_lw, s=point_size, zorder=3
                )

    if include_baseline_line:
        base_rows = agg.loc[agg["display_method"] == 'No Missings ("truth")', accuracy_col].dropna()
        if base_rows.empty:
            base_df = df.loc[df[method_col] == baseline_name, accuracy_col].dropna()
            baseline_value = float(np.median(base_df.to_numpy())) if len(base_df) else None
        else:
            baseline_value = float(np.median(base_rows.to_numpy()))

        if baseline_value is not None and np.isfinite(baseline_value):
            ax.axvline(float(baseline_value), linestyle="--", linewidth=0.8, color="gray", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_order)
    ax.invert_yaxis()

    vals = agg[accuracy_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        vmin = float(vals.min())
        vmax = float(vals.max())
        pad = 0.03
        ax.set_xlim(max(0.0, vmin - pad), min(1.0, vmax + pad))

    ax.set_xlabel("accuracy")
    ax.set_title(title)
    ax.grid(False)

    legend_elems = []
    for mech in mech_order:
        mk = _marker_for(mech)
        col = _color_for(mech)
        if mk in {"+", "x"}:
            legend_elems.append(Line2D([0], [0], marker=mk, linestyle="", color=col,
                                       label=str(mech), markersize=legend_ms))
        else:
            legend_elems.append(Line2D([0], [0], marker=mk, linestyle="",
                                       markerfacecolor="none", markeredgecolor=col,
                                       markeredgewidth=legend_edge_lw,
                                       label=str(mech), markersize=legend_ms))

    ax.legend(
        handles=legend_elems,
        title="Mechanism",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    return fig, ax


def plot_pred_brier_mech_means(
    df: pd.DataFrame,
    *,
    mech_col: str = "mech",
    method_col: str = "method",
    scenario_col: str = "scenario",
    unit_col: str = "rep",
    baseline_name: str = "NoMissing",
    title: str = "Brier Score by Method and Mechanism",
    include_baseline_line: bool = True,
    figsize=(10, 8),
    mech_markers: dict[str, str] | None = None,
    mech_colors: dict[str, str] | None = None,
    brier_col: str = "brier",
):
    """
    Point plot of mean Brier score across replications, grouped by missingness mechanism.
    Y-axis: methods/scenarios, X-axis: Brier score.
    Smaller is better.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    required = {brier_col, mech_col, method_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    def _canon_mech(x):
        s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        if s in {"SLIGHTLY_MAR", "SLIGHTLYMAR", "SLIGHT_MAR"}:
            return "SLIGHTLY_MAR"
        return s

    df[mech_col] = df[mech_col].map(_canon_mech)


    def _canon_scenario(x):
        s = str(x).strip().lower()
        s = s.replace("→", "->")
        s = " ".join(s.split())

        if s in {"baseline"}:
            return "Baseline"
        if s in {"impute w/o y", "impute without y", "without y", "w/o y"}:
            return "Impute w/o y"
        if s in {"impute with y", "with y", "impute + y"}:
            return "Impute with y"
        if s in {
            "drop y-miss -> impute w/o y",
            "drop y-missingness -> impute w/o y",
            "drop y miss -> impute w/o y",
        }:
            return "Drop y-miss → impute w/o y"
        if s in {
            "drop y-miss -> impute + y",
            "drop y-missingness -> impute + y",
            "drop y miss -> impute + y",
            "drop y-miss -> impute with y",
        }:
            return "Drop y-miss → impute + y"

        return str(x)

    if scenario_col in df.columns:
        df[scenario_col] = df[scenario_col].map(_canon_scenario)
    else:
        df[scenario_col] = ""

    def _display_method(row):
        m = str(row[method_col]).strip()
        scen = str(row.get(scenario_col, "")).strip()

        if scen == "Baseline":
            if m == baseline_name:
                return 'No Missings ("truth")'
            if m == "Complete Case":
                return "Complete Case"
            return f"{m} baseline"

        if m == baseline_name:
            if scen == "Drop y-miss → impute w/o y":
                return 'No Missings ("truth", drop y-miss) w/o y'
            if scen == "Drop y-miss → impute + y":
                return 'No Missings ("truth", drop y-miss) with y'
            if scen == "Impute with y":
                return 'No Missings ("truth") with y'
            return 'No Missings ("truth") w/o y'

        if m == "Complete Case":
            if scen == "Drop y-miss → impute w/o y":
                return "Complete Case (drop y-miss) w/o y"
            if scen == "Drop y-miss → impute + y":
                return "Complete Case (drop y-miss) with y"
            if scen == "Impute with y":
                return "Complete Case with y"
            return "Complete Case w/o y"

        if m == "Drop Missingness":
            return "Drop Missingness"

        if scen == "Impute with y":
            return f"{m} with y"
        if scen == "Impute w/o y":
            return f"{m} w/o y"
        if scen == "Drop y-miss → impute w/o y":
            return f"{m} drop-y w/o y"
        if scen == "Drop y-miss → impute + y":
            return f"{m} drop-y with y"

        return m

    df["display_method"] = df.apply(_display_method, axis=1)


    agg = (
        df[[mech_col, "display_method", brier_col]]
        .dropna(subset=[mech_col, "display_method", brier_col])
        .groupby([mech_col, "display_method"], as_index=False)[brier_col]
        .mean()
    )
    if agg.empty:
        raise ValueError("No non-NaN values for Brier score; nothing to plot.")


    def _split_display_name(name: str):
        s = str(name)


        special_map = {
            'No Missings ("truth")': ('No Missings ("truth")', 0),
            'No Missings ("truth") w/o y': ('No Missings ("truth")', 1),
            'No Missings ("truth") with y': ('No Missings ("truth")', 2),
            'No Missings ("truth", drop y-miss) w/o y': ('No Missings ("truth")', 3),
            'No Missings ("truth", drop y-miss) with y': ('No Missings ("truth")', 4),

            'Complete Case': ('Complete Case', 0),
            'Complete Case w/o y': ('Complete Case', 1),
            'Complete Case with y': ('Complete Case', 2),
            'Complete Case (drop y-miss) w/o y': ('Complete Case', 3),
            'Complete Case (drop y-miss) with y': ('Complete Case', 4),

            'Drop Missingness': ('Drop Missingness', 0),
        }
        if s in special_map:
            return special_map[s]

        # generic methods
        if s.endswith(" w/o y"):
            return (s[:-6], 1)
        if s.endswith(" with y"):
            return (s[:-7], 2)
        if " drop-y w/o y" in s:
            return (s.replace(" drop-y w/o y", ""), 3)
        if " drop-y with y" in s:
            return (s.replace(" drop-y with y", ""), 4)
        if s.endswith(" baseline"):
            return (s[:-9], 0)

        return (s, 99)

    base_method_order = [
        'No Missings ("truth")',
        "Complete Case",
        "Drop Missingness",
    ]

    discovered_base_methods = []
    for name in agg["display_method"].unique():
        base, _ = _split_display_name(name)
        if base not in discovered_base_methods and base not in base_method_order:
            discovered_base_methods.append(base)

    full_base_order = base_method_order + sorted(discovered_base_methods)

    def _method_rank(name):
        base, scen_rank = _split_display_name(name)
        try:
            base_rank = full_base_order.index(base)
        except ValueError:
            base_rank = 999
        return (base_rank, scen_rank, str(name))

    method_order = sorted(agg["display_method"].unique(), key=_method_rank)


    preferred_mech_order = ["MCAR", "SLIGHTLY_MAR", "MAR", "MNAR"]
    mech_unique = list(agg[mech_col].unique())
    mech_order = [m for m in preferred_mech_order if m in mech_unique] + [
        m for m in mech_unique if m not in preferred_mech_order
    ]


    if mech_markers is None:
        mech_markers = {"MCAR": "o", "SLIGHTLY_MAR": "+", "MAR": "v", "MNAR": "x"}
    if mech_colors is None:
        mech_colors = {
            "MCAR": "#1f77b4",
            "SLIGHTLY_MAR": "#ff7f0e",
            "MAR": "#2ca02c",
            "MNAR": "#d62728",
        }

    def _marker_for(mech):
        return mech_markers.get(mech, "o")

    def _color_for(mech):
        return mech_colors.get(mech, "black")


    point_size = 28
    edge_lw = 1.0
    legend_ms = 5
    legend_edge_lw = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(method_order))

    for y in y_pos:
        ax.axhline(y, linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)

    for mech in mech_order:
        mdf = agg[agg[mech_col] == mech]
        mk = _marker_for(mech)
        col = _color_for(mech)

        for i, method_name in enumerate(method_order):
            row = mdf[mdf["display_method"] == method_name]
            if row.empty:
                continue
            x = row[brier_col].iloc[0]
            if pd.isna(x):
                continue

            if mk in {"+", "x"}:
                ax.scatter(
                    float(x), y_pos[i],
                    marker=mk,
                    color=col,
                    linewidths=edge_lw,
                    s=point_size,
                    zorder=3,
                )
            else:
                ax.scatter(
                    float(x), y_pos[i],
                    marker=mk,
                    facecolors="none",
                    edgecolors=col,
                    linewidths=edge_lw,
                    s=point_size,
                    zorder=3,
                )


    if include_baseline_line:
        base_rows = agg.loc[
            agg["display_method"].isin([
                'No Missings ("truth")',
                'No Missings ("truth") w/o y',
                'No Missings ("truth") with y'
            ]),
            brier_col
        ].dropna()

        if base_rows.empty:
            base_df = df.loc[df[method_col] == baseline_name, brier_col].dropna()
            baseline_value = float(np.median(base_df.to_numpy())) if len(base_df) else None
        else:
            baseline_value = float(np.median(base_rows.to_numpy()))

        if baseline_value is not None and np.isfinite(baseline_value):
            ax.axvline(float(baseline_value), linestyle="--", linewidth=0.8, color="gray", zorder=1)


    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_order)
    ax.invert_yaxis()

    vals = agg[brier_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        vmin = float(vals.min())
        vmax = float(vals.max())
        pad = 0.03 * max(1.0, vmax - vmin)
        ax.set_xlim(max(0.0, vmin - pad), vmax + pad)

    ax.set_xlabel("Brier score")
    ax.set_title(title)
    ax.grid(False)


    legend_elems = []
    for mech in mech_order:
        mk = _marker_for(mech)
        col = _color_for(mech)
        if mk in {"+", "x"}:
            legend_elems.append(
                Line2D([0], [0], marker=mk, linestyle="", color=col,
                       label=str(mech), markersize=legend_ms)
            )
        else:
            legend_elems.append(
                Line2D([0], [0], marker=mk, linestyle="",
                       markerfacecolor="none", markeredgecolor=col,
                       markeredgewidth=legend_edge_lw,
                       label=str(mech), markersize=legend_ms)
            )

    ax.legend(
        handles=legend_elems,
        title="Mechanism",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    return fig, ax
