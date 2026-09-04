from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Helpers
# ============================================================

def get_method_data(
    df,
    method,
):
    """
    Select one method and sort by autocorrelation parameter a.
    """

    method_df = (
        df[
            df["method"] == method
        ]
        .copy()
        .sort_values("a")
    )

    if len(method_df) == 0:
        raise ValueError(
            f"No results found for method: {method}"
        )

    return method_df


def plot_rate_with_sem(
    ax,
    method_df,
    mean_column,
    sem_column,
    label,
    marker,
    linestyle="-",
):
    """
    Plot percentage-valued metric using mean ± SEM.
    """

    x = (
        method_df["a"]
        .to_numpy(dtype=float)
    )

    mean = (
        method_df[mean_column]
        .to_numpy(dtype=float)
        * 100.0
    )

    sem = (
        method_df[sem_column]
        .to_numpy(dtype=float)
        * 100.0
    )

    ax.errorbar(
        x,
        mean,
        yerr=sem,
        marker=marker,
        linestyle=linestyle,
        capsize=4,
        label=label,
    )


def plot_rate_without_errorbar(
    ax,
    method_df,
    value_column,
    label,
    marker,
    linestyle="-",
):
    """
    Plot a pooled percentage metric.

    Pooled ratios currently do not have realization-level SEM,
    so they are drawn without error bars.
    """

    x = (
        method_df["a"]
        .to_numpy(dtype=float)
    )

    values = (
        method_df[value_column]
        .to_numpy(dtype=float)
        * 100.0
    )

    ax.plot(
        x,
        values,
        marker=marker,
        linestyle=linestyle,
        label=label,
    )


def format_percentage_axis(
    ax,
    title,
    a_values,
):
    """
    Shared formatting for percentage panels.
    """

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Autocorrelation parameter a"
    )

    ax.set_ylabel(
        "Rate (%)"
    )

    ax.set_ylim(
        -2,
        102,
    )

    ax.set_xticks(
        a_values
    )

    ax.set_xticklabels(
        [
            f"{value:g}"
            for value in a_values
        ]
    )

    ax.grid(
        alpha=0.25
    )


# ============================================================
# 2. Adjacency TPR
# ============================================================

def plot_adjacency_tpr(
    ax,
    summary_df,
    methods,
    a_values,
):
    """
    Plot TPR for:
        - lagged cross-links
        - contemporaneous adjacencies
        - autodependencies

    Note:
        at a = 0 there may be no true autodependency edge,
        so auto TPR can correctly be NaN.
    """

    markers = {
        "lagged": "o",
        "contemp": "s",
        "auto": "^",
    }

    linestyles = {
        "lagged": "-",
        "contemp": "--",
        "auto": ":",
    }

    for method in methods:

        method_df = get_method_data(
            summary_df,
            method,
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "lagged_tpr_mean",
            sem_column=
                "lagged_tpr_sem",
            label=
                f"{method} / Lagged",
            marker=
                markers["lagged"],
            linestyle=
                linestyles["lagged"],
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "contemp_tpr_mean",
            sem_column=
                "contemp_tpr_sem",
            label=
                f"{method} / Contemp.",
            marker=
                markers["contemp"],
            linestyle=
                linestyles["contemp"],
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "auto_tpr_mean",
            sem_column=
                "auto_tpr_sem",
            label=
                f"{method} / Auto",
            marker=
                markers["auto"],
            linestyle=
                linestyles["auto"],
        )

    format_percentage_axis(
        ax=ax,
        title="Adjacency TPR",
        a_values=a_values,
    )

    ax.legend(
        fontsize=8
    )


# ============================================================
# 3. Adjacency FPR
# ============================================================

def plot_adjacency_fpr(
    ax,
    summary_df,
    methods,
    a_values,
):
    """
    Plot FPR for the three adjacency categories.
    """

    markers = {
        "lagged": "o",
        "contemp": "s",
        "auto": "^",
    }

    linestyles = {
        "lagged": "-",
        "contemp": "--",
        "auto": ":",
    }

    for method in methods:

        method_df = get_method_data(
            summary_df,
            method,
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "lagged_fpr_mean",
            sem_column=
                "lagged_fpr_sem",
            label=
                f"{method} / Lagged",
            marker=
                markers["lagged"],
            linestyle=
                linestyles["lagged"],
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "contemp_fpr_mean",
            sem_column=
                "contemp_fpr_sem",
            label=
                f"{method} / Contemp.",
            marker=
                markers["contemp"],
            linestyle=
                linestyles["contemp"],
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "auto_fpr_mean",
            sem_column=
                "auto_fpr_sem",
            label=
                f"{method} / Auto",
            marker=
                markers["auto"],
            linestyle=
                linestyles["auto"],
        )

    ax.set_title(
        "Adjacency FPR"
    )

    ax.set_xlabel(
        "Autocorrelation parameter a"
    )

    ax.set_ylabel(
        "False positive rate (%)"
    )

    ax.set_xticks(
        a_values
    )

    ax.set_xticklabels(
        [
            f"{value:g}"
            for value in a_values
        ]
    )

    ax.set_ylim(
        bottom=-0.2
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8
    )


# ============================================================
# 4. Orientation recall
# ============================================================

def plot_orientation_recall(
    ax,
    summary_df,
    methods,
    a_values,
):
    """
    Plot current operational DAG-direction recall.

    This is still NOT the final CPDAG-aware
    Runge (2020) orientation metric.
    """

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    for index, method in enumerate(
        methods
    ):

        method_df = get_method_data(
            summary_df,
            method,
        )

        plot_rate_with_sem(
            ax=ax,
            method_df=method_df,
            mean_column=
                "orient_recall_op_mean",
            sem_column=
                "orient_recall_op_sem",
            label=method,
            marker=
                markers[
                    index
                    % len(markers)
                ],
        )

    format_percentage_axis(
        ax=ax,
        title=(
            "Contemporaneous orientation recall\n"
            "(operational)"
        ),
        a_values=a_values,
    )

    ax.legend()


# ============================================================
# 5. Orientation precision
# ============================================================

def plot_orientation_precision(
    ax,
    summary_df,
    methods,
    a_values,
):
    """
    Prefer pooled orientation precision:

        sum(correct orientations)
        -------------------------
        sum(estimated contemporaneous adjacencies)

    If the pooled column is unavailable, fall back to
    mean-of-ratios with SEM.
    """

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    pooled_column = (
        "orient_precision_op_pooled"
    )

    use_pooled = (
        pooled_column
        in summary_df.columns
    )

    for index, method in enumerate(
        methods
    ):

        method_df = get_method_data(
            summary_df,
            method,
        )

        if use_pooled:

            plot_rate_without_errorbar(
                ax=ax,
                method_df=method_df,
                value_column=
                    pooled_column,
                label=
                    f"{method} / pooled",
                marker=
                    markers[
                        index
                        % len(markers)
                    ],
            )

        else:

            plot_rate_with_sem(
                ax=ax,
                method_df=method_df,
                mean_column=
                    "orient_precision_op_mean",
                sem_column=
                    "orient_precision_op_sem",
                label=
                    f"{method} / mean ratio",
                marker=
                    markers[
                        index
                        % len(markers)
                    ],
            )

    title_suffix = (
        "pooled"
        if use_pooled
        else "mean of ratios"
    )

    format_percentage_axis(
        ax=ax,
        title=(
            "Contemporaneous orientation precision\n"
            f"(operational, {title_suffix})"
        ),
        a_values=a_values,
    )

    ax.legend()


# ============================================================
# 6. Runtime
# ============================================================

def plot_runtime(
    ax,
    summary_df,
    methods,
    a_values,
):
    """
    Plot mean runtime with empirical 5%-95% range.
    """

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    for index, method in enumerate(
        methods
    ):

        method_df = get_method_data(
            summary_df,
            method,
        )

        x = (
            method_df["a"]
            .to_numpy(dtype=float)
        )

        mean = (
            method_df[
                "runtime_mean"
            ]
            .to_numpy(dtype=float)
        )

        p05 = (
            method_df[
                "runtime_p05"
            ]
            .to_numpy(dtype=float)
        )

        p95 = (
            method_df[
                "runtime_p95"
            ]
            .to_numpy(dtype=float)
        )

        lower = np.maximum(
            mean - p05,
            0.0,
        )

        upper = np.maximum(
            p95 - mean,
            0.0,
        )

        yerr = np.vstack(
            [
                lower,
                upper,
            ]
        )

        ax.errorbar(
            x,
            mean,
            yerr=yerr,
            marker=
                markers[
                    index
                    % len(markers)
                ],
            capsize=4,
            label=method,
        )

    ax.set_title(
        "Runtime"
    )

    ax.set_xlabel(
        "Autocorrelation parameter a"
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    ax.set_xticks(
        a_values
    )

    ax.set_xticklabels(
        [
            f"{value:g}"
            for value in a_values
        ]
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()


# ============================================================
# 7. Conflicts
# ============================================================

def plot_conflicts(
    ax,
    summary_df,
    methods,
    a_values,
):
    """
    Prefer pooled conflict rate:

        sum(conflicting contemporaneous adjacencies)
        --------------------------------------------
        sum(estimated contemporaneous adjacencies)

    If pooled values are unavailable, fall back to
    the previous mean-of-ratios statistic.
    """

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    pooled_column = (
        "conflict_rate_pooled"
    )

    use_pooled = (
        pooled_column
        in summary_df.columns
    )

    for index, method in enumerate(
        methods
    ):

        method_df = get_method_data(
            summary_df,
            method,
        )

        if use_pooled:

            plot_rate_without_errorbar(
                ax=ax,
                method_df=method_df,
                value_column=
                    pooled_column,
                label=
                    f"{method} / pooled",
                marker=
                    markers[
                        index
                        % len(markers)
                    ],
            )

        else:

            plot_rate_with_sem(
                ax=ax,
                method_df=method_df,
                mean_column=
                    "conflict_rate_mean",
                sem_column=
                    "conflict_rate_sem",
                label=
                    f"{method} / mean ratio",
                marker=
                    markers[
                        index
                        % len(markers)
                    ],
            )

    title_suffix = (
        "pooled"
        if use_pooled
        else "mean of ratios"
    )

    format_percentage_axis(
        ax=ax,
        title=(
            "Conflicts\n"
            f"({title_suffix})"
        ),
        a_values=a_values,
    )

    ax.legend()


# ============================================================
# 8. Main
# ============================================================

def main():

    # ========================================================
    # Paths
    # ========================================================

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    results_dir = (
        script_dir
        / "results"
    )

    summary_path = (
        results_dir
        / "07_autocorrelation_summary.csv"
    )

    output_png = (
        results_dir
        / "08_autocorrelation_figure.png"
    )

    output_pdf = (
        results_dir
        / "08_autocorrelation_figure.pdf"
    )

    # ========================================================
    # Check input
    # ========================================================

    if not summary_path.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"{summary_path}\n\n"
            "Run 07_autocorrelation_benchmark.py first."
        )

    # ========================================================
    # Load summary
    # ========================================================

    summary_df = pd.read_csv(
        summary_path
    )

    print(
        "Loaded summary:"
    )

    print(
        summary_path
    )

    print(
        "\nRows =",
        len(summary_df),
    )

    # ========================================================
    # Methods
    # ========================================================

    preferred_order = [
        "PCMCI+",
        "PC",
    ]

    available_methods = (
        summary_df[
            "method"
        ]
        .drop_duplicates()
        .tolist()
    )

    methods = [
        method
        for method
        in preferred_order
        if method
        in available_methods
    ]

    methods += [
        method
        for method
        in available_methods
        if method
        not in methods
    ]

    # ========================================================
    # a values
    # ========================================================

    a_values = sorted(
        summary_df[
            "a"
        ]
        .dropna()
        .unique()
        .astype(float)
        .tolist()
    )

    print(
        "Methods =",
        methods,
    )

    print(
        "a values =",
        a_values,
    )

    # ========================================================
    # Realization counts
    # ========================================================

    if (
        "n_realizations"
        in summary_df.columns
    ):

        print(
            "\nRealizations per condition:"
        )

        print(
            summary_df[
                [
                    "method",
                    "a",
                    "n_realizations",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # Required columns
    # ========================================================

    required_columns = [
        "method",
        "a",

        "lagged_tpr_mean",
        "lagged_tpr_sem",

        "lagged_fpr_mean",
        "lagged_fpr_sem",

        "auto_tpr_mean",
        "auto_tpr_sem",

        "auto_fpr_mean",
        "auto_fpr_sem",

        "contemp_tpr_mean",
        "contemp_tpr_sem",

        "contemp_fpr_mean",
        "contemp_fpr_sem",

        "orient_recall_op_mean",
        "orient_recall_op_sem",

        "runtime_mean",
        "runtime_p05",
        "runtime_p95",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in summary_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # Tell us which aggregation is being used
    # ========================================================

    if (
        "orient_precision_op_pooled"
        in summary_df.columns
    ):

        print(
            "\nOrientation precision:"
            " using pooled aggregation."
        )

    else:

        print(
            "\nOrientation precision:"
            " pooled column unavailable; "
            "using mean-of-ratios."
        )

    if (
        "conflict_rate_pooled"
        in summary_df.columns
    ):

        print(
            "Conflict rate:"
            " using pooled aggregation."
        )

    else:

        print(
            "Conflict rate:"
            " pooled column unavailable; "
            "using mean-of-ratios."
        )

    # ========================================================
    # Create Figure 2A-style 2x3 figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            17,
            9,
        ),
        constrained_layout=True,
    )

    # ========================================================
    # Top row
    # ========================================================

    plot_adjacency_tpr(
        ax=axes[0, 0],
        summary_df=summary_df,
        methods=methods,
        a_values=a_values,
    )

    plot_orientation_recall(
        ax=axes[0, 1],
        summary_df=summary_df,
        methods=methods,
        a_values=a_values,
    )

    plot_runtime(
        ax=axes[0, 2],
        summary_df=summary_df,
        methods=methods,
        a_values=a_values,
    )

    # ========================================================
    # Bottom row
    # ========================================================

    plot_adjacency_fpr(
        ax=axes[1, 0],
        summary_df=summary_df,
        methods=methods,
        a_values=a_values,
    )

    plot_orientation_precision(
        ax=axes[1, 1],
        summary_df=summary_df,
        methods=methods,
        a_values=a_values,
    )

    plot_conflicts(
        ax=axes[1, 2],
        summary_df=summary_df,
        methods=methods,
        a_values=a_values,
    )

    # ========================================================
    # Overall title
    # ========================================================

    fig.suptitle(
        "Runge (2020) Figure 2A-style reproduction\n"
        "N=5, T=500, tau_max=5, alpha=0.01",
        fontsize=14,
    )

    # ========================================================
    # Save
    # ========================================================

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    print(
        "\nFigure saved:"
    )

    print(
        output_png
    )

    print(
        output_pdf
    )

    # ========================================================
    # Display
    # ========================================================

    plt.show()

    print(
        "\n08 autocorrelation plotting completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()