from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic plotting helpers
# ============================================================

def get_method_data(
    df,
    method,
):
    """
    Select one method and sort rows by sample size T.
    """

    method_df = (
        df[
            df["method"] == method
        ]
        .copy()
        .sort_values("T")
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
    Plot a rate metric as percentage with mean ± SEM.
    """

    x = method_df["T"].to_numpy()

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


def format_percentage_axis(
    ax,
    title,
):
    """
    Common formatting for percentage panels.
    """

    ax.set_title(title)

    ax.set_xlabel(
        "Sample size T"
    )

    ax.set_ylabel(
        "Rate (%)"
    )

    ax.set_ylim(
        -2,
        102,
    )

    ax.set_xticks(
        [200, 500, 1000]
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
):
    """
    Plot lagged / contemporaneous / auto TPR.
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
        ax,
        "Adjacency TPR",
    )

    ax.legend(
        fontsize=8,
    )


# ============================================================
# 3. Adjacency FPR
# ============================================================

def plot_adjacency_fpr(
    ax,
    summary_df,
    methods,
):
    """
    Plot lagged / contemporaneous / auto FPR.

    For the first reproduction version we use a linear
    percentage axis. This keeps zero-FPR points visible.

    The Runge (2020) figure uses a compressed/log-like
    presentation for small FPR values. We can reproduce
    that more strictly later.
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
        "Sample size T"
    )

    ax.set_ylabel(
        "False positive rate (%)"
    )

    ax.set_xticks(
        [200, 500, 1000]
    )

    # FPR is usually much smaller than TPR.
    # Let matplotlib determine the upper limit automatically.
    ax.set_ylim(
        bottom=-0.2
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8,
    )


# ============================================================
# 4. Contemporaneous orientation recall
# ============================================================

def plot_orientation_recall(
    ax,
    summary_df,
    methods,
):
    """
    Current operational DAG-direction orientation recall.

    IMPORTANT:
    This is not yet the final CPDAG-aware reproduction
    of Runge (2020).
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
        ax,
        "Contemporaneous orientation recall",
    )

    ax.legend()


# ============================================================
# 5. Contemporaneous orientation precision
# ============================================================

def plot_orientation_precision(
    ax,
    summary_df,
    methods,
):
    """
    Current operational DAG-direction orientation precision.

    Some realizations may have NaN precision when an
    algorithm estimates no contemporaneous adjacency.
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
                "orient_precision_op_mean",
            sem_column=
                "orient_precision_op_sem",
            label=method,
            marker=
                markers[
                    index
                    % len(markers)
                ],
        )

    format_percentage_axis(
        ax,
        "Contemporaneous orientation precision",
    )

    ax.legend()


# ============================================================
# 6. Runtime
# ============================================================

def plot_runtime(
    ax,
    summary_df,
    methods,
):
    """
    Plot mean runtime.

    Error bars use the empirical 5th-95th percentile range,
    corresponding to a 90% empirical range.
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
            method_df["T"]
            .to_numpy()
        )

        mean = (
            method_df[
                "runtime_mean"
            ]
            .to_numpy(dtype=float)
        )

        lower = (
            mean
            -
            method_df[
                "runtime_p05"
            ].to_numpy(dtype=float)
        )

        upper = (
            method_df[
                "runtime_p95"
            ].to_numpy(dtype=float)
            -
            mean
        )

        # Numerical protection against very small
        # floating point negatives.
        lower = np.maximum(
            lower,
            0.0,
        )

        upper = np.maximum(
            upper,
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
        "Sample size T"
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    ax.set_xticks(
        [200, 500, 1000]
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()


# ============================================================
# 7. Conflict rate
# ============================================================

def plot_conflicts(
    ax,
    summary_df,
    methods,
):
    """
    Plot fraction of conflicting contemporaneous links.
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
                "conflict_rate_mean",
            sem_column=
                "conflict_rate_sem",
            label=method,
            marker=
                markers[
                    index
                    % len(markers)
                ],
        )

    format_percentage_axis(
        ax,
        "Conflicts",
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
        / "05_sample_size_summary.csv"
    )

    output_png = (
        results_dir
        / "06_sample_size_figure.png"
    )

    output_pdf = (
        results_dir
        / "06_sample_size_figure.pdf"
    )

    # ========================================================
    # Make sure 05 has already been run
    # ========================================================

    if not summary_path.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"{summary_path}\n\n"
            "Run 05_sample_size_benchmark.py first."
        )

    # ========================================================
    # Read benchmark summary
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

    # Add any future methods that are not yet
    # included in preferred_order.
    methods += [
        method
        for method
        in available_methods
        if method
        not in methods
    ]

    print(
        "Methods =",
        methods,
    )

    print(
        "T values =",
        sorted(
            summary_df[
                "T"
            ].unique()
        ),
    )

    # ========================================================
    # Required columns check
    # ========================================================

    required_columns = [
        "method",
        "T",

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

        "orient_precision_op_mean",
        "orient_precision_op_sem",

        "runtime_mean",
        "runtime_p05",
        "runtime_p95",

        "conflict_rate_mean",
        "conflict_rate_sem",
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
            "Missing columns in summary CSV:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # Create Figure 2C-style 2 x 3 panel figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            9,
        ),
        constrained_layout=True,
    )

    # Top row
    plot_adjacency_tpr(
        ax=axes[0, 0],
        summary_df=summary_df,
        methods=methods,
    )

    plot_orientation_recall(
        ax=axes[0, 1],
        summary_df=summary_df,
        methods=methods,
    )

    plot_runtime(
        ax=axes[0, 2],
        summary_df=summary_df,
        methods=methods,
    )

    # Bottom row
    plot_adjacency_fpr(
        ax=axes[1, 0],
        summary_df=summary_df,
        methods=methods,
    )

    plot_orientation_precision(
        ax=axes[1, 1],
        summary_df=summary_df,
        methods=methods,
    )

    plot_conflicts(
        ax=axes[1, 2],
        summary_df=summary_df,
        methods=methods,
    )

    # ========================================================
    # Overall title
    # ========================================================

    fig.suptitle(
        "Runge (2020) Figure 2C-style reproduction\n"
        "N=5, a=0.95, tau_max=5, alpha=0.01",
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
        "\n06 sample-size plotting completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()