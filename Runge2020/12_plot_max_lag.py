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
    Select one method and sort by search tau_max.
    """

    method_df = (
        df[
            df["method"] == method
        ]
        .copy()
        .sort_values(
            "search_tau_max"
        )
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

    x = (
        method_df[
            "search_tau_max"
        ]
        .to_numpy(dtype=float)
    )

    mean = (
        method_df[
            mean_column
        ]
        .to_numpy(dtype=float)
        * 100.0
    )

    sem = (
        method_df[
            sem_column
        ]
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


def plot_pooled_rate(
    ax,
    method_df,
    value_column,
    label,
    marker,
    linestyle="-",
):

    x = (
        method_df[
            "search_tau_max"
        ]
        .to_numpy(dtype=float)
    )

    values = (
        method_df[
            value_column
        ]
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
    tau_values,
):

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Maximum searched lag"
    )

    ax.set_ylabel(
        "Rate (%)"
    )

    ax.set_ylim(
        -2,
        102,
    )

    ax.set_xticks(
        tau_values
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
    tau_values,
):

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

        method_df = (
            get_method_data(
                summary_df,
                method,
            )
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
        tau_values=tau_values,
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
    tau_values,
):

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
        "Maximum searched lag"
    )

    ax.set_ylabel(
        "False positive rate (%)"
    )

    ax.set_xticks(
        tau_values
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
    tau_values,
):

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    for index, method in enumerate(
        methods
    ):

        method_df = (
            get_method_data(
                summary_df,
                method,
            )
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
        tau_values=tau_values,
    )

    ax.legend()


# ============================================================
# 5. Orientation precision
# ============================================================

def plot_orientation_precision(
    ax,
    summary_df,
    methods,
    tau_values,
):

    pooled_column = (
        "orient_precision_op_pooled"
    )

    use_pooled = (
        pooled_column
        in summary_df.columns
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    for index, method in enumerate(
        methods
    ):

        method_df = (
            get_method_data(
                summary_df,
                method,
            )
        )

        if use_pooled:

            plot_pooled_rate(
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

    suffix = (
        "pooled"
        if use_pooled
        else "mean of ratios"
    )

    format_percentage_axis(
        ax=ax,
        title=(
            "Contemporaneous orientation precision\n"
            f"(operational, {suffix})"
        ),
        tau_values=tau_values,
    )

    ax.legend()


# ============================================================
# 6. Runtime
# ============================================================

def plot_runtime(
    ax,
    summary_df,
    methods,
    tau_values,
):

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    for index, method in enumerate(
        methods
    ):

        method_df = (
            get_method_data(
                summary_df,
                method,
            )
        )

        x = (
            method_df[
                "search_tau_max"
            ]
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
        "Maximum searched lag"
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    ax.set_xticks(
        tau_values
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
    tau_values,
):

    pooled_column = (
        "conflict_rate_pooled"
    )

    use_pooled = (
        pooled_column
        in summary_df.columns
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    for index, method in enumerate(
        methods
    ):

        method_df = (
            get_method_data(
                summary_df,
                method,
            )
        )

        if use_pooled:

            plot_pooled_rate(
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

    suffix = (
        "pooled"
        if use_pooled
        else "mean of ratios"
    )

    format_percentage_axis(
        ax=ax,
        title=(
            "Conflicts\n"
            f"({suffix})"
        ),
        tau_values=tau_values,
    )

    ax.legend()


# ============================================================
# 8. Main
# ============================================================

def main():

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
        / "11_max_lag_summary.csv"
    )

    output_png = (
        results_dir
        / "12_max_lag_figure.png"
    )

    output_pdf = (
        results_dir
        / "12_max_lag_figure.pdf"
    )

    # ========================================================
    # Load
    # ========================================================

    if not summary_path.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"{summary_path}\n\n"
            "Run 11_max_lag_benchmark.py first."
        )

    summary_df = pd.read_csv(
        summary_path
    )

    print(
        "Loaded:"
    )

    print(
        summary_path
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
    # Search tau values
    # ========================================================

    tau_values = sorted(
        summary_df[
            "search_tau_max"
        ]
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )

    print(
        "Methods =",
        methods,
    )

    print(
        "Search tau values =",
        tau_values,
    )

    print(
        "True tau max remains fixed at 5."
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
                    "search_tau_max",
                    "n_realizations",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # Figure
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

    plot_adjacency_tpr(
        ax=axes[0, 0],
        summary_df=summary_df,
        methods=methods,
        tau_values=tau_values,
    )

    plot_orientation_recall(
        ax=axes[0, 1],
        summary_df=summary_df,
        methods=methods,
        tau_values=tau_values,
    )

    plot_runtime(
        ax=axes[0, 2],
        summary_df=summary_df,
        methods=methods,
        tau_values=tau_values,
    )

    plot_adjacency_fpr(
        ax=axes[1, 0],
        summary_df=summary_df,
        methods=methods,
        tau_values=tau_values,
    )

    plot_orientation_precision(
        ax=axes[1, 1],
        summary_df=summary_df,
        methods=methods,
        tau_values=tau_values,
    )

    plot_conflicts(
        ax=axes[1, 2],
        summary_df=summary_df,
        methods=methods,
        tau_values=tau_values,
    )

    # ========================================================
    # Overall title
    # ========================================================

    fig.suptitle(
        "Runge (2020) Figure 2D-style reproduction\n"
        "N=5, T=500, a=0.95, true tau_max=5, alpha=0.01",
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
        "\nSaved:"
    )

    print(
        output_png
    )

    print(
        output_pdf
    )

    plt.show()

    print(
        "\n12 max-lag plotting completed."
    )


if __name__ == "__main__":
    main()