import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from causal_methods import run_causal_method


# ============================================================
# 1. Import generator
# ============================================================

scm_module = importlib.import_module(
    "01_generate_linear_scm"
)

generate_linear_scm = (
    scm_module.generate_linear_scm
)


# ============================================================
# 2. Import evaluator
# ============================================================

metrics_module = importlib.import_module(
    "03_metrics"
)

evaluate_graph = (
    metrics_module.evaluate_graph
)


# ============================================================
# 3. Convert metric report to one flat result row
# ============================================================

def build_result_row(
    method_result,
    metric_report,
    model_info,
    seed,
    sample_size,
):
    """
    Convert one algorithm run into one flat row
    suitable for CSV and pandas aggregation.
    """

    lagged = (
        metric_report[
            "lagged_cross"
        ]
    )

    auto = (
        metric_report[
            "auto"
        ]
    )

    contemp = (
        metric_report[
            "contemporaneous"
        ]
    )

    orient = (
        metric_report[
            "orientation"
        ]
    )

    return {
        # ----------------------------------------------------
        # Experimental identifiers
        # ----------------------------------------------------

        "method":
            method_result["method"],

        "seed":
            seed,

        "T":
            sample_size,

        # ----------------------------------------------------
        # SCM metadata
        # ----------------------------------------------------

        "spectral_radius":
            model_info[
                "spectral_radius"
            ],

        "generation_attempts":
            model_info[
                "attempts"
            ],

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        "runtime":
            method_result[
                "runtime"
            ],

        # ----------------------------------------------------
        # Lagged cross-link adjacency
        # ----------------------------------------------------

        "lagged_tp":
            lagged["TP"],

        "lagged_fp":
            lagged["FP"],

        "lagged_fn":
            lagged["FN"],

        "lagged_tn":
            lagged["TN"],

        "lagged_tpr":
            lagged["TPR"],

        "lagged_fpr":
            lagged["FPR"],

        # ----------------------------------------------------
        # Autodependency adjacency
        # ----------------------------------------------------

        "auto_tp":
            auto["TP"],

        "auto_fp":
            auto["FP"],

        "auto_fn":
            auto["FN"],

        "auto_tn":
            auto["TN"],

        "auto_tpr":
            auto["TPR"],

        "auto_fpr":
            auto["FPR"],

        # ----------------------------------------------------
        # Contemporaneous adjacency
        # ----------------------------------------------------

        "contemp_tp":
            contemp["TP"],

        "contemp_fp":
            contemp["FP"],

        "contemp_fn":
            contemp["FN"],

        "contemp_tn":
            contemp["TN"],

        "contemp_tpr":
            contemp["TPR"],

        "contemp_fpr":
            contemp["FPR"],

        # ----------------------------------------------------
        # Contemporaneous orientation
        #
        # IMPORTANT:
        # These are our CURRENT operational DAG-direction
        # metrics, not yet the final CPDAG-aware reproduction.
        # ----------------------------------------------------

        "orient_correct":
            orient["correct"],

        "orient_reversed":
            orient["reversed"],

        "orient_unresolved":
            orient["unresolved"],

        "orient_conflicts":
            orient["conflicts"],

        "orient_recall_op":
            orient["recall"],

        "orient_precision_op":
            orient["precision"],

        "conflict_rate":
            orient["conflict_rate"],
    }


# ============================================================
# 4. Summarize across seeds
# ============================================================

def build_summary(
    raw_df,
):
    """
    Aggregate results across realizations.

    For main metrics:
        mean
        std
        SEM

    Runtime additionally includes:
        5th percentile
        95th percentile

    The paper reports standard errors for most metrics
    and a 90% range for runtime.
    """

    metric_columns = [
        "lagged_tpr",
        "lagged_fpr",

        "auto_tpr",
        "auto_fpr",

        "contemp_tpr",
        "contemp_fpr",

        "orient_recall_op",
        "orient_precision_op",

        "conflict_rate",

        "runtime",
    ]

    summary_rows = []

    grouped = raw_df.groupby(
        [
            "method",
            "T",
        ],
        sort=True,
    )

    for (
        method,
        sample_size,
    ), group in grouped:

        row = {
            "method":
                method,

            "T":
                sample_size,

            "n_realizations":
                len(group),
        }

        for metric in metric_columns:

            values = (
                group[metric]
                .dropna()
                .astype(float)
            )

            if len(values) == 0:

                mean = np.nan
                std = np.nan
                sem = np.nan

            else:

                mean = (
                    values.mean()
                )

                if len(values) > 1:

                    std = (
                        values.std(
                            ddof=1
                        )
                    )

                    sem = (
                        std
                        / np.sqrt(
                            len(values)
                        )
                    )

                else:

                    std = np.nan
                    sem = np.nan

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_std"
            ] = std

            row[
                f"{metric}_sem"
            ] = sem

        # ----------------------------------------------------
        # Runtime 90% empirical range
        # ----------------------------------------------------

        runtime_values = (
            group["runtime"]
            .dropna()
            .astype(float)
        )

        if len(runtime_values) > 0:

            row[
                "runtime_p05"
            ] = np.percentile(
                runtime_values,
                5,
            )

            row[
                "runtime_p95"
            ] = np.percentile(
                runtime_values,
                95,
            )

        else:

            row[
                "runtime_p05"
            ] = np.nan

            row[
                "runtime_p95"
            ] = np.nan

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# 5. Main benchmark
# ============================================================

def main():

    # ========================================================
    # Runge (2020) Figure 2C-style setup
    # ========================================================

    N = 5

    A_MAX = 0.95

    TAU_MAX = 5

    PC_ALPHA = 0.01

    BURN_IN = 500

    # --------------------------------------------------------
    # Sample sizes used in Figure 2C
    # --------------------------------------------------------

    T_VALUES = [
        200,
        500,
        1000,
    ]

    # --------------------------------------------------------
    # FIRST dry run:
    #
    # N_SEEDS = 5
    #
    # After validation:
    #
    # N_SEEDS = 20
    #
    # Paper final experiments use much more
    # (500 realizations).
    # --------------------------------------------------------

    N_SEEDS = 20

    METHODS = [
        "PCMCI+",
        "PC",
    ]

    VAR_NAMES = [
        f"X{i}"
        for i in range(N)
    ]

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "05_sample_size_raw.csv"
    )

    summary_path = (
        output_dir
        / "05_sample_size_summary.csv"
    )

    print(
        "Runge (2020) - "
        "Sample-size benchmark"
    )

    print("=" * 72)

    print(
        "T values =",
        T_VALUES,
    )

    print(
        "number of seeds =",
        N_SEEDS,
    )

    print(
        "methods =",
        METHODS,
    )

    print()

    # ========================================================
    # Raw experiment rows
    # ========================================================

    rows = []

    total_runs = (
        len(T_VALUES)
        * N_SEEDS
        * len(METHODS)
    )

    run_index = 0

    # ========================================================
    # Benchmark loop
    # ========================================================

    for sample_size in T_VALUES:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"T = {sample_size}"
        )

        print(
            "=" * 72
        )

        for seed in range(
            N_SEEDS
        ):

            # =================================================
            # IMPORTANT:
            # Generate dataset ONCE for this (T, seed).
            # Both algorithms receive exactly this same data.
            # =================================================

            (
                data,
                true_edges,
                model_info,
            ) = generate_linear_scm(
                n_variables=N,
                n_samples=sample_size,
                a_max=A_MAX,
                tau_max=TAU_MAX,
                seed=seed,
                burn_in=BURN_IN,
            )

            print(
                f"\nSeed {seed:3d} | "
                f"rho="
                f"{model_info['spectral_radius']:.4f}"
            )

            # =================================================
            # Run both methods on SAME data
            # =================================================

            for method in METHODS:

                run_index += 1

                print(
                    f"  "
                    f"[{run_index:3d}/{total_runs}] "
                    f"{method:<7}",
                    end="",
                    flush=True,
                )

                method_result = (
                    run_causal_method(
                        method=method,
                        data=data,
                        var_names=VAR_NAMES,
                        tau_max=TAU_MAX,
                        pc_alpha=PC_ALPHA,
                        verbosity=0,
                    )
                )

                # =============================================
                # Evaluate
                # =============================================

                metric_report = (
                    evaluate_graph(
                        true_edges=
                            true_edges,

                        graph=
                            method_result[
                                "graph"
                            ],

                        n_variables=N,
                        tau_max=TAU_MAX,
                    )
                )

                # =============================================
                # Flatten result
                # =============================================

                row = build_result_row(
                    method_result=
                        method_result,

                    metric_report=
                        metric_report,

                    model_info=
                        model_info,

                    seed=seed,

                    sample_size=
                        sample_size,
                )

                rows.append(
                    row
                )

                print(
                    " | "
                    f"lagTPR="
                    f"{row['lagged_tpr']:.3f} "
                    f"lagFPR="
                    f"{row['lagged_fpr']:.3f} "
                    f"conTPR="
                    f"{row['contemp_tpr']:.3f} "
                    f"time="
                    f"{row['runtime']:.3f}s"
                )

    # ========================================================
    # Save raw results
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # Aggregate across seeds
    # ========================================================

    summary_df = build_summary(
        raw_df
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Print concise summary
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "Benchmark summary"
    )

    print(
        "=" * 72
    )

    display_columns = [
        "method",
        "T",

        "lagged_tpr_mean",
        "lagged_fpr_mean",

        "auto_tpr_mean",
        "auto_fpr_mean",

        "contemp_tpr_mean",
        "contemp_fpr_mean",

        "orient_recall_op_mean",
        "orient_precision_op_mean",

        "runtime_mean",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nRaw results saved to:"
    )

    print(
        raw_path
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_path
    )

    print(
        "\n05 sample-size benchmark completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()