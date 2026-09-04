from pathlib import Path

import numpy as np
import pandas as pd


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def main():

    # ========================================================
    # Paths
    # ========================================================

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    raw_path = (
        script_dir
        / "results"
        / "05_sample_size_raw.csv"
    )

    if not raw_path.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{raw_path}\n"
            "Run 05_sample_size_benchmark.py first."
        )

    # ========================================================
    # Load raw results
    # ========================================================

    raw_df = pd.read_csv(
        raw_path
    )

    print(
        "Loaded:"
    )

    print(
        raw_path
    )

    print(
        "\nTotal raw rows =",
        len(raw_df),
    )

    # ========================================================
    # 1. Check number of realizations
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REALIZATION COUNT CHECK"
    )

    print(
        "=" * 80
    )

    count_table = (
        raw_df
        .groupby(
            [
                "method",
                "T",
            ]
        )
        .agg(
            rows=(
                "seed",
                "size",
            ),

            unique_seeds=(
                "seed",
                "nunique",
            ),

            min_seed=(
                "seed",
                "min",
            ),

            max_seed=(
                "seed",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        count_table.to_string(
            index=False
        )
    )

    # ========================================================
    # Check pairing:
    #
    # every (T, seed) should contain both PC and PCMCI+
    # ========================================================

    pairing = (
        raw_df
        .groupby(
            [
                "T",
                "seed",
            ]
        )["method"]
        .nunique()
    )

    bad_pairs = (
        pairing[
            pairing != 2
        ]
    )

    if len(bad_pairs) == 0:

        print(
            "\nPairing check passed:"
        )

        print(
            "every (T, seed) has exactly two methods."
        )

    else:

        print(
            "\nWARNING: incomplete method pairs:"
        )

        print(
            bad_pairs
        )

    # ========================================================
    # 2. Pooled orientation metrics
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "POOLED ORIENTATION / CONFLICT CHECK"
    )

    print(
        "=" * 80
    )

    rows = []

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

        # ----------------------------------------------------
        # Number of estimated contemporaneous adjacencies
        #
        # estimated = TP + FP
        # ----------------------------------------------------

        estimated_contemp = (
            group[
                "contemp_tp"
            ].sum()
            +
            group[
                "contemp_fp"
            ].sum()
        )

        # ----------------------------------------------------
        # Number of true contemporaneous adjacencies
        #
        # true = TP + FN
        # ----------------------------------------------------

        true_contemp = (
            group[
                "contemp_tp"
            ].sum()
            +
            group[
                "contemp_fn"
            ].sum()
        )

        correct_orientation = (
            group[
                "orient_correct"
            ].sum()
        )

        conflicts = (
            group[
                "orient_conflicts"
            ].sum()
        )

        # ====================================================
        # Existing mean-of-ratios
        # ====================================================

        precision_mean_of_ratios = (
            group[
                "orient_precision_op"
            ]
            .dropna()
            .mean()
        )

        recall_mean_of_ratios = (
            group[
                "orient_recall_op"
            ]
            .dropna()
            .mean()
        )

        conflict_mean_of_ratios = (
            group[
                "conflict_rate"
            ]
            .dropna()
            .mean()
        )

        # ====================================================
        # New pooled ratios
        # ====================================================

        precision_pooled = (
            safe_divide(
                correct_orientation,
                estimated_contemp,
            )
        )

        recall_pooled = (
            safe_divide(
                correct_orientation,
                true_contemp,
            )
        )

        conflict_pooled = (
            safe_divide(
                conflicts,
                estimated_contemp,
            )
        )

        rows.append(
            {
                "method":
                    method,

                "T":
                    sample_size,

                "n_realizations":
                    group[
                        "seed"
                    ].nunique(),

                "estimated_contemp_total":
                    estimated_contemp,

                "true_contemp_total":
                    true_contemp,

                "correct_orientation_total":
                    correct_orientation,

                "conflicts_total":
                    conflicts,

                "precision_mean_ratio":
                    precision_mean_of_ratios,

                "precision_pooled":
                    precision_pooled,

                "recall_mean_ratio":
                    recall_mean_of_ratios,

                "recall_pooled":
                    recall_pooled,

                "conflict_mean_ratio":
                    conflict_mean_of_ratios,

                "conflict_pooled":
                    conflict_pooled,
            }
        )

    pooled_df = pd.DataFrame(
        rows
    )

    percentage_columns = [
        "precision_mean_ratio",
        "precision_pooled",
        "recall_mean_ratio",
        "recall_pooled",
        "conflict_mean_ratio",
        "conflict_pooled",
    ]

    display_df = (
        pooled_df.copy()
    )

    for column in percentage_columns:

        display_df[column] = (
            display_df[column]
            * 100.0
        )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # ========================================================
    # Save diagnostic table
    # ========================================================

    output_path = (
        script_dir
        / "results"
        / "06b_pooled_metrics_check.csv"
    )

    pooled_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        "\n06b pooled-metric check completed."
    )


if __name__ == "__main__":
    main()