import numpy as np
import pandas as pd


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def build_result_row(
    method_result,
    metric_report,
    model_info,
    seed,
    **experiment_values,
):
    """
    Convert one method run into a flat result row.

    experiment_values examples:
        N=10
        search_tau_max=20
    """

    lagged = metric_report[
        "lagged_cross"
    ]

    auto = metric_report[
        "auto"
    ]

    contemp = metric_report[
        "contemporaneous"
    ]

    orient = metric_report[
        "orientation"
    ]

    row = {
        "method":
            method_result["method"],

        "seed":
            seed,

        "spectral_radius":
            model_info[
                "spectral_radius"
            ],

        "generation_attempts":
            model_info[
                "attempts"
            ],

        "runtime":
            method_result[
                "runtime"
            ],

        # Lagged cross
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

        # Auto
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

        # Contemporaneous
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

        # Orientation
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

    row.update(
        experiment_values
    )

    return row


def build_summary(
    raw_df,
    x_column,
):
    """
    Aggregate:
        method × experimental variable

    Example:
        x_column = "N"
        x_column = "search_tau_max"
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
            x_column,
        ],
        sort=True,
    )

    for (
        method,
        x_value,
    ), group in grouped:

        row = {
            "method":
                method,

            x_column:
                x_value,

            "n_realizations":
                group[
                    "seed"
                ].nunique(),
        }

        # ====================================================
        # Mean / std / SEM
        # ====================================================

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

                mean = values.mean()

                if len(values) > 1:

                    std = values.std(
                        ddof=1
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

        # ====================================================
        # Runtime 90% empirical interval
        # ====================================================

        runtime_values = (
            group["runtime"]
            .dropna()
            .astype(float)
        )

        if len(runtime_values) > 0:

            row["runtime_p05"] = (
                np.percentile(
                    runtime_values,
                    5,
                )
            )

            row["runtime_p95"] = (
                np.percentile(
                    runtime_values,
                    95,
                )
            )

        else:

            row["runtime_p05"] = np.nan
            row["runtime_p95"] = np.nan

        # ====================================================
        # Pooled orientation statistics
        # ====================================================

        estimated_contemp = (
            group[
                "contemp_tp"
            ].sum()
            +
            group[
                "contemp_fp"
            ].sum()
        )

        true_contemp = (
            group[
                "contemp_tp"
            ].sum()
            +
            group[
                "contemp_fn"
            ].sum()
        )

        correct = (
            group[
                "orient_correct"
            ].sum()
        )

        conflicts = (
            group[
                "orient_conflicts"
            ].sum()
        )

        row[
            "estimated_contemp_total"
        ] = estimated_contemp

        row[
            "true_contemp_total"
        ] = true_contemp

        row[
            "orient_correct_total"
        ] = correct

        row[
            "orient_conflicts_total"
        ] = conflicts

        row[
            "orient_precision_op_pooled"
        ] = safe_divide(
            correct,
            estimated_contemp,
        )

        row[
            "orient_recall_op_pooled"
        ] = safe_divide(
            correct,
            true_contemp,
        )

        row[
            "conflict_rate_pooled"
        ] = safe_divide(
            conflicts,
            estimated_contemp,
        )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )