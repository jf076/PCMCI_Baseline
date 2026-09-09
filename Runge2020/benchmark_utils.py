import numpy as np
import pandas as pd


# ============================================================
# 1. Safe division
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    """
    Safely divide two values.

    Returns
    -------
    float
        numerator / denominator

        If denominator == 0, return np.nan.
    """

    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
    )


# ============================================================
# 2. Convert one method run into one flat result row
# ============================================================

def build_result_row(
    method_result,
    metric_report,
    model_info,
    seed,
    **experiment_values,
):
    """
    Convert the result of ONE causal-discovery run
    into one flat dictionary suitable for:

        pandas.DataFrame
        CSV storage
        later aggregation

    Parameters
    ----------
    method_result : dict
        Returned by run_causal_method().

        Expected fields include:

            method
            runtime
            graph

    metric_report : dict
        Returned by evaluate_graph() in 03_metrics.py.

    model_info : dict
        Metadata returned by generate_linear_scm().

    seed : int
        Random realization seed.

    **experiment_values
        Experimental variables.

        Examples:

            T=500

            a=0.95

            N=20

            search_tau_max=15

            true_tau_max=5

    Returns
    -------
    row : dict
        Flat result row.
    """

    # ========================================================
    # Extract metric categories
    # ========================================================

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

    # ========================================================
    # Method-specific metric applicability
    # ========================================================

    method_name = (
        method_result[
            "method"
        ]
    )

    is_lingam = (
        method_name
        == "LiNGAM"
    )

    # --------------------------------------------------------
    # Why treat LiNGAM specially?
    #
    # PC / PCMCI+ / GCresPC:
    #
    #     contemporaneous graph may contain
    #     x-x conflict states.
    #
    # LiNGAM:
    #
    #     outputs a directed structural graph directly.
    #     It does NOT have the PC-style concept of
    #     conflicting orientation rules.
    #
    # Therefore:
    #
    #     conflict = N/A
    #
    # rather than:
    #
    #     conflict = 0
    #
    # --------------------------------------------------------

    if is_lingam:

        orient_conflicts = np.nan
        conflict_rate = np.nan

        # LiNGAM does not use the same alpha-based
        # CI-testing FPR control as PC-family methods.
        fpr_control_applicable = False

        conflict_metric_applicable = False

    else:

        orient_conflicts = (
            orient[
                "conflicts"
            ]
        )

        conflict_rate = (
            orient[
                "conflict_rate"
            ]
        )

        fpr_control_applicable = True

        conflict_metric_applicable = True

    # ========================================================
    # Build flat row
    # ========================================================

    row = {

        # ====================================================
        # Experimental identity
        # ====================================================

        "method":
            method_name,

        "seed":
            seed,

        # ====================================================
        # SCM metadata
        # ====================================================

        "spectral_radius":
            model_info[
                "spectral_radius"
            ],

        "generation_attempts":
            model_info[
                "attempts"
            ],

        # ====================================================
        # Runtime
        # ====================================================

        "runtime":
            method_result[
                "runtime"
            ],

        # ====================================================
        # Lagged cross-link adjacency
        # ====================================================

        "lagged_tp":
            lagged[
                "TP"
            ],

        "lagged_fp":
            lagged[
                "FP"
            ],

        "lagged_fn":
            lagged[
                "FN"
            ],

        "lagged_tn":
            lagged[
                "TN"
            ],

        "lagged_tpr":
            lagged[
                "TPR"
            ],

        "lagged_fpr":
            lagged[
                "FPR"
            ],

        # ====================================================
        # Autodependency adjacency
        # ====================================================

        "auto_tp":
            auto[
                "TP"
            ],

        "auto_fp":
            auto[
                "FP"
            ],

        "auto_fn":
            auto[
                "FN"
            ],

        "auto_tn":
            auto[
                "TN"
            ],

        "auto_tpr":
            auto[
                "TPR"
            ],

        "auto_fpr":
            auto[
                "FPR"
            ],

        # ====================================================
        # Contemporaneous adjacency
        # ====================================================

        "contemp_tp":
            contemp[
                "TP"
            ],

        "contemp_fp":
            contemp[
                "FP"
            ],

        "contemp_fn":
            contemp[
                "FN"
            ],

        "contemp_tn":
            contemp[
                "TN"
            ],

        "contemp_tpr":
            contemp[
                "TPR"
            ],

        "contemp_fpr":
            contemp[
                "FPR"
            ],

        # ====================================================
        # Current operational orientation metrics
        #
        # IMPORTANT:
        #
        # These are still our current DAG-direction
        # operational metrics.
        #
        # They are NOT yet the final CPDAG-aware
        # Runge (2020) orientation metrics.
        # ====================================================

        "orient_correct":
            orient[
                "correct"
            ],

        "orient_reversed":
            orient[
                "reversed"
            ],

        "orient_unresolved":
            orient[
                "unresolved"
            ],

        # LiNGAM:
        #     NaN
        #
        # PC / PCMCI+ / GCresPC:
        #     actual number of x-x conflicts
        #
        "orient_conflicts":
            orient_conflicts,

        "orient_recall_op":
            orient[
                "recall"
            ],

        "orient_precision_op":
            orient[
                "precision"
            ],

        "conflict_rate":
            conflict_rate,

        # ====================================================
        # Applicability flags
        # ====================================================

        "fpr_control_applicable":
            fpr_control_applicable,

        "conflict_metric_applicable":
            conflict_metric_applicable,
        # ====================================================
        # ICA是否收敛，主要针对LiNGAM算法
        # ====================================================
        "algorithm_converged":
            method_result.get(
                "ica_converged",
                True,
            ),
    }

    # ========================================================
    # Add experiment-specific values
    # ========================================================
    #
    # Examples:
    #
    #     T=500
    #
    #     a=0.95
    #
    #     N=20
    #
    #     search_tau_max=20
    #
    # ========================================================

    row.update(
        experiment_values
    )

    return row


# ============================================================
# 3. Aggregate multiple realizations
# ============================================================

def build_summary(
    raw_df,
    x_column,
):
    """
    Aggregate raw benchmark results across realizations.

    Grouping:

        method × x_column

    Examples
    --------

        x_column = "T"

        x_column = "a"

        x_column = "N"

        x_column = "search_tau_max"

    For each rate metric, calculate:

        mean
        standard deviation
        standard error of the mean (SEM)

    Runtime additionally includes:

        empirical 5th percentile
        empirical 95th percentile

    Contemporaneous orientation also includes
    pooled metrics.
    """

    # ========================================================
    # Metrics aggregated realization-by-realization
    # ========================================================

    metric_columns = [

        # Lagged adjacency
        "lagged_tpr",
        "lagged_fpr",

        # Auto adjacency
        "auto_tpr",
        "auto_fpr",

        # Contemporaneous adjacency
        "contemp_tpr",
        "contemp_fpr",

        # Operational orientation
        "orient_recall_op",
        "orient_precision_op",

        # PC-family conflict rate
        "conflict_rate",

        # Runtime
        "runtime",
    ]

    summary_rows = []

    # ========================================================
    # Group by:
    #
    #     method × experimental variable
    # ========================================================

    grouped = raw_df.groupby(
        [
            "method",
            x_column,
        ],
        sort=True,
    )

    # ========================================================
    # Process every experimental condition
    # ========================================================

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
        # Applicability metadata
        # ====================================================

        if (
            "fpr_control_applicable"
            in group.columns
        ):

            row[
                "fpr_control_applicable"
            ] = bool(
                group[
                    "fpr_control_applicable"
                ]
                .all()
            )

        if (
            "conflict_metric_applicable"
            in group.columns
        ):

            row[
                "conflict_metric_applicable"
            ] = bool(
                group[
                    "conflict_metric_applicable"
                ]
                .all()
            )

        # ====================================================
        # A. Mean / std / SEM
        # ====================================================

        for metric in metric_columns:

            # ------------------------------------------------
            # dropna() is important.
            #
            # Examples:
            #
            # 1. orientation precision may be NaN
            #    if no contemporaneous adjacency was found.
            #
            # 2. LiNGAM conflict_rate is intentionally NaN.
            #
            # 3. auto TPR can be NaN at a=0 if the true
            #    SCM contains no non-zero autodependency.
            # ------------------------------------------------

            values = (
                group[
                    metric
                ]
                .dropna()
                .astype(float)
            )

            # ------------------------------------------------
            # No applicable observations
            # ------------------------------------------------

            if len(values) == 0:

                mean = np.nan
                std = np.nan
                sem = np.nan

            # ------------------------------------------------
            # At least one observation
            # ------------------------------------------------

            else:

                mean = (
                    values.mean()
                )

                # --------------------------------------------
                # std / SEM require >= 2 realizations
                # --------------------------------------------

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

            # ------------------------------------------------
            # Keep track of how many non-NaN realizations
            # actually contributed to this metric.
            #
            # This is useful especially for:
            #
            # orientation precision
            # conflict rate
            #
            # because their denominator may be absent.
            # ------------------------------------------------

            row[
                f"{metric}_n"
            ] = len(
                values
            )

        # ====================================================
        # B. Runtime 90% empirical range
        # ====================================================

        runtime_values = (
            group[
                "runtime"
            ]
            .dropna()
            .astype(float)
        )

        if len(
            runtime_values
        ) > 0:

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

        # ====================================================
        # C. Pooled contemporaneous quantities
        # ====================================================
        #
        # estimated contemporaneous adjacency:
        #
        #     TP + FP
        #
        # true contemporaneous adjacency:
        #
        #     TP + FN
        #
        # ====================================================

        estimated_contemp_total = (
            group[
                "contemp_tp"
            ].sum()
            +
            group[
                "contemp_fp"
            ].sum()
        )

        true_contemp_total = (
            group[
                "contemp_tp"
            ].sum()
            +
            group[
                "contemp_fn"
            ].sum()
        )

        correct_orientation_total = (
            group[
                "orient_correct"
            ].sum()
        )

        # ====================================================
        # D. Pooled orientation precision
        # ====================================================
        #
        #                   sum(correct direction)
        # precision_pool = ------------------------
        #                   sum(estimated contemp)
        #
        # ====================================================

        orientation_precision_pooled = (
            safe_divide(
                correct_orientation_total,
                estimated_contemp_total,
            )
        )

        # ====================================================
        # E. Pooled orientation recall
        # ====================================================
        #
        #                sum(correct direction)
        # recall_pool = ----------------------
        #                sum(true contemp)
        #
        # ====================================================

        orientation_recall_pooled = (
            safe_divide(
                correct_orientation_total,
                true_contemp_total,
            )
        )

        # ====================================================
        # F. Pooled conflicts
        # ====================================================
        #
        # Important for LiNGAM:
        #
        # orient_conflicts contains only NaN.
        #
        # pandas.sum() over an all-NaN series would normally
        # give 0.0, which would incorrectly imply:
        #
        #     LiNGAM conflict rate = 0
        #
        # We therefore explicitly check whether any valid
        # conflict observations exist.
        # ====================================================

        conflict_values = (
            group[
                "orient_conflicts"
            ]
            .dropna()
            .astype(float)
        )

        if len(
            conflict_values
        ) == 0:

            conflict_total = np.nan

            conflict_rate_pooled = (
                np.nan
            )

        else:

            conflict_total = (
                conflict_values.sum()
            )

            conflict_rate_pooled = (
                safe_divide(
                    conflict_total,
                    estimated_contemp_total,
                )
            )

        # ====================================================
        # Store pooled values
        # ====================================================

        row[
            "estimated_contemp_total"
        ] = (
            estimated_contemp_total
        )

        row[
            "true_contemp_total"
        ] = (
            true_contemp_total
        )

        row[
            "orient_correct_total"
        ] = (
            correct_orientation_total
        )

        row[
            "orient_conflicts_total"
        ] = (
            conflict_total
        )

        row[
            "orient_precision_op_pooled"
        ] = (
            orientation_precision_pooled
        )

        row[
            "orient_recall_op_pooled"
        ] = (
            orientation_recall_pooled
        )

        row[
            "conflict_rate_pooled"
        ] = (
            conflict_rate_pooled
        )

        # ====================================================
        # Finish this method × condition
        # ====================================================

        summary_rows.append(
            row
        )

    # ========================================================
    # Return summary table
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    return summary_df