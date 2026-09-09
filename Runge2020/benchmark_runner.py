import importlib
import json
from pathlib import Path

import pandas as pd

from causal_methods import (
    run_causal_method,
)

from benchmark_utils import (
    build_result_row,
    build_summary,
)


# ============================================================
# 1. Import SCM generator
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
# 3. Default Figure-2 methods
# ============================================================

DEFAULT_METHODS = [
    "GCresPC",
    "LiNGAM",
    "PC",
    "PCMCI+",
]


# ============================================================
# 4. Resolve one experimental condition
# ============================================================

def resolve_parameters(
    x_column,
    x_value,
    fixed_params,
):
    """
    Combine:

        one changing experimental variable

    with:

        all fixed experimental variables

    to obtain one complete experimental condition.


    Example 1
    ---------
    x_column = "T"
    x_value  = 500

    fixed_params = {
        "N": 5,
        "a": 0.95,
        "true_tau_max": 5,
        "search_tau_max": 5,
    }


    Example 2
    ---------
    x_column = "a"
    x_value  = 0.98

    fixed_params = {
        "N": 5,
        "T": 500,
        "true_tau_max": 5,
        "search_tau_max": 5,
    }


    Returns
    -------
    params : dict
        Complete condition containing:

            N
            T
            a
            true_tau_max
            search_tau_max
    """

    params = dict(
        fixed_params
    )

    # --------------------------------------------------------
    # Insert the experimental variable.
    # --------------------------------------------------------

    params[
        x_column
    ] = x_value

    # --------------------------------------------------------
    # Every benchmark condition must finally define
    # these five quantities.
    # --------------------------------------------------------

    required = [
        "N",
        "T",
        "a",
        "true_tau_max",
        "search_tau_max",
    ]

    missing = [
        name
        for name in required
        if name
        not in params
    ]

    if missing:

        raise ValueError(
            "Incomplete benchmark configuration.\n"
            "Missing parameters:\n"
            + "\n".join(
                missing
            )
        )

    # --------------------------------------------------------
    # Normalize types.
    # --------------------------------------------------------

    params["N"] = int(
        params["N"]
    )

    params["T"] = int(
        params["T"]
    )

    params["a"] = float(
        params["a"]
    )

    params[
        "true_tau_max"
    ] = int(
        params[
            "true_tau_max"
        ]
    )

    params[
        "search_tau_max"
    ] = int(
        params[
            "search_tau_max"
        ]
    )

    # --------------------------------------------------------
    # Current evaluator expects the search space to contain
    # all true lagged edges.
    #
    # Figure 2D satisfies this because:
    #
    #     true_tau_max = 5
    #
    # while:
    #
    #     search_tau_max >= 5
    #
    # --------------------------------------------------------

    if (
        params[
            "search_tau_max"
        ]
        <
        params[
            "true_tau_max"
        ]
    ):

        raise ValueError(
            "Current benchmark evaluator requires:\n"
            "search_tau_max >= true_tau_max\n\n"
            f"Got true_tau_max="
            f"{params['true_tau_max']}, "
            f"search_tau_max="
            f"{params['search_tau_max']}."
        )

    return params


# ============================================================
# 5. Build variable names
# ============================================================

def build_var_names(
    n_variables,
):
    """
    Generate:

        X0, X1, ..., X(N-1)
    """

    return [
        f"X{i}"
        for i in range(
            n_variables
        )
    ]


# ============================================================
# 6. Dataset-generation key
# ============================================================

def build_dataset_key(
    params,
    seed,
    burn_in,
):
    """
    Build a key describing one unique generated dataset.

    Important for Figure 2D:

        search_tau_max changes

    but:

        N
        T
        a
        true_tau_max
        seed

    remain identical.

    Therefore all search_tau_max values should use
    EXACTLY the same generated SCM/data for a given seed.
    """

    return (
        params["N"],
        params["T"],
        params["a"],
        params["true_tau_max"],
        int(seed),
        int(burn_in),
    )


# ============================================================
# 7. Generate or reuse one dataset
# ============================================================

def get_dataset(
    dataset_cache,
    params,
    seed,
    burn_in,
):
    """
    Generate one SCM/data realization.

    A small cache is used within each seed.

    This matters especially for the max-lag benchmark:

        search_tau_max = 5, 10, ..., 40

    must all operate on exactly the same data.
    """

    dataset_key = (
        build_dataset_key(
            params=params,
            seed=seed,
            burn_in=burn_in,
        )
    )

    if (
        dataset_key
        not in dataset_cache
    ):

        dataset_cache[
            dataset_key
        ] = (
            generate_linear_scm(
                n_variables=
                    params["N"],

                n_samples=
                    params["T"],

                a_max=
                    params["a"],

                # --------------------------------------------
                # IMPORTANT:
                #
                # Ground Truth SCM lag range.
                #
                # This is NOT necessarily equal to the
                # algorithm's search_tau_max.
                # --------------------------------------------

                tau_max=
                    params[
                        "true_tau_max"
                    ],

                seed=
                    seed,

                burn_in=
                    burn_in,
            )
        )

    return (
        dataset_cache[
            dataset_key
        ]
    )


# ============================================================
# 8. Run one causal-discovery method
# ============================================================

def run_one_method(
    method,
    data,
    true_edges,
    model_info,
    params,
    seed,
    pc_alpha,
    verbosity,
    experiment_name,
    x_column,
    x_value,
):
    """
    Run ONE method on ONE generated dataset,
    evaluate its graph, and return one flat row.
    """

    var_names = (
        build_var_names(
            params["N"]
        )
    )

    # ========================================================
    # A. Run causal discovery
    # ========================================================

    method_result = (
        run_causal_method(
            method=method,

            data=data,

            var_names=
                var_names,

            # ------------------------------------------------
            # Algorithm search range.
            #
            # Figure 2D changes THIS quantity.
            # ------------------------------------------------

            tau_max=
                params[
                    "search_tau_max"
                ],

            pc_alpha=
                pc_alpha,

            verbosity=
                verbosity,

            # ------------------------------------------------
            # Relevant mainly for LiNGAM.
            #
            # Same seed -> reproducible algorithm state.
            # ------------------------------------------------

            random_state=
                seed,
        )
    )

    # ========================================================
    # B. Evaluate discovered graph
    # ========================================================

    metric_report = (
        evaluate_graph(
            true_edges=
                true_edges,

            graph=
                method_result[
                    "graph"
                ],

            n_variables=
                params["N"],

            # ------------------------------------------------
            # Candidate space must correspond to what
            # the algorithm actually searched.
            # ------------------------------------------------

            tau_max=
                params[
                    "search_tau_max"
                ],
        )
    )

    # ========================================================
    # C. Store complete experimental metadata
    # ========================================================

    experiment_metadata = {

        "experiment":
            experiment_name,

        "N":
            params["N"],

        "T":
            params["T"],

        "a":
            params["a"],

        "true_tau_max":
            params[
                "true_tau_max"
            ],

        "search_tau_max":
            params[
                "search_tau_max"
            ],

        "pc_alpha":
            pc_alpha,
    }

    # Ensure the experimental x variable is stored exactly
    # under the requested x_column name.
    experiment_metadata[
        x_column
    ] = x_value

    # ========================================================
    # D. Flatten everything into one CSV row
    # ========================================================

    row = (
        build_result_row(
            method_result=
                method_result,

            metric_report=
                metric_report,

            model_info=
                model_info,

            seed=
                seed,

            **experiment_metadata,
        )
    )

    return row


# ============================================================
# 9. Print one run
# ============================================================

def print_run_result(
    run_index,
    total_runs,
    row,
    x_column,
    x_value,
):
    """
    Compact progress output.
    """

    method = (
        row["method"]
    )

    print(
        f"  "
        f"[{run_index:4d}/{total_runs}] "
        f"{method:<8} "
        f"| {x_column}={x_value} "
        f"| lagTPR="
        f"{row['lagged_tpr']:.3f} "
        f"| lagFPR="
        f"{row['lagged_fpr']:.3f} "
        f"| conTPR="
        f"{row['contemp_tpr']:.3f} "
        f"| time="
        f"{row['runtime']:.3f}s"
    )


# ============================================================
# 10. Print final concise summary
# ============================================================

def print_benchmark_summary(
    summary_df,
    x_column,
):
    """
    Print the most important summary metrics.
    """

    display_columns = [
        "method",
        x_column,

        "n_realizations",

        "lagged_tpr_mean",
        "lagged_fpr_mean",

        "auto_tpr_mean",
        "auto_fpr_mean",

        "contemp_tpr_mean",
        "contemp_fpr_mean",

        "orient_recall_op_mean",

        "orient_precision_op_pooled",

        "conflict_rate_pooled",

        "runtime_mean",
    ]

    # --------------------------------------------------------
    # Keep only columns that exist.
    # --------------------------------------------------------

    display_columns = [
        column
        for column
        in display_columns
        if column
        in summary_df.columns
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,

            float_format=
                lambda value:
                f"{value:.4f}",
        )
    )


# ============================================================
# 11. Main generic benchmark runner
# ============================================================

def run_benchmark(
    experiment_name,
    x_column,
    x_values,
    fixed_params,
    output_prefix,
    n_seeds,
    methods=None,
    pc_alpha=0.01,
    burn_in=500,
    verbosity=0,
    results_dir=None,
):
    """
    Generic Figure-2 benchmark engine.

    Parameters
    ----------
    experiment_name : str
        Human-readable experiment name.

        Examples:
            "sample_size"
            "autocorrelation"
            "variable_count"
            "max_lag"


    x_column : str
        Variable changed in THIS experiment.

        Examples:
            "T"
            "a"
            "N"
            "search_tau_max"


    x_values : list
        Values of the changing variable.


    fixed_params : dict
        All remaining fixed model/search parameters.

        Example for sample size:

            {
                "N": 5,
                "a": 0.95,
                "true_tau_max": 5,
                "search_tau_max": 5,
            }


    output_prefix : str
        Example:

            "05_sample_size"

        Produces:

            05_sample_size_raw.csv
            05_sample_size_summary.csv
            05_sample_size_config.json


    n_seeds : int
        Number of SCM realizations.


    methods : list[str] or None
        Default:

            GCresPC
            LiNGAM
            PC
            PCMCI+


    pc_alpha : float
        Significance threshold for PC-family algorithms.


    burn_in : int
        SCM simulation burn-in.


    verbosity : int
        Tigramite verbosity.


    results_dir : Path or str or None
        Defaults to:

            Runge2020/results


    Returns
    -------
    raw_df : pd.DataFrame

    summary_df : pd.DataFrame
    """

    # ========================================================
    # A. Methods
    # ========================================================

    if methods is None:

        methods = list(
            DEFAULT_METHODS
        )

    # ========================================================
    # B. Output directory
    # ========================================================

    if results_dir is None:

        results_dir = (
            Path(__file__)
            .resolve()
            .parent
            / "results"
        )

    else:

        results_dir = Path(
            results_dir
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        results_dir
        / f"{output_prefix}_raw.csv"
    )

    summary_path = (
        results_dir
        / f"{output_prefix}_summary.csv"
    )

    config_path = (
        results_dir
        / f"{output_prefix}_config.json"
    )

    # ========================================================
    # C. Save experimental configuration
    # ========================================================

    config = {
        "experiment_name":
            experiment_name,

        "x_column":
            x_column,

        "x_values":
            list(
                x_values
            ),

        "fixed_params":
            dict(
                fixed_params
            ),

        "methods":
            list(
                methods
            ),

        "n_seeds":
            int(
                n_seeds
            ),

        "pc_alpha":
            float(
                pc_alpha
            ),

        "burn_in":
            int(
                burn_in
            ),
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # D. Header
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Benchmark: {experiment_name}"
    )

    print(
        "=" * 80
    )

    print(
        "Changing variable =",
        x_column,
    )

    print(
        "Values =",
        list(
            x_values
        ),
    )

    print(
        "Methods =",
        methods,
    )

    print(
        "Seeds =",
        n_seeds,
    )

    print(
        "Fixed parameters =",
        fixed_params,
    )

    # ========================================================
    # E. Total algorithm runs
    # ========================================================

    total_runs = (
        len(
            x_values
        )
        *
        n_seeds
        *
        len(
            methods
        )
    )

    run_index = 0

    rows = []

    # ========================================================
    # F. Seed loop
    #
    # We put seed outside the condition loop so that the
    # same realization can be reused when appropriate.
    #
    # Especially:
    #
    # Figure 2D:
    #
    #   search_tau_max changes
    #
    # but the SCM/data must remain identical.
    # ========================================================

    for seed in range(
        n_seeds
    ):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Seed = {seed}"
        )

        print(
            "-" * 80
        )

        # ----------------------------------------------------
        # Cache is local to ONE seed.
        #
        # It avoids keeping the whole benchmark dataset
        # collection in memory.
        # ----------------------------------------------------

        dataset_cache = {}

        # ====================================================
        # Experimental-condition loop
        # ====================================================

        for x_value in x_values:

            params = (
                resolve_parameters(
                    x_column=
                        x_column,

                    x_value=
                        x_value,

                    fixed_params=
                        fixed_params,
                )
            )

            # =================================================
            # Generate/reuse dataset
            # =================================================

            (
                data,
                true_edges,
                model_info,
            ) = get_dataset(
                dataset_cache=
                    dataset_cache,

                params=
                    params,

                seed=
                    seed,

                burn_in=
                    burn_in,
            )

            print(
                f"\n"
                f"{x_column} = {x_value} "
                f"| N={params['N']} "
                f"| T={params['T']} "
                f"| a={params['a']} "
                f"| true_tau="
                f"{params['true_tau_max']} "
                f"| search_tau="
                f"{params['search_tau_max']} "
                f"| rho="
                f"{model_info['spectral_radius']:.4f} "
                f"| attempt="
                f"{model_info['attempts']}"
            )

            # =================================================
            # Run all methods on EXACTLY SAME data
            # =================================================

            for method in methods:

                run_index += 1

                row = (
                    run_one_method(
                        method=
                            method,

                        data=
                            data,

                        true_edges=
                            true_edges,

                        model_info=
                            model_info,

                        params=
                            params,

                        seed=
                            seed,

                        pc_alpha=
                            pc_alpha,

                        verbosity=
                            verbosity,

                        experiment_name=
                            experiment_name,

                        x_column=
                            x_column,

                        x_value=
                            x_value,
                    )
                )

                rows.append(
                    row
                )

                print_run_result(
                    run_index=
                        run_index,

                    total_runs=
                        total_runs,

                    row=
                        row,

                    x_column=
                        x_column,

                    x_value=
                        x_value,
                )

        # ====================================================
        # G. Checkpoint after every seed
        #
        # If a long N=40 / tau=40 experiment is interrupted,
        # at least completed seeds remain on disk.
        # ====================================================

        checkpoint_df = (
            pd.DataFrame(
                rows
            )
        )

        checkpoint_df.to_csv(
            raw_path,
            index=False,
        )

    # ========================================================
    # H. Final raw table
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    # ========================================================
    # I. Aggregate across seeds
    # ========================================================

    summary_df = (
        build_summary(
            raw_df=
                raw_df,

            x_column=
                x_column,
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # J. Final summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{experiment_name} summary"
    )

    print(
        "=" * 80
    )

    print_benchmark_summary(
        summary_df=
            summary_df,

        x_column=
            x_column,
    )

    # ========================================================
    # K. Output paths
    # ========================================================

    print(
        "\nSaved:"
    )

    print(
        raw_path
    )

    print(
        summary_path
    )

    print(
        config_path
    )

    print(
        "\nBenchmark completed."
    )

    # ========================================================
    # L. Return dataframes
    # ========================================================

    return (
        raw_df,
        summary_df,
    )