import importlib
from pathlib import Path

import pandas as pd

from causal_methods import (
    run_causal_method,
)

from benchmark_utils import (
    build_result_row,
    build_summary,
)


scm_module = importlib.import_module(
    "01_generate_linear_scm"
)

generate_linear_scm = (
    scm_module.generate_linear_scm
)


metrics_module = importlib.import_module(
    "03_metrics"
)

evaluate_graph = (
    metrics_module.evaluate_graph
)


def main():

    # ========================================================
    # Figure 2D setup
    # ========================================================

    N = 5
    T = 500

    A_MAX = 0.95

    # --------------------------------------------------------
    # TRUE SCM NEVER CHANGES:
    # all true cross-lags remain <= 5.
    # --------------------------------------------------------

    TAU_TRUE_MAX = 5

    # --------------------------------------------------------
    # Only algorithm search range changes.
    # --------------------------------------------------------

    TAU_SEARCH_VALUES = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    ]

    PC_ALPHA = 0.01

    BURN_IN = 500

    METHODS = [
        "PCMCI+",
        "PC",
    ]

    # First validation
    N_SEEDS = 20

    VAR_NAMES = [
        f"X{i}"
        for i in range(N)
    ]

    output_dir = (
        Path(__file__)
        .resolve()
        .parent
        / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "11_max_lag_raw.csv"
    )

    summary_path = (
        output_dir
        / "11_max_lag_summary.csv"
    )

    rows = []

    total_runs = (
        N_SEEDS
        * len(TAU_SEARCH_VALUES)
        * len(METHODS)
    )

    run_index = 0

    print(
        "Runge (2020) - "
        "Figure 2D-style max-lag benchmark"
    )

    print("=" * 78)

    print(
        "TRUE tau max =",
        TAU_TRUE_MAX,
    )

    print(
        "SEARCH tau values =",
        TAU_SEARCH_VALUES,
    )

    print(
        "N =",
        N,
    )

    print(
        "T =",
        T,
    )

    print(
        "a =",
        A_MAX,
    )

    # ========================================================
    # IMPORTANT:
    #
    # Generate ONE SCM/data set per seed.
    #
    # Then expose EXACTLY SAME DATA to different
    # algorithm search tau_max values.
    # ========================================================

    for seed in range(
        N_SEEDS
    ):

        (
            data,
            true_edges,
            model_info,
        ) = generate_linear_scm(
            n_variables=N,

            n_samples=T,

            a_max=A_MAX,

            # TRUE lag remains 5.
            tau_max=
                TAU_TRUE_MAX,

            seed=seed,

            burn_in=BURN_IN,
        )

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"Seed = {seed} | "
            f"rho="
            f"{model_info['spectral_radius']:.4f}"
        )

        print(
            "=" * 78
        )

        for search_tau_max in (
            TAU_SEARCH_VALUES
        ):

            for method in METHODS:

                run_index += 1

                print(
                    f"  "
                    f"[{run_index:3d}/{total_runs}] "
                    f"tau={search_tau_max:<2} "
                    f"{method:<7}",
                    end="",
                    flush=True,
                )

                result = (
                    run_causal_method(
                        method=method,

                        data=data,

                        var_names=
                            VAR_NAMES,

                        # ONLY THIS changes.
                        tau_max=
                            search_tau_max,

                        pc_alpha=
                            PC_ALPHA,

                        verbosity=0,
                    )
                )

                # =================================================
                # Candidate space must match algorithm search range.
                #
                # True edges are <= 5, but false candidates
                # may exist up to search_tau_max.
                # =================================================

                report = (
                    evaluate_graph(
                        true_edges=
                            true_edges,

                        graph=
                            result["graph"],

                        n_variables=N,

                        tau_max=
                            search_tau_max,
                    )
                )

                row = (
                    build_result_row(
                        method_result=
                            result,

                        metric_report=
                            report,

                        model_info=
                            model_info,

                        seed=seed,

                        search_tau_max=
                            search_tau_max,

                        true_tau_max=
                            TAU_TRUE_MAX,
                    )
                )

                rows.append(
                    row
                )

                print(
                    " | "
                    f"lagTPR="
                    f"{row['lagged_tpr']:.3f} "
                    f"lagFP="
                    f"{row['lagged_fp']} "
                    f"lagFPR="
                    f"{row['lagged_fpr']:.4f} "
                    f"time="
                    f"{row['runtime']:.3f}s"
                )

    # ========================================================
    # Save raw + summary
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    summary_df = (
        build_summary(
            raw_df=
                raw_df,

            x_column=
                "search_tau_max",
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Maximum-lag benchmark summary"
    )

    print(
        "=" * 78
    )

    columns = [
        "method",
        "search_tau_max",

        "lagged_tpr_mean",
        "lagged_fpr_mean",

        "auto_fpr_mean",

        "contemp_tpr_mean",
        "contemp_fpr_mean",

        "orient_recall_op_mean",

        "runtime_mean",
    ]

    print(
        summary_df[
            columns
        ].to_string(
            index=False,

            float_format=
                lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()