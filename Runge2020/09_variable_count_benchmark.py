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
    # Figure 2B setup
    # ========================================================

    T = 500
    A_MAX = 0.95

    TAU_TRUE_MAX = 5
    TAU_SEARCH_MAX = 5

    PC_ALPHA = 0.01
    BURN_IN = 500

    N_VALUES = [
        2,
        3,
        5,
        10,
        20,
        30,
        40,
    ]

    METHODS = [
        "PCMCI+",
        "PC",
    ]

    # --------------------------------------------------------
    # Start small.
    #
    # N=40 PC can already become considerably slower.
    # --------------------------------------------------------

    N_SEEDS = 3

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
        / "09_variable_count_raw.csv"
    )

    summary_path = (
        output_dir
        / "09_variable_count_summary.csv"
    )

    rows = []

    total_runs = (
        len(N_VALUES)
        * N_SEEDS
        * len(METHODS)
    )

    run_index = 0

    print(
        "Runge (2020) - "
        "Figure 2B-style variable-count benchmark"
    )

    print("=" * 78)

    print(
        "N values =",
        N_VALUES,
    )

    print(
        "T =",
        T,
    )

    print(
        "a =",
        A_MAX,
    )

    print(
        "tau_max =",
        TAU_SEARCH_MAX,
    )

    print(
        "seeds =",
        N_SEEDS,
    )

    # ========================================================
    # Vary N
    # ========================================================

    for n_variables in N_VALUES:

        var_names = [
            f"X{i}"
            for i
            in range(n_variables)
        ]

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"N = {n_variables}"
        )

        print(
            "=" * 78
        )

        for seed in range(
            N_SEEDS
        ):

            # =================================================
            # Generate ONE model/data set for this N, seed
            # =================================================

            (
                data,
                true_edges,
                model_info,
            ) = generate_linear_scm(
                n_variables=
                    n_variables,

                n_samples=T,

                a_max=A_MAX,

                tau_max=
                    TAU_TRUE_MAX,

                seed=seed,

                burn_in=BURN_IN,
            )

            print(
                f"\nSeed {seed:3d} | "
                f"rho="
                f"{model_info['spectral_radius']:.4f} | "
                f"edges={len(true_edges)}"
            )

            for method in METHODS:

                run_index += 1

                print(
                    f"  "
                    f"[{run_index:3d}/{total_runs}] "
                    f"{method:<7}",
                    end="",
                    flush=True,
                )

                result = (
                    run_causal_method(
                        method=method,

                        data=data,

                        var_names=
                            var_names,

                        tau_max=
                            TAU_SEARCH_MAX,

                        pc_alpha=
                            PC_ALPHA,

                        verbosity=0,
                    )
                )

                report = (
                    evaluate_graph(
                        true_edges=
                            true_edges,

                        graph=
                            result["graph"],

                        n_variables=
                            n_variables,

                        tau_max=
                            TAU_SEARCH_MAX,
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

                        N=n_variables,
                    )
                )

                rows.append(
                    row
                )

                print(
                    " | "
                    f"lagTPR="
                    f"{row['lagged_tpr']:.3f} "
                    f"conTPR="
                    f"{row['contemp_tpr']:.3f} "
                    f"lagFPR="
                    f"{row['lagged_fpr']:.3f} "
                    f"time="
                    f"{row['runtime']:.3f}s"
                )

    # ========================================================
    # Save
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

            x_column="N",
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
        "Variable-count benchmark summary"
    )

    print(
        "=" * 78
    )

    columns = [
        "method",
        "N",

        "lagged_tpr_mean",
        "lagged_fpr_mean",

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