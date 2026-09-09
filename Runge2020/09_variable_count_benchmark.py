from benchmark_runner import (
    run_benchmark,
)


def main():

    run_benchmark(

        # ====================================================
        # Figure 2B-style:
        # variable-count experiment
        # ====================================================

        experiment_name=
            "variable_count",

        # ----------------------------------------------------
        # Only N changes
        # ----------------------------------------------------

        x_column=
            "N",

        x_values=[
            2,
            3,
            5,
            10,
            20,
            30,
            40,
        ],

        # ----------------------------------------------------
        # Everything else remains fixed
        # ----------------------------------------------------

        fixed_params={

            "T": 500,

            "a": 0.95,

            "true_tau_max": 5,

            "search_tau_max": 5,
        },

        # ----------------------------------------------------
        # N=40 can be substantially slower,
        # especially for standard PC.
        #
        # Smoke test:
        #     2
        #
        # Current formal run:
        #     20
        # ----------------------------------------------------

        n_seeds=5,

        pc_alpha=0.01,

        burn_in=500,

        output_prefix=
            "09_variable_count",
    )


if __name__ == "__main__":
    main()