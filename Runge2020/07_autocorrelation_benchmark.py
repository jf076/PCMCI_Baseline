from benchmark_runner import (
    run_benchmark,
)


def main():

    run_benchmark(

        # ====================================================
        # Figure 2A-style:
        # autocorrelation experiment
        # ====================================================

        experiment_name=
            "autocorrelation",

        # ----------------------------------------------------
        # Only a changes
        # ----------------------------------------------------

        x_column=
            "a",

        x_values=[
            0.0,
            0.4,
            0.6,
            0.9,
            0.98,
            0.999,
        ],

        # ----------------------------------------------------
        # Everything else remains fixed
        # ----------------------------------------------------

        fixed_params={

            "N": 5,

            "T": 500,

            "true_tau_max": 5,

            "search_tau_max": 5,
        },

        # ----------------------------------------------------
        # First validation:
        #     n_seeds = 2
        #
        # Formal experiment:
        #     n_seeds = 20
        #
        # Paper-scale:
        #     n_seeds = 500
        # ----------------------------------------------------

        n_seeds=5,

        pc_alpha=0.01,

        burn_in=500,

        output_prefix=
            "07_autocorrelation",
    )


if __name__ == "__main__":
    main()