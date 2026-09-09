from benchmark_runner import (
    run_benchmark,
)


def main():

    run_benchmark(

        # ====================================================
        # Figure 2D-style:
        # maximum searched lag experiment
        # ====================================================

        experiment_name=
            "max_lag",

        # ----------------------------------------------------
        # ONLY algorithm search range changes.
        #
        # Ground Truth tau_max remains fixed at 5.
        # ----------------------------------------------------

        x_column=
            "search_tau_max",

        x_values=[
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
        ],

        # ----------------------------------------------------
        # Everything else remains fixed
        # ----------------------------------------------------

        fixed_params={

            "N": 5,

            "T": 500,

            "a": 0.95,

            # IMPORTANT:
            #
            # The true SCM always has
            # maximum lag = 5.
            #
            "true_tau_max": 5,
        },

        # ----------------------------------------------------
        # tau=40 can become much slower for
        # PC / GCresPC / LiNGAM.
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
            "11_max_lag",
    )


if __name__ == "__main__":
    main()