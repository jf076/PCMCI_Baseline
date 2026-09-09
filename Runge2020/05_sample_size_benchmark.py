from benchmark_runner import (
    run_benchmark,
)


def main():

    run_benchmark(

        # ====================================================
        # Experiment identity
        # ====================================================

        experiment_name=
            "sample_size",

        # ====================================================
        # Only T changes
        # ====================================================

        x_column=
            "T",

        x_values=[
            200,
            500,
            1000,
        ],

        # ====================================================
        # Everything else remains fixed
        # ====================================================

        fixed_params={

            "N": 5,

            "a": 0.95,

            "true_tau_max": 5,

            "search_tau_max": 5,
        },

        # ====================================================
        # Four algorithms
        # ====================================================

        methods=[
            "GCresPC",
            "LiNGAM",
            "PC",
            "PCMCI+",
        ],

        # ====================================================
        # First test:
        #
        # use 2 seeds only.
        #
        # After validation:
        # 5 -> 20 -> 500
        # ====================================================

        n_seeds=5,

        pc_alpha=0.01,

        burn_in=500,

        output_prefix=
            "05_sample_size",
    )


if __name__ == "__main__":
    main()