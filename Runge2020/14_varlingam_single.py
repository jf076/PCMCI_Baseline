import importlib

from causal_methods import (
    run_varlingam,
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

print_metric_report = (
    metrics_module.print_metric_report
)


print_module = importlib.import_module(
    "02_pcmciplus_single"
)

print_estimated_graph = (
    print_module.print_estimated_graph
)


def main():

    N = 5
    T = 500
    A_MAX = 0.95
    TAU_MAX = 5
    SEED = 0
    BURN_IN = 500

    VAR_NAMES = [
        f"X{i}"
        for i in range(N)
    ]

    (
        data,
        true_edges,
        model_info,
    ) = generate_linear_scm(
        n_variables=N,
        n_samples=T,
        a_max=A_MAX,
        tau_max=TAU_MAX,
        seed=SEED,
        burn_in=BURN_IN,
    )

    result = run_varlingam(
        data=data,
        var_names=VAR_NAMES,
        tau_max=TAU_MAX,
        random_state=SEED,
    )

    print(
        "\nLiNGAM runtime = "
        f"{result['runtime']:.4f} s"
    )

    print(
        "B matrix shape =",
        result[
            "adjacency_matrices"
        ].shape,
    )

    print(
        "Causal order =",
        result[
            "causal_order"
        ],
    )

    print(
        "\nNOTE:"
        " LiNGAM has no PC-style edge p-values, "
        "so p=nan is expected."
    )

    print_estimated_graph(
        graph=
            result["graph"],

        p_matrix=
            result["p_matrix"],

        val_matrix=
            result["val_matrix"],

        var_names=
            VAR_NAMES,

        method_name=
            "LiNGAM",
    )

    report = evaluate_graph(
        true_edges=true_edges,
        graph=result["graph"],
        n_variables=N,
        tau_max=TAU_MAX,
    )

    print_metric_report(
        report=report,
        var_names=VAR_NAMES,
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "LiNGAM conflict rate should be treated "
        "as N/A rather than 0."
    )

    print(
        "LiNGAM FPR is descriptive only; "
        "there is no alpha=0.01 false-positive "
        "control comparable with PC/PCMCI+."
    )


if __name__ == "__main__":
    main()