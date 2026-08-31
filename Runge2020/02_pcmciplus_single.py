import importlib
import time

import numpy as np

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ============================================================
# 1. Load generator from 01_generate_linear_scm.py
# ============================================================

# We cannot write:
#
# from 01_generate_linear_scm import ...
#
# because Python identifiers cannot start with a digit.
#
# importlib allows us to keep the numbered filename.

scm_generator = importlib.import_module(
    "01_generate_linear_scm"
)

generate_linear_scm = (
    scm_generator.generate_linear_scm
)

print_ground_truth = (
    scm_generator.print_ground_truth
)


# ============================================================
# 2. Helper: print PCMCI+ estimated graph
# ============================================================

def print_estimated_graph(
    graph,
    p_matrix,
    val_matrix,
    var_names,
    method_name="PCMCI+",
):
    """
    Print PCMCI+ estimated graph using the categories
    used in Runge (2020):

    1. Autodependency links
    2. Lagged cross-links
    3. Contemporaneous links

    Definitions
    -----------
    Autodependency:
        Xi(t-tau) -> Xi(t), tau > 0

    Lagged cross-link:
        Xi(t-tau) -> Xj(t),
        i != j, tau > 0

    Contemporaneous:
        Xi(t) -- Xj(t), tau = 0
    """

    n_variables = graph.shape[0]
    tau_max = graph.shape[2] - 1

    print(f"\n{method_name} Estimated Graph:")
    print("-" * 80)

    # ========================================================
    # 1. Autodependency links
    # ========================================================

    print("\n[Autodependency links]")

    n_auto = 0

    for tau in range(
        1,
        tau_max + 1,
    ):

        for variable in range(
            n_variables
        ):

            link = graph[
                variable,
                variable,
                tau,
            ]

            if link == "":
                continue

            n_auto += 1

            p_value = p_matrix[
                variable,
                variable,
                tau,
            ]

            value = val_matrix[
                variable,
                variable,
                tau,
            ]

            print(
                f"{var_names[variable]}"
                f"(t-{tau}) "
                f"-> "
                f"{var_names[variable]}(t)"
                f"    "
                f"p={p_value:.5f}"
                f"    "
                f"val={value:+.4f}"
            )

    if n_auto == 0:

        print(
            "No autodependency links detected."
        )

    # ========================================================
    # 2. Lagged cross-links
    # ========================================================

    print("\n[Lagged cross-links]")

    n_lagged_cross = 0

    for tau in range(
        1,
        tau_max + 1,
    ):

        for source in range(
            n_variables
        ):

            for target in range(
                n_variables
            ):

                # Cross-link requires:
                # source != target
                if source == target:
                    continue

                link = graph[
                    source,
                    target,
                    tau,
                ]

                if link == "":
                    continue

                n_lagged_cross += 1

                p_value = p_matrix[
                    source,
                    target,
                    tau,
                ]

                value = val_matrix[
                    source,
                    target,
                    tau,
                ]

                print(
                    f"{var_names[source]}"
                    f"(t-{tau}) "
                    f"-> "
                    f"{var_names[target]}(t)"
                    f"    "
                    f"p={p_value:.5f}"
                    f"    "
                    f"val={value:+.4f}"
                )

    if n_lagged_cross == 0:

        print(
            "No lagged cross-links detected."
        )

    # ========================================================
    # 3. Contemporaneous links
    # ========================================================

    print(
        "\n[Contemporaneous links]"
    )

    n_contemp = 0

    # For tau = 0 Tigramite stores the two
    # representations symmetrically.
    #
    # We inspect only i < j to avoid counting
    # one adjacency twice.
    for i in range(
        n_variables
    ):

        for j in range(
            i + 1,
            n_variables,
        ):

            link = graph[
                i,
                j,
                0,
            ]

            if link == "":
                continue

            n_contemp += 1

            p_value = p_matrix[
                i,
                j,
                0,
            ]

            value = val_matrix[
                i,
                j,
                0,
            ]

            if link == "-->":

                relation = (
                    f"{var_names[i]}(t) "
                    f"-> "
                    f"{var_names[j]}(t)"
                )

            elif link == "<--":

                relation = (
                    f"{var_names[j]}(t) "
                    f"-> "
                    f"{var_names[i]}(t)"
                )

            elif link == "o-o":

                relation = (
                    f"{var_names[i]}(t) "
                    f"o-o "
                    f"{var_names[j]}(t)"
                )

            elif link == "x-x":

                relation = (
                    f"{var_names[i]}(t) "
                    f"x-x "
                    f"{var_names[j]}(t)"
                )

            else:

                relation = (
                    f"{var_names[i]}(t) "
                    f"{link} "
                    f"{var_names[j]}(t)"
                )

            print(
                f"{relation:<25}"
                f"p={p_value:.5f}"
                f"    "
                f"val={value:+.4f}"
            )

    if n_contemp == 0:

        print(
            "No contemporaneous links detected."
        )

    # ========================================================
    # 4. Summary
    # ========================================================

    print(
        "\nEstimated graph summary:"
    )

    print(
        "autodependency links  =",
        n_auto,
    )

    print(
        "lagged cross-links    =",
        n_lagged_cross,
    )

    print(
        "contemporaneous adj.  =",
        n_contemp,
    )


# ============================================================
# 3. Main
# ============================================================

def main():

    # ========================================================
    # Experimental setup
    # ========================================================

    N = 5
    T = 500

    A_MAX = 0.95

    TAU_MAX = 5

    PC_ALPHA = 0.01

    SEED = 0

    BURN_IN = 500

    VAR_NAMES = [
        f"X{i}"
        for i in range(N)
    ]

    print(
        "Runge (2020) - "
        "PCMCI+ single experiment"
    )

    print("=" * 70)

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

    print(
        "tau_max =",
        TAU_MAX,
    )

    print(
        "pc_alpha =",
        PC_ALPHA,
    )

    print(
        "seed =",
        SEED,
    )

    # ========================================================
    # 4. Generate one stationary SCM dataset
    # ========================================================

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

    print(
        "\nGenerated dataset:"
    )

    print(
        "data shape =",
        data.shape,
    )

    print(
        "spectral radius = "
        f"{model_info['spectral_radius']:.6f}"
    )

    print(
        "generation attempts =",
        model_info["attempts"],
    )

    # ========================================================
    # 5. Print Ground Truth
    #
    # IMPORTANT:
    # This is ONLY for us as experimenters.
    # PCMCI+ does not receive true_edges.
    # ========================================================

    print_ground_truth(
        true_edges
    )

    # ========================================================
    # 6. Convert NumPy data to Tigramite DataFrame
    # ========================================================

    dataframe = pp.DataFrame(
        data=data,
        var_names=VAR_NAMES,
    )

    print(
        "\nTigramite DataFrame created."
    )

    # ========================================================
    # 7. Conditional independence test
    # ========================================================

    parcorr = ParCorr()

    print(
        "Conditional independence test:"
        " ParCorr"
    )

    # ========================================================
    # 8. PCMCI+ object
    # ========================================================

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0,
    )

    # ========================================================
    # 9. Run PCMCI+
    # ========================================================

    print(
        "\nRunning PCMCI+..."
    )

    start_time = (
        time.perf_counter()
    )

    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        reset_lagged_links=False,
        max_combinations=1,
        fdr_method="none",
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        "PCMCI+ finished."
    )

    print(
        f"Runtime = "
        f"{runtime:.4f} s"
    )

    # ========================================================
    # 10. Extract result matrices
    # ========================================================

    graph = results[
        "graph"
    ]

    p_matrix = results[
        "p_matrix"
    ]

    val_matrix = results[
        "val_matrix"
    ]

    print(
        "\nResult matrix shapes:"
    )

    print(
        "graph      =",
        graph.shape,
    )

    print(
        "p_matrix   =",
        p_matrix.shape,
    )

    print(
        "val_matrix =",
        val_matrix.shape,
    )

    # ========================================================
    # 11. Print readable estimated graph
    # ========================================================

    print_estimated_graph(
        graph=graph,
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        var_names=VAR_NAMES,
    )

    # ========================================================
    # 12. Final check
    # ========================================================

    assert graph.shape == (
        N,
        N,
        TAU_MAX + 1,
    )

    assert p_matrix.shape == (
        N,
        N,
        TAU_MAX + 1,
    )

    assert val_matrix.shape == (
        N,
        N,
        TAU_MAX + 1,
    )

    print(
        "\n02 PCMCI+ single run completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()