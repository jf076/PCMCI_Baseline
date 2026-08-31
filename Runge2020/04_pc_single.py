import importlib
import time

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ============================================================
# 1. Import utilities from previous scripts
# ============================================================

# ------------------------------------------------------------
# SCM generator from 01
# ------------------------------------------------------------

scm_generator = importlib.import_module(
    "01_generate_linear_scm"
)

generate_linear_scm = (
    scm_generator.generate_linear_scm
)

print_ground_truth = (
    scm_generator.print_ground_truth
)


# ------------------------------------------------------------
# Graph printer from 02
# ------------------------------------------------------------

pcmci_single = importlib.import_module(
    "02_pcmciplus_single"
)

print_estimated_graph = (
    pcmci_single.print_estimated_graph
)


# ------------------------------------------------------------
# Evaluation functions from 03
# ------------------------------------------------------------

metrics_module = importlib.import_module(
    "03_metrics"
)

evaluate_graph = (
    metrics_module.evaluate_graph
)

print_metric_report = (
    metrics_module.print_metric_report
)


# ============================================================
# 2. Main
# ============================================================

def main():

    # ========================================================
    # Experimental setup
    #
    # Keep these IDENTICAL to the PCMCI+ single experiment.
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
        "PC single baseline experiment"
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
    # 3. Generate EXACTLY the same seed=0 dataset
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
    # 4. Ground Truth
    #
    # PC does NOT receive this.
    # It is printed only for us as experimenters.
    # ========================================================

    print_ground_truth(
        true_edges
    )

    # ========================================================
    # 5. Tigramite DataFrame
    # ========================================================

    dataframe = pp.DataFrame(
        data=data,
        var_names=VAR_NAMES,
    )

    # ========================================================
    # 6. Conditional independence test
    # ========================================================

    parcorr = ParCorr()

    print(
        "\nConditional independence test:"
        " ParCorr"
    )

    # ========================================================
    # 7. Create PC object
    # ========================================================

    pc = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0,
    )

    # ========================================================
    # 8. Run STANDARD time-series PC
    # ========================================================

    print(
        "\nRunning standard time-series PC..."
    )

    start_time = (
        time.perf_counter()
    )

    results = pc.run_pcalg(

        # ----------------------------------------------------
        # Search contemporaneous AND lagged links
        # ----------------------------------------------------

        tau_min=0,
        tau_max=TAU_MAX,

        # ----------------------------------------------------
        # Same significance threshold as PCMCI+
        # ----------------------------------------------------

        pc_alpha=PC_ALPHA,

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # mode="standard"
        #
        # means standard PC adapted to time series.
        #
        # mode="contemp_conds" would instead implement
        # part of PCMCI+ and is NOT the PC baseline.
        # ----------------------------------------------------

        mode="standard",

        # ----------------------------------------------------
        # Majority-rule PC-stable orientation
        # ----------------------------------------------------

        contemp_collider_rule="majority",

        conflict_resolution=True,

        # ----------------------------------------------------
        # Do not artificially limit conditioning-set size.
        # ----------------------------------------------------

        max_conds_dim=None,

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Standard PC needs to consider the possible
        # conditioning-set combinations.
        #
        # None -> unrestricted / all combinations
        # in current Tigramite implementation.
        #
        # Do NOT copy PCMCI+'s max_combinations=1 here.
        # ----------------------------------------------------

        max_combinations=None,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        "PC finished."
    )

    print(
        f"Runtime = "
        f"{runtime:.4f} s"
    )

    # ========================================================
    # 9. Extract graph matrices
    # ========================================================

    graph = (
        results["graph"]
    )

    p_matrix = (
        results["p_matrix"]
    )

    val_matrix = (
        results["val_matrix"]
    )

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
    # 10. Print estimated PC graph
    # ========================================================

    print_estimated_graph(
        graph=graph,
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        var_names=VAR_NAMES,
        method_name="PC",
    )

    # ========================================================
    # 11. Evaluate with exactly the SAME evaluator as PCMCI+
    # ========================================================

    report = evaluate_graph(
        true_edges=true_edges,
        graph=graph,
        n_variables=N,
        tau_max=TAU_MAX,
    )

    print_metric_report(
        report=report,
        var_names=VAR_NAMES,
    )

    # ========================================================
    # 12. Runtime
    # ========================================================

    print(
        "\n[Runtime]"
    )

    print(
        f"PC runtime = "
        f"{runtime:.4f} s"
    )

    # ========================================================
    # 13. Sanity checks
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
        "\n04 PC single baseline completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()