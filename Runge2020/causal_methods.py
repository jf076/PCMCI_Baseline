import time

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ============================================================
# 1. Build Tigramite PCMCI object
# ============================================================

def build_pcmci_object(
    data,
    var_names,
    verbosity=0,
):
    """
    Create a Tigramite PCMCI object using ParCorr.

    Parameters
    ----------
    data : np.ndarray
        Time-series data with shape (T, N).

    var_names : list[str]
        Variable names.

    verbosity : int
        Tigramite output verbosity.

    Returns
    -------
    pcmci : tigramite.pcmci.PCMCI
    """

    dataframe = pp.DataFrame(
        data=data,
        var_names=var_names,
    )

    parcorr = ParCorr()

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=verbosity,
    )

    return pcmci


# ============================================================
# 2. Run PCMCI+
# ============================================================

def run_pcmciplus(
    data,
    var_names,
    tau_max,
    pc_alpha=0.01,
    verbosity=0,
):
    """
    Run PCMCI+ using the Runge (2020) linear-Gaussian
    benchmark settings.

    Returns
    -------
    output : dict

        output["method"]
        output["runtime"]
        output["graph"]
        output["p_matrix"]
        output["val_matrix"]
        output["results"]
    """

    pcmci = build_pcmci_object(
        data=data,
        var_names=var_names,
        verbosity=verbosity,
    )

    start_time = time.perf_counter()

    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,

        contemp_collider_rule="majority",
        conflict_resolution=True,

        reset_lagged_links=False,

        # PCMCI+ lagged-selection strategy
        max_combinations=1,

        fdr_method="none",
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "method": "PCMCI+",

        "runtime": runtime,

        "graph":
            results["graph"],

        "p_matrix":
            results["p_matrix"],

        "val_matrix":
            results["val_matrix"],

        "results":
            results,
    }


# ============================================================
# 3. Run standard time-series PC
# ============================================================

def run_pc(
    data,
    var_names,
    tau_max,
    pc_alpha=0.01,
    verbosity=0,
):
    """
    Run standard PC adapted to time series.

    This is the PC baseline used for comparison
    with PCMCI+.

    Important:
        mode="standard"

    Do NOT use:
        mode="contemp_conds"

    because that corresponds to part of PCMCI+.
    """

    pcmci = build_pcmci_object(
        data=data,
        var_names=var_names,
        verbosity=verbosity,
    )

    start_time = time.perf_counter()

    results = pcmci.run_pcalg(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,

        mode="standard",

        contemp_collider_rule="majority",
        conflict_resolution=True,

        max_conds_dim=None,

        # Standard PC considers the required
        # conditioning-set combinations.
        max_combinations=None,
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "method": "PC",

        "runtime": runtime,

        "graph":
            results["graph"],

        "p_matrix":
            results["p_matrix"],

        "val_matrix":
            results["val_matrix"],

        "results":
            results,
    }


# ============================================================
# 4. Generic method dispatcher
# ============================================================

def run_causal_method(
    method,
    data,
    var_names,
    tau_max,
    pc_alpha=0.01,
    verbosity=0,
):
    """
    Generic interface for causal discovery methods.

    Supported methods:
        "PCMCI+"
        "PC"
    """

    method_upper = (
        method
        .strip()
        .upper()
    )

    if method_upper in {
        "PCMCI+",
        "PCMCI_PLUS",
        "PCMCI-PLUS",
    }:

        return run_pcmciplus(
            data=data,
            var_names=var_names,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            verbosity=verbosity,
        )

    if method_upper == "PC":

        return run_pc(
            data=data,
            var_names=var_names,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            verbosity=verbosity,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )