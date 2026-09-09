import time

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning

try:
    from lingam import (
        VARLiNGAM,
        ICALiNGAM,
    )

except ImportError:

    VARLiNGAM = None
    ICALiNGAM = None
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
# 4. Helper for full-VAR lagged regression
# ============================================================

def _build_lagged_design(
    data,
    tau_max,
):
    """
    Build a full lagged design matrix.

    For every current observation X(t), predictors are:

        X(t-1), X(t-2), ..., X(t-tau_max)

    Column order:

        lag 1:
            X0, X1, ..., X(N-1)

        lag 2:
            X0, X1, ..., X(N-1)

        ...

    Returns
    -------
    centered_data : np.ndarray
    X_lagged      : np.ndarray
    Y_current     : np.ndarray
    """

    data = np.asarray(
        data,
        dtype=float,
    )

    T, N = data.shape

    if tau_max < 1:

        raise ValueError(
            "GCresPC requires tau_max >= 1."
        )

    if T <= tau_max:

        raise ValueError(
            "Time series is too short "
            "for the requested tau_max."
        )

    # --------------------------------------------------------
    # The SCM is theoretically zero-mean.
    # We center finite samples explicitly and then fit
    # VAR equations without an intercept.
    # --------------------------------------------------------

    centered_data = (
        data
        -
        np.mean(
            data,
            axis=0,
            keepdims=True,
        )
    )

    # Current values:
    #
    # t = tau_max, ..., T-1
    #
    Y_current = centered_data[
        tau_max:
    ]

    lag_blocks = []

    for tau in range(
        1,
        tau_max + 1,
    ):

        block = centered_data[
            tau_max - tau:
            T - tau
        ]

        lag_blocks.append(
            block
        )

    X_lagged = np.hstack(
        lag_blocks
    )

    return (
        centered_data,
        X_lagged,
        Y_current,
    )


# ============================================================
# 5. Run GCresPC
# ============================================================

def run_gcrespc(
    data,
    var_names,
    tau_max,
    pc_alpha=0.01,
    verbosity=0,
):
    """
    Figure-2-style GCresPC implementation.

    Stage 1
    -------
    Fit a full linear VAR:

        X_j(t)
        =
        sum_tau sum_i
        beta[j, i, tau] X_i(t-tau)
        + u_j(t)

    Each lag-specific beta coefficient is tested.
    Significant coefficients become lagged edges.

    Stage 2
    -------
    Run standard PC with ParCorr on the VAR residuals
    to recover contemporaneous structure.

    Important
    ---------
    GCresPC does NOT use lagged links in the
    contemporaneous orientation phase.

    This is the key distinction emphasized in Runge (2020).
    """

    start_time = (
        time.perf_counter()
    )

    data = np.asarray(
        data,
        dtype=float,
    )

    T, N = data.shape

    if len(var_names) != N:

        raise ValueError(
            "var_names length must match "
            "number of variables."
        )

    # ========================================================
    # A. Full VAR design
    # ========================================================

    (
        centered_data,
        X_lagged,
        Y_current,
    ) = _build_lagged_design(
        data=data,
        tau_max=tau_max,
    )

    n_effective_samples = (
        X_lagged.shape[0]
    )

    n_predictors = (
        X_lagged.shape[1]
    )

    if (
        n_effective_samples
        <= n_predictors
    ):

        warnings.warn(
            "GCresPC full VAR has at least as many "
            "predictors as effective samples. "
            "Coefficient significance tests may "
            "be unstable.",
            RuntimeWarning,
        )

    # ========================================================
    # B. Fit one VAR equation per target variable
    # ========================================================

    # Shape:
    #
    # coefficients[tau-1, target, source]
    #
    var_coefficients = np.zeros(
        (
            tau_max,
            N,
            N,
        ),
        dtype=float,
    )

    var_pvalues = np.ones(
        (
            tau_max,
            N,
            N,
        ),
        dtype=float,
    )

    residuals = np.zeros(
        (
            Y_current.shape[0],
            N,
        ),
        dtype=float,
    )

    for target in range(N):

        model = sm.OLS(
            Y_current[
                :,
                target,
            ],
            X_lagged,
        )

        fit = model.fit()

        residuals[
            :,
            target,
        ] = fit.resid

        for tau in range(
            1,
            tau_max + 1,
        ):

            for source in range(N):

                column = (
                    (tau - 1) * N
                    + source
                )

                coefficient = float(
                    fit.params[
                        column
                    ]
                )

                p_value = float(
                    fit.pvalues[
                        column
                    ]
                )

                var_coefficients[
                    tau - 1,
                    target,
                    source,
                ] = coefficient

                var_pvalues[
                    tau - 1,
                    target,
                    source,
                ] = p_value

    # ========================================================
    # C. Initialize Tigramite-style output matrices
    # ========================================================

    graph = np.full(
        (
            N,
            N,
            tau_max + 1,
        ),
        "",
        dtype="<U3",
    )

    p_matrix = np.ones(
        (
            N,
            N,
            tau_max + 1,
        ),
        dtype=float,
    )

    val_matrix = np.zeros(
        (
            N,
            N,
            tau_max + 1,
        ),
        dtype=float,
    )

    # ========================================================
    # D. Lagged Granger / VAR coefficient edges
    # ========================================================

    for tau in range(
        1,
        tau_max + 1,
    ):

        for source in range(N):

            for target in range(N):

                p_value = (
                    var_pvalues[
                        tau - 1,
                        target,
                        source,
                    ]
                )

                coefficient = (
                    var_coefficients[
                        tau - 1,
                        target,
                        source,
                    ]
                )

                p_matrix[
                    source,
                    target,
                    tau,
                ] = p_value

                val_matrix[
                    source,
                    target,
                    tau,
                ] = coefficient

                if (
                    np.isfinite(
                        p_value
                    )
                    and
                    p_value
                    < pc_alpha
                ):

                    graph[
                        source,
                        target,
                        tau,
                    ] = "-->"

    # ========================================================
    # E. PC on VAR residuals
    # ========================================================

    residual_dataframe = (
        pp.DataFrame(
            data=residuals,
            var_names=var_names,
        )
    )

    residual_parcorr = (
        ParCorr()
    )

    residual_pc = PCMCI(
        dataframe=
            residual_dataframe,

        cond_ind_test=
            residual_parcorr,

        verbosity=
            verbosity,
    )

    residual_pc_results = (
        residual_pc.run_pcalg(
            # Only contemporaneous graph:
            tau_min=0,
            tau_max=0,

            pc_alpha=
                pc_alpha,

            mode=
                "standard",

            contemp_collider_rule=
                "majority",

            conflict_resolution=
                True,

            max_conds_dim=
                None,

            max_combinations=
                None,
        )
    )

    # Copy contemporaneous layer.
    graph[
        :,
        :,
        0,
    ] = (
        residual_pc_results[
            "graph"
        ][
            :,
            :,
            0,
        ]
    )

    p_matrix[
        :,
        :,
        0,
    ] = (
        residual_pc_results[
            "p_matrix"
        ][
            :,
            :,
            0,
        ]
    )

    val_matrix[
        :,
        :,
        0,
    ] = (
        residual_pc_results[
            "val_matrix"
        ][
            :,
            :,
            0,
        ]
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "method":
            "GCresPC",

        "runtime":
            runtime,

        "graph":
            graph,

        "p_matrix":
            p_matrix,

        "val_matrix":
            val_matrix,

        "var_coefficients":
            var_coefficients,

        "var_pvalues":
            var_pvalues,

        "residuals":
            residuals,

        "residual_pc_results":
            residual_pc_results,

        # Helpful reminder for debugging:
        "value_semantics":
            (
                "tau>0: full-VAR coefficient; "
                "tau=0: residual ParCorr statistic"
            ),
    }


# ============================================================
# 6. Run VAR-LiNGAM
# ============================================================

def run_varlingam(
    data,
    var_names,
    tau_max,
    random_state=0,
    coef_atol=1e-12,
):
    """
    Run autoregressive LiNGAM.

    Model:

        X(t)
        =
        B0 X(t)
        +
        B1 X(t-1)
        + ...
        +
        Bp X(t-p)
        +
        e(t)

    Pipeline:

        1. Fit reduced-form VAR
        2. Obtain VAR residuals
        3. ICA-LiNGAM on residuals -> B0
        4. Recover structural lag matrices B_tau
        5. Adaptive-LASSO pruning

    Important:
        LiNGAM assumes linear, independent,
        NON-GAUSSIAN disturbances.

    The main Runge Figure 2 experiment uses Gaussian
    noise, so LiNGAM's key identifiability assumption
    is intentionally violated there.
    """

    if (
        VARLiNGAM is None
        or
        ICALiNGAM is None
    ):

        raise ImportError(
            "LiNGAM is not installed. "
            "Run: python -m pip install lingam"
        )

    start_time = (
        time.perf_counter()
    )

    data = np.asarray(
        data,
        dtype=float,
    )

    T, N = data.shape

    if len(var_names) != N:

        raise ValueError(
            "var_names length must match N."
        )

    # --------------------------------------------------------
    # The theoretical generator has zero mean.
    # Center finite samples before VAR fitting.
    # --------------------------------------------------------

    centered_data = (
        data
        -
        np.mean(
            data,
            axis=0,
            keepdims=True,
        )
    )

    # ========================================================
    # ICA-based contemporaneous LiNGAM
    # ========================================================

    ica_lingam = (
        ICALiNGAM(
            random_state=
                random_state,

            max_iter=
                2000,
        )
    )

    # ========================================================
    # VAR-LiNGAM
    #
    # criterion=None:
    #     use EXACTLY tau_max lags,
    #     rather than selecting a smaller order by BIC.
    #
    # prune=True:
    #     adaptive-LASSO pruning.
    # ========================================================

    model = VARLiNGAM(
        lags=
            tau_max,

        criterion=
            None,

        prune=
            True,

        lingam_model=
            ica_lingam,

        random_state=
            random_state,
    )

    with warnings.catch_warnings(
            record=True
    ) as caught_warnings:

        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            centered_data
        )

    ica_converged = not any(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning in caught_warnings
    )

    B_matrices = np.asarray(
        model.adjacency_matrices_,
        dtype=float,
    )

    # Current maintained implementation returns:
    #
    # B_matrices[0]   = B0
    # B_matrices[1]   = B1
    # ...
    # B_matrices[p]   = Bp
    #
    expected_shape = (
        tau_max + 1,
        N,
        N,
    )

    if (
        B_matrices.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected VARLiNGAM adjacency shape: "
            f"{B_matrices.shape}, "
            f"expected {expected_shape}."
        )

    # ========================================================
    # Convert into our Tigramite-style graph representation
    # ========================================================

    graph = np.full(
        (
            N,
            N,
            tau_max + 1,
        ),
        "",
        dtype="<U3",
    )

    # LiNGAM does not provide the same alpha-based
    # edge p-values as PC / PCMCI+.
    p_matrix = np.full(
        (
            N,
            N,
            tau_max + 1,
        ),
        np.nan,
        dtype=float,
    )

    val_matrix = np.zeros(
        (
            N,
            N,
            tau_max + 1,
        ),
        dtype=float,
    )

    # ========================================================
    # A. Contemporaneous B0
    # ========================================================

    B0 = B_matrices[0]

    for target in range(N):

        for source in range(N):

            if source == target:
                continue

            coefficient = float(
                B0[
                    target,
                    source,
                ]
            )

            if np.isclose(
                coefficient,
                0.0,
                atol=coef_atol,
                rtol=0.0,
            ):
                continue

            # source -> target
            graph[
                source,
                target,
                0,
            ] = "-->"

            # Tigramite-style reverse representation
            graph[
                target,
                source,
                0,
            ] = "<--"

            val_matrix[
                source,
                target,
                0,
            ] = coefficient

            val_matrix[
                target,
                source,
                0,
            ] = coefficient

    # ========================================================
    # B. Lagged structural matrices
    # ========================================================

    for tau in range(
        1,
        tau_max + 1,
    ):

        B_tau = (
            B_matrices[
                tau
            ]
        )

        for target in range(N):

            for source in range(N):

                coefficient = float(
                    B_tau[
                        target,
                        source,
                    ]
                )

                if np.isclose(
                    coefficient,
                    0.0,
                    atol=coef_atol,
                    rtol=0.0,
                ):
                    continue

                graph[
                    source,
                    target,
                    tau,
                ] = "-->"

                val_matrix[
                    source,
                    target,
                    tau,
                ] = coefficient

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "method":
            "LiNGAM",

        "runtime":
            runtime,

        "graph":
            graph,

        "p_matrix":
            p_matrix,

        "val_matrix":
            val_matrix,

        "adjacency_matrices":
            B_matrices,

        "causal_order":
            model.causal_order_,

        "residuals":
            model.residuals_,

        "model":
            model,

        "fpr_control_applicable":
            False,

        "conflict_metric_applicable":
            False,

        "value_semantics":
            "structural LiNGAM coefficient",

        "ica_converged":
            ica_converged,
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
    random_state=0,
):
    """
    Generic interface for all Figure-2 methods.

    Supported:
        PCMCI+
        PC
        GCresPC
        LiNGAM
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

    if method_upper in {
        "GCRESPC",
        "GCRES-PC",
    }:

        return run_gcrespc(
            data=data,
            var_names=var_names,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            verbosity=verbosity,
        )

    if method_upper in {
        "LINGAM",
        "VARLINGAM",
        "VAR-LINGAM",
    }:

        return run_varlingam(
            data=data,
            var_names=var_names,
            tau_max=tau_max,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )