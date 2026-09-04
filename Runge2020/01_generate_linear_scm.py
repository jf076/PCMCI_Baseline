import numpy as np


# ============================================================
# 1. Build coefficient matrices
# ============================================================

def build_coefficient_matrices(
    true_edges,
    n_variables,
    tau_max,
):
    """
    Build B_0, B_1, ..., B_tau_max for the linear SCM:

        X_t
        = B_0 X_t
        + B_1 X_{t-1}
        + ...
        + B_p X_{t-p}
        + noise

    Matrix convention:
        B_lag[target, source] = coefficient

    Example:
        X2(t-3) -> X4(t), coefficient = 0.3

    corresponds to:
        B_3[4, 2] = 0.3
    """

    B_mats = [
        np.zeros(
            (n_variables, n_variables),
            dtype=float,
        )
        for _ in range(tau_max + 1)
    ]

    for edge in true_edges:

        source = edge["source"]
        target = edge["target"]
        lag = edge["lag"]
        coefficient = edge["coefficient"]

        B_mats[lag][target, source] += coefficient

    return B_mats


# ============================================================
# 2. Stationarity check
# ============================================================

def check_stationarity(
    true_edges,
    n_variables,
    tau_max,
):
    """
    Check stationarity of the linear SCM.

    Original model:

        X_t
        = B_0 X_t
        + B_1 X_{t-1}
        + ...
        + B_p X_{t-p}
        + noise

    Rearranging:

        (I - B_0) X_t
        = B_1 X_{t-1}
        + ...
        + B_p X_{t-p}
        + noise

    Therefore:

        X_t
        = A_1 X_{t-1}
        + ...
        + A_p X_{t-p}
        + transformed_noise

    where:

        A_lag = (I - B_0)^(-1) B_lag

    Then construct the VAR(p) companion matrix.

    Stationarity criterion:

        spectral_radius < 1

    Returns
    -------
    is_stationary : bool
    spectral_radius : float
    B_mats : list[np.ndarray]
    """

    B_mats = build_coefficient_matrices(
        true_edges=true_edges,
        n_variables=n_variables,
        tau_max=tau_max,
    )

    # --------------------------------------------------------
    # Remove contemporaneous effects
    # --------------------------------------------------------

    identity = np.eye(n_variables)

    instantaneous_matrix = (
        identity - B_mats[0]
    )

    A_mats = []

    for lag in range(1, tau_max + 1):

        # Equivalent to:
        #
        # inv(I - B0) @ B_lag
        #
        # but solve() is numerically preferable.
        A_lag = np.linalg.solve(
            instantaneous_matrix,
            B_mats[lag],
        )

        A_mats.append(A_lag)

    # --------------------------------------------------------
    # Build VAR(p) companion matrix
    # --------------------------------------------------------

    companion = np.zeros(
        (
            n_variables * tau_max,
            n_variables * tau_max,
        ),
        dtype=float,
    )

    # First block row:
    #
    # [A1  A2  ...  Ap]
    companion[:n_variables, :] = np.hstack(
        A_mats
    )

    # Lower identity blocks:
    #
    # [I 0 0 ...]
    # [0 I 0 ...]
    # [0 0 I ...]
    #
    if tau_max > 1:

        companion[
            n_variables:,
            :-n_variables,
        ] = np.eye(
            n_variables * (tau_max - 1)
        )

    # --------------------------------------------------------
    # Spectral radius
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvals(
        companion
    )

    spectral_radius = float(
        np.max(
            np.abs(eigenvalues)
        )
    )

    is_stationary = (
        spectral_radius < 1.0
    )

    return (
        is_stationary,
        spectral_radius,
        B_mats,
    )


# ============================================================
# 3. Generate one candidate SCM structure
# ============================================================

def generate_candidate_structure(
    rng,
    n_variables,
    a_max,
    tau_max,
    contemp_fraction=0.3,
):
    """
    Generate ONE candidate linear SCM structure.

    This function generates:

    1. Auto-dependencies
    2. Contemporaneous cross-links
    3. Lagged cross-links

    It does NOT:
    - check stationarity
    - generate noise
    - simulate time-series observations

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    n_variables : int
        Number of variables N.

    a_max : float
        Upper bound of autoregressive coefficients.

    tau_max : int
        Maximum true time lag.

    contemp_fraction : float
        Fraction of cross-links treated as contemporaneous.
        Current clear-reproduction choice: 0.3.

    Returns
    -------
    true_edges : list[dict]
        Ground-truth causal edges.

    model_info : dict
        Metadata describing the generated SCM.
    """

    # ========================================================
    # 3.1 Number of cross-links
    # ========================================================

    # Runge (2020):
    #
    # L = floor(1.5 * N)
    #
    # except N = 2, where L = 1.
    if n_variables == 2:

        n_cross_links = 1

    else:

        n_cross_links = int(
            np.floor(
                1.5 * n_variables
            )
        )

    # --------------------------------------------------------
    # Current implementation choice:
    #
    # approximately 30% of cross-links are contemporaneous.
    #
    # For N=5:
    #
    # L = floor(1.5 * 5) = 7
    #
    # round(0.3 * 7) = 2 contemporaneous
    # remaining 5 are lagged.
    #
    # Paper states "30%" but does not specify in the main text
    # whether this is implemented as an exact rounded count
    # or Bernoulli sampling edge-by-edge.
    # --------------------------------------------------------

    n_contemp = int(
        round(
            contemp_fraction
            * n_cross_links
        )
    )

    n_lagged_cross = (
        n_cross_links
        - n_contemp
    )

    # ========================================================
    # 3.2 Autoregressive coefficients
    # ========================================================

    # Runge (2020):
    #
    # a_j ~ Uniform(
    #     max(0, a - 0.3),
    #     a
    # )
    #
    a_low = max(
        0.0,
        a_max - 0.3,
    )

    auto_coeffs = rng.uniform(
        low=a_low,
        high=a_max,
        size=n_variables,
    )

    true_edges = []

    # Every variable receives:
    #
    # X_j(t-1) -> X_j(t)
    #
    for j in range(n_variables):

        coefficient = float(
            auto_coeffs[j]
        )

        # coefficient = 0 means there is no
        # causal autodependency edge.
        if np.isclose(
                coefficient,
                0.0,
        ):
            continue

        true_edges.append(
            {
                "source": j,
                "target": j,
                "lag": 1,
                "coefficient":
                    coefficient,
                "edge_type": "auto",
            }
        )

    # ========================================================
    # 3.3 Contemporaneous DAG
    # ========================================================

    # Random topological order.
    #
    # Only allow edges from an earlier node
    # to a later node in this order.
    #
    # This guarantees that the contemporaneous
    # graph is acyclic.
    topological_order = rng.permutation(
        n_variables
    )

    contemp_candidates = []

    for pos_i in range(n_variables):

        for pos_j in range(
            pos_i + 1,
            n_variables,
        ):

            source = int(
                topological_order[pos_i]
            )

            target = int(
                topological_order[pos_j]
            )

            contemp_candidates.append(
                (
                    source,
                    target,
                )
            )

    # Randomly choose the required number
    # of contemporaneous cross-links.
    selected_contemp_indices = rng.choice(
        len(contemp_candidates),
        size=n_contemp,
        replace=False,
    )

    contemp_edges = [
        contemp_candidates[int(idx)]
        for idx
        in selected_contemp_indices
    ]

    # Assign coefficients:
    #
    # c ~ +/- Uniform(0.1, 0.5)
    #
    for source, target in contemp_edges:

        magnitude = rng.uniform(
            low=0.1,
            high=0.5,
        )

        sign = rng.choice(
            [-1.0, 1.0]
        )

        coefficient = float(
            sign * magnitude
        )

        true_edges.append(
            {
                "source": source,
                "target": target,
                "lag": 0,
                "coefficient": coefficient,
                "edge_type": "contemporaneous",
            }
        )

    # ========================================================
    # 3.4 Lagged cross-links
    # ========================================================

    lagged_candidates = []

    for source in range(n_variables):

        for target in range(n_variables):

            # Cross-link:
            #
            # source != target
            #
            if source == target:
                continue

            # lag = 1, ..., tau_max
            #
            for lag in range(
                1,
                tau_max + 1,
            ):

                lagged_candidates.append(
                    (
                        source,
                        target,
                        lag,
                    )
                )

    # Draw lag-specific links without replacement.
    #
    # Therefore an identical:
    #
    # (source, target, lag)
    #
    # cannot occur twice.
    selected_lagged_indices = rng.choice(
        len(lagged_candidates),
        size=n_lagged_cross,
        replace=False,
    )

    lagged_edges = [
        lagged_candidates[int(idx)]
        for idx
        in selected_lagged_indices
    ]

    # Assign coefficients
    for (
        source,
        target,
        lag,
    ) in lagged_edges:

        magnitude = rng.uniform(
            low=0.1,
            high=0.5,
        )

        sign = rng.choice(
            [-1.0, 1.0]
        )

        coefficient = float(
            sign * magnitude
        )

        true_edges.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "coefficient": coefficient,
                "edge_type": "lagged_cross",
            }
        )

    # ========================================================
    # 3.5 Save model information
    # ========================================================

    model_info = {
        "auto_coeffs":
            auto_coeffs,

        "topological_order":
            topological_order,

        "n_cross_links":
            n_cross_links,

        "n_contemporaneous":
            n_contemp,

        "n_lagged_cross":
            n_lagged_cross,
    }

    return (
        true_edges,
        model_info,
    )

def simulate_linear_scm(
    rng,
    true_edges,
    topological_order,
    n_variables,
    n_samples,
    tau_max,
    burn_in=500,
):
    """
    Simulate observations from an accepted stationary
    linear Gaussian SCM.

    Model:

        X_j(t)
        = sum(auto effects)
        + sum(lagged cross effects)
        + sum(contemporaneous effects)
        + eta_j(t)

    with:

        eta_j(t) ~ N(0, sigma_j^2)

    and:

        sigma_j ~ Uniform(0.5, 2.0)

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    true_edges : list[dict]
        Ground-truth causal edges.

    topological_order : array-like
        Topological order of contemporaneous DAG.

    n_variables : int
        Number of variables.

    n_samples : int
        Number of observations kept after burn-in.

    tau_max : int
        Maximum true lag.

    burn_in : int
        Number of initial simulated observations discarded.

    Returns
    -------
    data : np.ndarray
        Final observed time series with shape
        (n_samples, n_variables).

    noise_stds : np.ndarray
        Gaussian noise standard deviation for each variable.
    """

    # ========================================================
    # 1. Noise standard deviations
    # ========================================================

    noise_stds = rng.uniform(
        low=0.5,
        high=2.0,
        size=n_variables,
    )

    # ========================================================
    # 2. Allocate simulation array
    # ========================================================

    total_t = (
        burn_in
        + n_samples
    )

    data_full = np.zeros(
        (
            total_t,
            n_variables,
        ),
        dtype=float,
    )

    # ========================================================
    # 3. Split edges by type
    # ========================================================

    auto_edges = [
        edge
        for edge in true_edges
        if edge["edge_type"] == "auto"
    ]

    lagged_edges = [
        edge
        for edge in true_edges
        if edge["edge_type"] == "lagged_cross"
    ]

    contemp_edges = [
        edge
        for edge in true_edges
        if edge["edge_type"] == "contemporaneous"
    ]

    # ========================================================
    # 4. Simulate time series
    # ========================================================

    for t in range(
        tau_max,
        total_t,
    ):

        # Contemporaneous variables must be generated
        # according to the DAG topological order.
        for j_raw in topological_order:

            j = int(j_raw)

            # ------------------------------------------------
            # Auto-dependency contribution
            # ------------------------------------------------

            auto_term = 0.0

            for edge in auto_edges:

                if edge["target"] != j:
                    continue

                source = edge["source"]
                lag = edge["lag"]
                coefficient = edge["coefficient"]

                auto_term += (
                    coefficient
                    * data_full[
                        t - lag,
                        source,
                    ]
                )

            # ------------------------------------------------
            # Lagged cross-link contribution
            # ------------------------------------------------

            lagged_term = 0.0

            for edge in lagged_edges:

                if edge["target"] != j:
                    continue

                source = edge["source"]
                lag = edge["lag"]
                coefficient = edge["coefficient"]

                lagged_term += (
                    coefficient
                    * data_full[
                        t - lag,
                        source,
                    ]
                )

            # ------------------------------------------------
            # Contemporaneous contribution
            # ------------------------------------------------

            contemp_term = 0.0

            for edge in contemp_edges:

                if edge["target"] != j:
                    continue

                source = edge["source"]
                coefficient = edge["coefficient"]

                contemp_term += (
                    coefficient
                    * data_full[
                        t,
                        source,
                    ]
                )

            # ------------------------------------------------
            # Gaussian dynamical noise
            # ------------------------------------------------

            noise_term = rng.normal(
                loc=0.0,
                scale=noise_stds[j],
            )

            # ------------------------------------------------
            # Structural equation
            # ------------------------------------------------

            data_full[t, j] = (
                auto_term
                + lagged_term
                + contemp_term
                + noise_term
            )

    # ========================================================
    # 5. Discard burn-in
    # ========================================================

    data = data_full[
        burn_in:
    ].copy()

    return (
        data,
        noise_stds,
    )

def generate_linear_scm(
    n_variables=5,
    n_samples=500,
    a_max=0.95,
    tau_max=5,
    seed=0,
    burn_in=500,
    contemp_fraction=0.3,
    max_attempts=1000,
):
    """
    Generate one stationary linear Gaussian SCM dataset.

    Workflow
    --------
    1. Initialize random number generator.
    2. Generate candidate SCM structure.
    3. Check stationarity.
    4. Reject non-stationary candidates.
    5. Simulate Gaussian time series from the first
       accepted stationary SCM.

    Returns
    -------
    data : np.ndarray
        Observational time series.
        Shape: (n_samples, n_variables)

    true_edges : list[dict]
        Ground-truth causal edges.

    model_info : dict
        Information about the accepted SCM, including:
        - autoregressive coefficients
        - topological order
        - noise standard deviations
        - spectral radius
        - number of generation attempts
        - model parameters
    """

    # ========================================================
    # 1. Initialize random number generator ONCE
    # ========================================================

    rng = np.random.default_rng(seed)

    # ========================================================
    # 2. Rejection sampling
    # ========================================================

    accepted = False

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        true_edges, model_info = (
            generate_candidate_structure(
                rng=rng,
                n_variables=n_variables,
                a_max=a_max,
                tau_max=tau_max,
                contemp_fraction=contemp_fraction,
            )
        )

        (
            is_stationary,
            spectral_radius,
            B_mats,
        ) = check_stationarity(
            true_edges=true_edges,
            n_variables=n_variables,
            tau_max=tau_max,
        )

        if is_stationary:
            accepted = True
            break

    # ========================================================
    # 3. Make sure a stationary model was found
    # ========================================================

    if not accepted:

        raise RuntimeError(
            "Failed to generate a stationary SCM "
            f"within {max_attempts} attempts."
        )

    # ========================================================
    # 4. Simulate observations
    # ========================================================

    data, noise_stds = simulate_linear_scm(
        rng=rng,
        true_edges=true_edges,
        topological_order=model_info[
            "topological_order"
        ],
        n_variables=n_variables,
        n_samples=n_samples,
        tau_max=tau_max,
        burn_in=burn_in,
    )

    # ========================================================
    # 5. Add useful metadata
    # ========================================================

    model_info["noise_stds"] = noise_stds

    model_info["spectral_radius"] = (
        spectral_radius
    )

    model_info["attempts"] = attempt

    model_info["seed"] = seed

    model_info["n_variables"] = (
        n_variables
    )

    model_info["n_samples"] = (
        n_samples
    )

    model_info["a_max"] = (
        a_max
    )

    model_info["tau_max"] = (
        tau_max
    )

    model_info["burn_in"] = (
        burn_in
    )

    model_info["B_mats"] = (
        B_mats
    )

    # ========================================================
    # 6. Final sanity checks
    # ========================================================

    assert data.shape == (
        n_samples,
        n_variables,
    )

    assert np.all(
        np.isfinite(data)
    )

    assert spectral_radius < 1.0

    # ========================================================
    # 7. Return everything needed by later experiments
    # ========================================================

    return (
        data,
        true_edges,
        model_info,
    )

# ============================================================
# 4. Helper function for printing Ground Truth
# ============================================================

def print_ground_truth(
    true_edges,
):
    """
    Pretty-print the Ground Truth causal edges.
    """

    print(
        "\nAccepted Ground Truth:"
    )

    print("-" * 75)

    for edge in true_edges:

        source = edge["source"]
        target = edge["target"]
        lag = edge["lag"]
        coefficient = edge["coefficient"]
        edge_type = edge["edge_type"]

        if lag == 0:

            relation = (
                f"X{source}(t) "
                f"-> X{target}(t)"
            )

        else:

            relation = (
                f"X{source}(t-{lag}) "
                f"-> X{target}(t)"
            )

        print(
            f"{relation:<25}"
            f"coef={coefficient:+.4f}   "
            f"type={edge_type}"
        )


# ============================================================
# 5. Main program
# ============================================================

def main():

    # ========================================================
    # Default Runge (2020) linear Gaussian setup
    # ========================================================

    N = 5
    T = 500
    A_MAX = 0.95
    TAU_TRUE_MAX = 5
    SEED = 0
    BURN_IN = 500

    print(
        "Runge (2020) linear Gaussian SCM generator"
    )

    print("=" * 60)

    # ========================================================
    # Generate one complete dataset
    # ========================================================

    (
        data,
        true_edges,
        model_info,
    ) = generate_linear_scm(
        n_variables=N,
        n_samples=T,
        a_max=A_MAX,
        tau_max=TAU_TRUE_MAX,
        seed=SEED,
        burn_in=BURN_IN,
    )

    # ========================================================
    # Model information
    # ========================================================

    print(
        "\nAccepted stationary model:"
    )

    print(
        "seed =",
        model_info["seed"],
    )

    print(
        "attempt =",
        model_info["attempts"],
    )

    print(
        "spectral radius = "
        f"{model_info['spectral_radius']:.6f}"
    )

    print(
        "\nTopological order:"
    )

    print(
        model_info[
            "topological_order"
        ]
    )

    # ========================================================
    # Ground Truth
    # ========================================================

    print_ground_truth(
        true_edges
    )

    n_auto = sum(
        edge["edge_type"] == "auto"
        for edge in true_edges
    )

    n_contemp = sum(
        edge["edge_type"]
        == "contemporaneous"
        for edge in true_edges
    )

    n_lagged = sum(
        edge["edge_type"]
        == "lagged_cross"
        for edge in true_edges
    )

    print(
        "\nGround Truth summary:"
    )

    print(
        "auto-dependencies     =",
        n_auto,
    )

    print(
        "contemporaneous links =",
        n_contemp,
    )

    print(
        "lagged cross-links    =",
        n_lagged,
    )

    print(
        "total links           =",
        len(true_edges),
    )

    # ========================================================
    # Noise
    # ========================================================

    print(
        "\nNoise standard deviations:"
    )

    for j, sigma in enumerate(
        model_info["noise_stds"]
    ):

        print(
            f"X{j}: "
            f"sigma={sigma:.4f}"
        )

    # ========================================================
    # Data diagnostics
    # ========================================================

    print(
        "\nGenerated time series:"
    )

    print(
        "data shape =",
        data.shape,
    )

    print(
        "\nFirst 5 observations:"
    )

    print(
        np.round(
            data[:5],
            4,
        )
    )

    print(
        "\nColumn means:"
    )

    print(
        np.round(
            np.mean(
                data,
                axis=0,
            ),
            4,
        )
    )

    print(
        "\nColumn standard deviations:"
    )

    print(
        np.round(
            np.std(
                data,
                axis=0,
            ),
            4,
        )
    )

    max_abs_value = float(
        np.max(
            np.abs(data)
        )
    )

    print(
        "\nMaximum absolute value:"
    )

    print(
        f"{max_abs_value:.4f}"
    )

    # ========================================================
    # Final checks
    # ========================================================

    assert data.shape == (
        T,
        N,
    )

    assert np.all(
        np.isfinite(data)
    )

    assert (
        model_info[
            "spectral_radius"
        ]
        < 1.0
    )

    print(
        "\nAll checks passed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()