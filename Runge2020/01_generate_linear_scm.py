import numpy as np

def build_coefficient_matrices(
    true_edges,
    n_variables,
    tau_max,
):
    """
    Build B_0, ..., B_tau_max for

        X_t = B_0 X_t
              + B_1 X_{t-1}
              + ...
              + B_p X_{t-p}
              + noise

    Matrix convention:
        B_lag[target, source] = coefficient
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

def check_stationarity(
    true_edges,
    n_variables,
    tau_max,
):
    """
    Check stationarity of the linear SCM using
    the spectral radius of the VAR companion matrix.

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
    # Remove contemporaneous effects:
    #
    # (I - B0) X_t
    #     = B1 X_{t-1} + ... + Bp X_{t-p} + noise
    # --------------------------------------------------------

    identity = np.eye(n_variables)

    instantaneous_matrix = (
        identity - B_mats[0]
    )

    A_mats = []

    for lag in range(1, tau_max + 1):

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
    # [A1 A2 ... Ap]
    companion[:n_variables, :] = np.hstack(
        A_mats
    )

    # Lower identity blocks:
    #
    # [I 0 ...]
    # [0 I ...]
    # ...
    if tau_max > 1:

        companion[
            n_variables:,
            :-n_variables,
        ] = np.eye(
            n_variables * (tau_max - 1)
        )

    eigenvalues = np.linalg.eigvals(
        companion
    )

    spectral_radius = float(
        np.max(np.abs(eigenvalues))
    )

    is_stationary = (
        spectral_radius < 1.0
    )

    return (
        is_stationary,
        spectral_radius,
        B_mats,
    )


def generate_candidate_structure(
    rng,
    n_variables,
    a_max,
    tau_max,
    contemp_fraction=0.3,
):
    """
    Generate one candidate linear SCM structure.

    This function generates:
    1. auto-dependencies
    2. contemporaneous DAG cross-links
    3. lagged cross-links

    It does NOT check stationarity
    and does NOT simulate time series data.
    """

    # ========================================================
    # 1. Number of cross-links
    # ========================================================

    if n_variables == 2:
        n_cross_links = 1
    else:
        n_cross_links = int(
            np.floor(1.5 * n_variables)
        )

    # Our current clear-reproduction choice:
    # approximately 30% contemporaneous.
    n_contemp = int(
        round(
            contemp_fraction
            * n_cross_links
        )
    )

    n_lagged_cross = (
        n_cross_links - n_contemp
    )

    # ========================================================
    # 2. Autoregressive coefficients
    # ========================================================

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

    for j in range(n_variables):

        true_edges.append(
            {
                "source": j,
                "target": j,
                "lag": 1,
                "coefficient": float(
                    auto_coeffs[j]
                ),
                "edge_type": "auto",
            }
        )

    # ========================================================
    # 3. Contemporaneous DAG
    # ========================================================

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
                (source, target)
            )

    selected_indices = rng.choice(
        len(contemp_candidates),
        size=n_contemp,
        replace=False,
    )

    contemp_edges = [
        contemp_candidates[int(idx)]
        for idx in selected_indices
    ]

    for source, target in contemp_edges:

        magnitude = rng.uniform(
            0.1,
            0.5,
        )

        sign = rng.choice(
            [-1.0, 1.0]
        )

        coefficient = (
            float(sign)
            * float(magnitude)
        )

        true_edges.append(
            {
                "source": source,
                "target": target,
                "lag": 0,
                "coefficient": coefficient,
                "edge_type":
                    "contemporaneous",
            }
        )

    # ========================================================
    # 4. Lagged cross-links
    # ========================================================

    lagged_candidates = []

    for source in range(n_variables):

        for target in range(n_variables):

            if source == target:
                continue

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

    selected_lagged_indices = (
        rng.choice(
            len(lagged_candidates),
            size=n_lagged_cross,
            replace=False,
        )
    )

    lagged_edges = [
        lagged_candidates[int(idx)]
        for idx
        in selected_lagged_indices
    ]

    for (
        source,
        target,
        lag,
    ) in lagged_edges:

        magnitude = rng.uniform(
            0.1,
            0.5,
        )

        sign = rng.choice(
            [-1.0, 1.0]
        )

        coefficient = (
            float(sign)
            * float(magnitude)
        )

        true_edges.append(
            {
                "source": source,
                "target": target,
                "lag": lag,
                "coefficient": coefficient,
                "edge_type":
                    "lagged_cross",
            }
        )

    # ========================================================
    # 5. Model metadata
    # ========================================================

    model_info = {
        "auto_coeffs": auto_coeffs,
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

# ============================================================
# Runge (2020) linear Gaussian simulation: default parameters
# ============================================================

N = 5
T = 500
A_MAX = 0.95
TAU_TRUE_MAX = 5
SEED = 0

# Number of cross-links in Runge (2020):
# L = floor(1.5 * N)
N_CROSS_LINKS = int(np.floor(1.5 * N))

print("N =", N)
print("T =", T)
print("a =", A_MAX)
print("true max lag =", TAU_TRUE_MAX)
print("number of cross-links =", N_CROSS_LINKS)

rng = np.random.default_rng(SEED)

a_low = max(0.0, A_MAX - 0.3)

auto_coeffs = rng.uniform(
    low=a_low,
    high=A_MAX,
    size=N
)

print("\nAutoregressive coefficients:")
for j, coeff in enumerate(auto_coeffs):
    print(f"X{j}: {coeff:.4f}")

true_edges = []

for j in range(N):
    true_edges.append(
        {
            "source": j,
            "target": j,
            "lag": 1,
            "coefficient": auto_coeffs[j],
            "edge_type": "auto",
        }
    )

print("\nAuto-dependency ground truth:")

for edge in true_edges:
    print(
        f"X{edge['source']}(t-{edge['lag']}) "
        f"-> X{edge['target']}(t), "
        f"coef={edge['coefficient']:.4f}"
    )

N_CONTEMP = int(round(0.3 * N_CROSS_LINKS))
N_LAGGED_CROSS = N_CROSS_LINKS - N_CONTEMP

print("\nCross-link allocation:")
print("contemporaneous =", N_CONTEMP)
print("lagged cross-links =", N_LAGGED_CROSS)

topological_order = rng.permutation(N)

print("\nContemporaneous topological order:")
print(topological_order)

contemp_candidates = []

for pos_i in range(N):
    for pos_j in range(pos_i + 1, N):

        source = int(topological_order[pos_i])
        target = int(topological_order[pos_j])

        contemp_candidates.append(
            (source, target)
        )

print("\nPossible contemporaneous DAG edges:")
print(contemp_candidates)

selected_indices = rng.choice(
    len(contemp_candidates),
    size=N_CONTEMP,
    replace=False
)

contemp_edges = [
    contemp_candidates[idx]
    for idx in selected_indices
]

print("\nSelected contemporaneous edges:")

for source, target in contemp_edges:
    print(f"X{source}(t) -> X{target}(t)")

for source, target in contemp_edges:

    magnitude = rng.uniform(0.1, 0.5)
    sign = rng.choice([-1.0, 1.0])

    coefficient = sign * magnitude

    true_edges.append(
        {
            "source": source,
            "target": target,
            "lag": 0,
            "coefficient": coefficient,
            "edge_type": "contemporaneous",
        }
    )

print("\nCurrent Ground Truth:")
print("-" * 60)

for edge in true_edges:

    source = edge["source"]
    target = edge["target"]
    lag = edge["lag"]
    coef = edge["coefficient"]
    edge_type = edge["edge_type"]

    if lag == 0:
        relation = f"X{source}(t) -> X{target}(t)"
    else:
        relation = f"X{source}(t-{lag}) -> X{target}(t)"

    print(
        f"{relation:<22} "
        f"coef={coef:+.4f} "
        f"type={edge_type}"
    )

# ============================================================
# Generate lagged cross-links
# ============================================================

lagged_candidates = []

for source in range(N):
    for target in range(N):

        # Cross-link means source and target must be different
        if source == target:
            continue

        for lag in range(1, TAU_TRUE_MAX + 1):
            lagged_candidates.append(
                (source, target, lag)
            )

print("\nNumber of possible lagged cross-links:")
print(len(lagged_candidates))

selected_lagged_indices = rng.choice(
    len(lagged_candidates),
    size=N_LAGGED_CROSS,
    replace=False
)

lagged_edges = [
    lagged_candidates[idx]
    for idx in selected_lagged_indices
]
print("\nSelected lagged cross-links:")

for source, target, lag in lagged_edges:
    print(
        f"X{source}(t-{lag}) -> X{target}(t)"
    )

for source, target, lag in lagged_edges:

    magnitude = rng.uniform(0.1, 0.5)
    sign = rng.choice([-1.0, 1.0])

    coefficient = sign * magnitude

    true_edges.append(
        {
            "source": source,
            "target": target,
            "lag": lag,
            "coefficient": coefficient,
            "edge_type": "lagged_cross",
        }
    )

print("\nComplete Ground Truth:")
print("-" * 70)

for edge in true_edges:

    source = edge["source"]
    target = edge["target"]
    lag = edge["lag"]
    coef = edge["coefficient"]
    edge_type = edge["edge_type"]

    if lag == 0:
        relation = f"X{source}(t) -> X{target}(t)"
    else:
        relation = f"X{source}(t-{lag}) -> X{target}(t)"

    print(
        f"{relation:<24}"
        f"coef={coef:+.4f}   "
        f"type={edge_type}"
    )

n_auto = sum(
    edge["edge_type"] == "auto"
    for edge in true_edges
)

n_contemp = sum(
    edge["edge_type"] == "contemporaneous"
    for edge in true_edges
)

n_lagged = sum(
    edge["edge_type"] == "lagged_cross"
    for edge in true_edges
)

print("\nGround Truth summary:")
print("auto-dependencies      =", n_auto)
print("contemporaneous links  =", n_contemp)
print("lagged cross-links     =", n_lagged)
print("total links            =", len(true_edges))

# ============================================================
# Sanity checks
# ============================================================

assert n_auto == N

assert n_contemp == N_CONTEMP

assert n_lagged == N_LAGGED_CROSS

assert len(true_edges) == (
    N + N_CONTEMP + N_LAGGED_CROSS
)

print("\nSanity checks passed.")

for edge in true_edges:

    if edge["edge_type"] == "lagged_cross":

        assert edge["source"] != edge["target"]

        assert 1 <= edge["lag"] <= TAU_TRUE_MAX

for edge in true_edges:

    if edge["edge_type"] == "contemporaneous":

        assert edge["source"] != edge["target"]

        assert edge["lag"] == 0

for edge in true_edges:

    if edge["edge_type"] == "auto":

        assert edge["source"] == edge["target"]

        assert edge["lag"] == 1

print("Edge-type checks passed.")

# ============================================================
# Build coefficient matrices B_0, ..., B_tau_max
# ============================================================

is_stationary, spectral_radius, B_mats = (
    check_stationarity(
        true_edges=true_edges,
        n_variables=N,
        tau_max=TAU_TRUE_MAX,
    )
)

for lag, B in enumerate(B_mats):

    print(f"\nB_{lag}:")
    print(np.round(B, 4))

print("\nStationarity check:")
print(
    f"spectral radius = "
    f"{spectral_radius:.6f}"
)
print(
    "stationary =",
    is_stationary,
)

if not is_stationary:

    print(
        "\nCandidate model rejected "
        "because it is non-stationary."
    )

    raise SystemExit

# ============================================================
# Noise parameters
# ============================================================

noise_stds = rng.uniform(
    low=0.5,
    high=2.0,
    size=N
)

print("\nNoise standard deviations:")

for j, sigma in enumerate(noise_stds):
    print(f"X{j}: sigma={sigma:.4f}")

BURN_IN = 500

TOTAL_T = T + BURN_IN

print("\nSimulation length:")
print("burn-in =", BURN_IN)
print("kept samples =", T)
print("total simulated =", TOTAL_T)

data_full = np.zeros(
    (TOTAL_T, N),
    dtype=float
)
print("\nInitial data array shape:")
print(data_full.shape)

# ============================================================
# Simulate linear Gaussian time series
# ============================================================

for t in range(TAU_TRUE_MAX, TOTAL_T):

    # Important:
    # contemporaneous variables must be generated
    # according to the DAG topological order.
    for j in topological_order:

        j = int(j)

        # ----------------------------------------------------
        # 1. Auto-dependency term
        # ----------------------------------------------------
        auto_term = (
            auto_coeffs[j]
            * data_full[t - 1, j]
        )

        # ----------------------------------------------------
        # 2. Lagged cross-link term
        # ----------------------------------------------------
        lagged_term = 0.0

        for edge in true_edges:

            if edge["edge_type"] != "lagged_cross":
                continue

            if edge["target"] != j:
                continue

            source = edge["source"]
            lag = edge["lag"]
            coefficient = edge["coefficient"]

            lagged_term += (
                coefficient
                * data_full[t - lag, source]
            )

        # ----------------------------------------------------
        # 3. Contemporaneous term
        # ----------------------------------------------------
        contemp_term = 0.0

        for edge in true_edges:

            if edge["edge_type"] != "contemporaneous":
                continue

            if edge["target"] != j:
                continue

            source = edge["source"]
            coefficient = edge["coefficient"]

            contemp_term += (
                coefficient
                * data_full[t, source]
            )

        # ----------------------------------------------------
        # 4. Gaussian dynamical noise
        # ----------------------------------------------------
        noise_term = rng.normal(
            loc=0.0,
            scale=noise_stds[j]
        )

        # ----------------------------------------------------
        # Structural equation
        # ----------------------------------------------------
        data_full[t, j] = (
            auto_term
            + lagged_term
            + contemp_term
            + noise_term
        )

data = data_full[BURN_IN:]
print("\nGenerated time series:")
print("full shape =", data_full.shape)
print("final shape =", data.shape)

print("\nFirst 5 observations:")
print(data[:5])
print("\nColumn means:")
print(np.mean(data, axis=0))

print("\nColumn standard deviations:")
print(np.std(data, axis=0))

# ============================================================
# NAN以及inf检查
# ============================================================
assert data.shape == (T, N)

assert np.all(np.isfinite(data))

print("\nData sanity checks passed.")

max_abs_value = np.max(np.abs(data))

print("\nMaximum absolute value:")
print(max_abs_value)

if max_abs_value > 1e6:
    print("WARNING: possible unstable simulation.")