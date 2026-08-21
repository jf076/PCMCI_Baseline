import numpy as np
import pandas as pd

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ==========================================================
# 1. Ground-truth causal graph
# ==========================================================

TRUE_EDGES = {
    ("X", "X", 1),
    ("X", "Y", 2),
    ("Y", "Y", 1),
    ("Y", "Z", 1),
    ("Z", "Z", 1),
}

VAR_NAMES = ["X", "Y", "Z"]


# ==========================================================
# 2. Generate the true causal system
# ==========================================================

def generate_true_data(T, seed, burn_in=200):

    rng = np.random.default_rng(seed)

    total_T = T + burn_in

    data = np.zeros((total_T, 3))

    process_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(total_T, 3)
    )

    for t in range(2, total_T):

        # X_t <- X_(t-1)
        data[t, 0] = (
            0.6 * data[t - 1, 0]
            + process_noise[t, 0]
        )

        # Y_t <- X_(t-2), Y_(t-1)
        data[t, 1] = (
            0.7 * data[t - 2, 0]
            + 0.5 * data[t - 1, 1]
            + process_noise[t, 1]
        )

        # Z_t <- Y_(t-1), Z_(t-1)
        data[t, 2] = (
            0.8 * data[t - 1, 1]
            + 0.4 * data[t - 1, 2]
            + process_noise[t, 2]
        )

    return data[burn_in:]


# ==========================================================
# 3. Add measurement noise
# ==========================================================

def add_measurement_noise(
    true_data,
    noise_scale,
    seed
):

    rng = np.random.default_rng(seed)

    measurement_noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=true_data.shape
    )

    observed_data = (
        true_data
        + measurement_noise
    )

    return observed_data


# ==========================================================
# 4. Run PCMCI+
# ==========================================================

def run_pcmci(data):

    dataframe = pp.DataFrame(
        data=data,
        var_names=VAR_NAMES
    )

    parcorr = ParCorr(
        significance="analytic"
    )

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )

    results = pcmci.run_pcmciplus(
        tau_min=1,
        tau_max=2,
        pc_alpha=0.01
    )

    graph = results["graph"]

    pred_edges = set()

    for tau in range(1, 3):

        for i, source in enumerate(VAR_NAMES):

            for j, target in enumerate(VAR_NAMES):

                if graph[i, j, tau] == "-->":

                    pred_edges.add(
                        (source, target, tau)
                    )

    return pred_edges


# ==========================================================
# 5. Experiment settings
# ==========================================================

T = 300

MEASUREMENT_NOISE_LEVELS = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0
]

N_RUNS = 50


# ==========================================================
# 6. Store edge-level results
# ==========================================================

edge_records = []


# ==========================================================
# 7. Run benchmark
# ==========================================================

for noise_scale in MEASUREMENT_NOISE_LEVELS:

    print("\n======================================")
    print(
        f"Measurement noise scale = "
        f"{noise_scale}"
    )
    print("======================================")

    for seed in range(N_RUNS):

        true_data = generate_true_data(
            T=T,
            seed=seed
        )

        observed_data = add_measurement_noise(
            true_data=true_data,
            noise_scale=noise_scale,
            seed=10000 + seed
        )

        pred_edges = run_pcmci(
            observed_data
        )

        # 对每一条真实边分别检查：
        # 这一次运行中有没有被 PCMCI+ 找回来
        for edge in sorted(TRUE_EDGES):

            recovered = int(
                edge in pred_edges
            )

            edge_records.append(
                {
                    "measurement_noise": noise_scale,
                    "seed": seed,
                    "source": edge[0],
                    "target": edge[1],
                    "lag": edge[2],
                    "recovered": recovered
                }
            )


# ==========================================================
# 8. Convert to DataFrame
# ==========================================================

df_edges = pd.DataFrame(
    edge_records
)


# ==========================================================
# 9. Compute edge recovery rate
# ==========================================================

edge_summary = (
    df_edges
    .groupby(
        [
            "measurement_noise",
            "source",
            "target",
            "lag"
        ]
    )
    .agg(
        recovery_rate=(
            "recovered",
            "mean"
        ),
        recovered_count=(
            "recovered",
            "sum"
        )
    )
    .reset_index()
)


# ==========================================================
# 10. Create readable edge labels
# ==========================================================

edge_summary["edge"] = (
    edge_summary["source"]
    + "(t-"
    + edge_summary["lag"].astype(str)
    + ") -> "
    + edge_summary["target"]
    + "(t)"
)


# ==========================================================
# 11. Print detailed summary
# ==========================================================

print("\n\n======================================================")
print("Edge Recovery Rate")
print("======================================================")

for noise_scale in MEASUREMENT_NOISE_LEVELS:

    print(
        f"\nMeasurement noise = "
        f"{noise_scale}"
    )

    subset = edge_summary[
        edge_summary[
            "measurement_noise"
        ] == noise_scale
    ]

    for _, row in subset.iterrows():

        print(
            f"{row['edge']:20s} | "
            f"Recovered "
            f"{int(row['recovered_count']):2d}"
            f"/{N_RUNS} | "
            f"Rate = "
            f"{row['recovery_rate']:.3f}"
        )


# ==========================================================
# 12. Pivot table
# ==========================================================

pivot_table = (
    edge_summary
    .pivot(
        index="measurement_noise",
        columns="edge",
        values="recovery_rate"
    )
    .reset_index()
)


print("\n\n======================================================")
print("Edge Recovery Matrix")
print("======================================================")

print(
    pivot_table.to_string(
        index=False
    )
)


# ==========================================================
# 13. Save results
# ==========================================================

df_edges.to_csv(
    "pcmci_edge_recovery_raw.csv",
    index=False
)

edge_summary.to_csv(
    "pcmci_edge_recovery_summary.csv",
    index=False
)

pivot_table.to_csv(
    "pcmci_edge_recovery_matrix.csv",
    index=False
)

print("\nResults saved:")
print(
    "  pcmci_edge_recovery_raw.csv"
)
print(
    "  pcmci_edge_recovery_summary.csv"
)
print(
    "  pcmci_edge_recovery_matrix.csv"
)