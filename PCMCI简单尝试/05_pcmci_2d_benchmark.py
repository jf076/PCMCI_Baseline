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
# 2. Generate true causal system
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
# 5. Evaluate graph
# ==========================================================

def evaluate_graph(pred_edges):

    tp_edges = TRUE_EDGES & pred_edges
    fp_edges = pred_edges - TRUE_EDGES
    fn_edges = TRUE_EDGES - pred_edges

    TP = len(tp_edges)
    FP = len(fp_edges)
    FN = len(fn_edges)

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return TP, FP, FN, precision, recall, f1


# ==========================================================
# 6. Benchmark settings
# ==========================================================

T_VALUES = [
    50,
    100,
    200,
    500,
    1000
]

MEASUREMENT_NOISE_LEVELS = [
    0.0,
    0.5,
    1.0,
    2.0,
    4.0
]

N_RUNS = 20

all_results = []


# ==========================================================
# 7. Run 2D benchmark
# ==========================================================

for T in T_VALUES:

    for noise_scale in MEASUREMENT_NOISE_LEVELS:

        print("\n======================================")
        print(
            f"T = {T}, "
            f"Measurement noise = {noise_scale}"
        )
        print("======================================")

        for seed in range(N_RUNS):

            # 真实系统
            true_data = generate_true_data(
                T=T,
                seed=seed
            )

            # 加测量噪声
            observed_data = add_measurement_noise(
                true_data=true_data,
                noise_scale=noise_scale,
                seed=10000 + seed
            )

            # PCMCI+
            pred_edges = run_pcmci(
                observed_data
            )

            (
                TP,
                FP,
                FN,
                precision,
                recall,
                f1
            ) = evaluate_graph(
                pred_edges
            )

            all_results.append(
                {
                    "T": T,
                    "measurement_noise": noise_scale,
                    "seed": seed,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                }
            )

            print(
                f"T={T:4d} | "
                f"noise={noise_scale:4.1f} | "
                f"seed={seed:2d} | "
                f"TP={TP} FP={FP} FN={FN} | "
                f"P={precision:.3f} "
                f"R={recall:.3f} "
                f"F1={f1:.3f}"
            )


# ==========================================================
# 8. Convert to DataFrame
# ==========================================================

df = pd.DataFrame(
    all_results
)


# ==========================================================
# 9. Mean ± Std for each T × noise combination
# ==========================================================

summary = (
    df.groupby(
        [
            "T",
            "measurement_noise"
        ]
    )
    .agg(
        Precision_mean=(
            "Precision",
            "mean"
        ),
        Precision_std=(
            "Precision",
            "std"
        ),

        Recall_mean=(
            "Recall",
            "mean"
        ),
        Recall_std=(
            "Recall",
            "std"
        ),

        F1_mean=(
            "F1",
            "mean"
        ),
        F1_std=(
            "F1",
            "std"
        )
    )
    .reset_index()
)


# ==========================================================
# 10. Print summary
# ==========================================================

print("\n\n======================================================")
print("2D Benchmark Summary: Mean ± Std")
print("======================================================")

for _, row in summary.iterrows():

    print(
        f"T={int(row['T']):4d} | "
        f"Noise={row['measurement_noise']:4.1f} | "
        f"Precision="
        f"{row['Precision_mean']:.3f} ± "
        f"{row['Precision_std']:.3f} | "
        f"Recall="
        f"{row['Recall_mean']:.3f} ± "
        f"{row['Recall_std']:.3f} | "
        f"F1="
        f"{row['F1_mean']:.3f} ± "
        f"{row['F1_std']:.3f}"
    )


# ==========================================================
# 11. F1 matrix
# ==========================================================

f1_matrix = (
    summary
    .pivot(
        index="T",
        columns="measurement_noise",
        values="F1_mean"
    )
)


print("\n\n======================================================")
print("F1 Mean Matrix")
print("Rows = T")
print("Columns = Measurement Noise")
print("======================================================")

print(
    f1_matrix.to_string()
)


# ==========================================================
# 12. Recall matrix
# ==========================================================

recall_matrix = (
    summary
    .pivot(
        index="T",
        columns="measurement_noise",
        values="Recall_mean"
    )
)


print("\n\n======================================================")
print("Recall Mean Matrix")
print("======================================================")

print(
    recall_matrix.to_string()
)


# ==========================================================
# 13. Precision matrix
# ==========================================================

precision_matrix = (
    summary
    .pivot(
        index="T",
        columns="measurement_noise",
        values="Precision_mean"
    )
)


print("\n\n======================================================")
print("Precision Mean Matrix")
print("======================================================")

print(
    precision_matrix.to_string()
)


# ==========================================================
# 14. Save results
# ==========================================================

df.to_csv(
    "pcmci_2d_benchmark_raw.csv",
    index=False
)

summary.to_csv(
    "pcmci_2d_benchmark_summary.csv",
    index=False
)

f1_matrix.to_csv(
    "pcmci_2d_f1_matrix.csv"
)

recall_matrix.to_csv(
    "pcmci_2d_recall_matrix.csv"
)

precision_matrix.to_csv(
    "pcmci_2d_precision_matrix.csv"
)

print("\nResults saved:")
print("  pcmci_2d_benchmark_raw.csv")
print("  pcmci_2d_benchmark_summary.csv")
print("  pcmci_2d_f1_matrix.csv")
print("  pcmci_2d_recall_matrix.csv")
print("  pcmci_2d_precision_matrix.csv")
