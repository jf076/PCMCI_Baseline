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
# 2. Generate synthetic causal time series
# ==========================================================

def generate_data(T, seed, burn_in=200):

    rng = np.random.default_rng(seed)

    total_T = T + burn_in

    data = np.zeros((total_T, 3))

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(total_T, 3)
    )

    for t in range(2, total_T):

        # X_t <- X_(t-1)
        data[t, 0] = (
            0.6 * data[t - 1, 0]
            + noise[t, 0]
        )

        # Y_t <- X_(t-2), Y_(t-1)
        data[t, 1] = (
            0.7 * data[t - 2, 0]
            + 0.5 * data[t - 1, 1]
            + noise[t, 1]
        )

        # Z_t <- Y_(t-1), Z_(t-1)
        data[t, 2] = (
            0.8 * data[t - 1, 1]
            + 0.4 * data[t - 1, 2]
            + noise[t, 2]
        )

    data = data[burn_in:]

    return data


# ==========================================================
# 3. Run PCMCI+
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
# 4. Evaluate graph
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
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return TP, FP, FN, precision, recall, f1


# ==========================================================
# 5. Benchmark settings
# ==========================================================

T_VALUES = [
    50,
    100,
    200,
    500,
    1000
]

N_RUNS = 20

all_results = []


# ==========================================================
# 6. Run benchmark
# ==========================================================

for T in T_VALUES:

    print("\n==============================")
    print(f"Running T = {T}")
    print("==============================")

    for seed in range(N_RUNS):

        data = generate_data(
            T=T,
            seed=seed
        )

        pred_edges = run_pcmci(data)

        TP, FP, FN, precision, recall, f1 = evaluate_graph(
            pred_edges
        )

        all_results.append(
            {
                "T": T,
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
            f"seed={seed:2d} | "
            f"TP={TP} FP={FP} FN={FN} | "
            f"P={precision:.3f} "
            f"R={recall:.3f} "
            f"F1={f1:.3f}"
        )


# ==========================================================
# 7. Convert to DataFrame
# ==========================================================

df = pd.DataFrame(all_results)


# ==========================================================
# 8. Mean and standard deviation
# ==========================================================

summary = (
    df.groupby("T")
    .agg(
        Precision_mean=("Precision", "mean"),
        Precision_std=("Precision", "std"),

        Recall_mean=("Recall", "mean"),
        Recall_std=("Recall", "std"),

        F1_mean=("F1", "mean"),
        F1_std=("F1", "std")
    )
    .reset_index()
)


# ==========================================================
# 9. Print summary
# ==========================================================

print("\n\n==============================================")
print("Benchmark Summary: Mean ± Std")
print("==============================================")

for _, row in summary.iterrows():

    print(
        f"T={int(row['T']):4d} | "
        f"Precision={row['Precision_mean']:.3f} ± "
        f"{row['Precision_std']:.3f} | "
        f"Recall={row['Recall_mean']:.3f} ± "
        f"{row['Recall_std']:.3f} | "
        f"F1={row['F1_mean']:.3f} ± "
        f"{row['F1_std']:.3f}"
    )


# ==========================================================
# 10. Save results
# ==========================================================

df.to_csv(
    "pcmci_benchmark_raw.csv",
    index=False
)

summary.to_csv(
    "pcmci_benchmark_summary.csv",
    index=False
)

print("\nResults saved:")
print("  pcmci_benchmark_raw.csv")
print("  pcmci_benchmark_summary.csv")