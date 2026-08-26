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
# 2. Generate the latent / true causal system
# ==========================================================

def generate_true_data(T, seed, burn_in=200):

    rng = np.random.default_rng(seed)

    total_T = T + burn_in

    data = np.zeros((total_T, 3))

    # 系统本身的过程噪声 / innovation noise
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

T = 300

MEASUREMENT_NOISE_LEVELS = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0
]

N_RUNS = 20

all_results = []


# ==========================================================
# 7. Run measurement-noise benchmark
# ==========================================================

for noise_scale in MEASUREMENT_NOISE_LEVELS:

    print("\n======================================")
    print(
        f"Measurement noise scale = "
        f"{noise_scale}"
    )
    print("======================================")

    for seed in range(N_RUNS):

        # Step 1:
        # 先生成真实系统
        true_data = generate_true_data(
            T=T,
            seed=seed
        )

        # Step 2:
        # 再在观测阶段加入额外测量噪声
        observed_data = add_measurement_noise(
            true_data=true_data,
            noise_scale=noise_scale,

            # 与系统生成 seed 分开，
            # 避免完全使用同一随机序列
            seed=10000 + seed
        )

        # Step 3:
        # PCMCI+ 只能看到 observed_data
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
            f"noise={noise_scale:4.2f} | "
            f"seed={seed:2d} | "
            f"TP={TP} FP={FP} FN={FN} | "
            f"P={precision:.3f} "
            f"R={recall:.3f} "
            f"F1={f1:.3f}"
        )


# ==========================================================
# 8. Raw results
# ==========================================================

df = pd.DataFrame(
    all_results
)


# ==========================================================
# 9. Mean ± Std
# ==========================================================

summary = (
    df.groupby(
        "measurement_noise"
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
print("Measurement Noise Robustness: Mean ± Std")
print("======================================================")

for _, row in summary.iterrows():

    print(
        f"Noise={row['measurement_noise']:4.2f} | "
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
# 11. Save results
# ==========================================================

df.to_csv(
    "pcmci_measurement_noise_raw.csv",
    index=False
)

summary.to_csv(
    "pcmci_measurement_noise_summary.csv",
    index=False
)

print("\nResults saved:")
print(
    "  pcmci_measurement_noise_raw.csv"
)
print(
    "  pcmci_measurement_noise_summary.csv"
)
