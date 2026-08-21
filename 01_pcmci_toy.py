import numpy as np
import matplotlib.pyplot as plt

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# ==============================
# 1. 生成具有已知因果结构的数据
# ==============================

SEED = 40
T = 50
BURN_IN = 200

rng = np.random.default_rng(SEED)

total_T = T + BURN_IN

# 三个变量：X, Y, Z
data = np.zeros((total_T, 3))

# 随机噪声
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


# 丢弃前 200 个过渡样本
data = data[BURN_IN:]

var_names = ["X", "Y", "Z"]

print("Data shape:", data.shape)
print("\nFirst 5 rows:")
print(data[:5])


# ==============================
# 2. 画前 200 个时间点
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(data[:200, 0], label="X")
plt.plot(data[:200, 1], label="Y")
plt.plot(data[:200, 2], label="Z")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Synthetic causal time series")
plt.legend()
plt.tight_layout()

plt.show()
# ==============================
# 3. 转换成 Tigramite DataFrame
# ==============================

dataframe = pp.DataFrame(
    data=data,
    var_names=var_names
)

# ==============================
# 4. 创建条件独立检验
# ==============================

parcorr = ParCorr(
    significance="analytic"
)

# ==============================
# 5. 创建 PCMCI 对象
# ==============================

pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=1
)

# ==============================
# 6. 运行 PCMCI+
# ==============================

results = pcmci.run_pcmciplus(
    tau_min=1,
    tau_max=2,
    pc_alpha=0.01
)

print("\nPCMCI+ finished.")
print("Graph shape:", results["graph"].shape)

# ==============================
# 7. 提取 PCMCI+ 发现的因果边
# ==============================

graph = results["graph"]

pred_edges = set()

for tau in range(1, 3):  # 因为我们设置了 tau_min=1, tau_max=2

    for i, source in enumerate(var_names):

        for j, target in enumerate(var_names):

            if graph[i, j, tau] == "-->":

                pred_edges.add(
                    (source, target, tau)
                )

print("\nPredicted edges:")

for edge in sorted(pred_edges):
    print(edge)

# ==============================
# 8. 定义 Ground Truth
# ==============================

true_edges = {
    ("X", "X", 1),
    ("X", "Y", 2),
    ("Y", "Y", 1),
    ("Y", "Z", 1),
    ("Z", "Z", 1),
}

print("\nTrue edges:")

for edge in sorted(true_edges):
    print(edge)

# ==============================
# 9. 比较预测图和真实图
# ==============================

tp_edges = true_edges & pred_edges
fp_edges = pred_edges - true_edges
fn_edges = true_edges - pred_edges

TP = len(tp_edges)
FP = len(fp_edges)
FN = len(fn_edges)

print("\nTP edges:")
for edge in sorted(tp_edges):
    print(edge)

print("\nFP edges:")
for edge in sorted(fp_edges):
    print(edge)

print("\nFN edges:")
for edge in sorted(fn_edges):
    print(edge)

# ==============================
# 10. 计算 Precision / Recall / F1
# ==============================

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)

print("\n==============================")
print("Evaluation Metrics")
print("==============================")

print(f"TP = {TP}")
print(f"FP = {FP}")
print(f"FN = {FN}")

print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1        = {f1:.4f}")