"""受约束的固定 15 mm 阻尼带 ABH 参数-频率单变量效应分析。

本工具只读取现有 24 阶独立 ABH 模态样本库；不调用 Abaqus，也不改动
反向设计、材料、边界、阻尼或几何建模代码。所谓“单变量效应”通过一组
代表性已观测几何作为条件锚点计算：仅改变一个参数，其他八个参数保持不变，
并且每个反事实点都必须通过当前 V2 的代数几何/固定 15 mm 阻尼带约束。

输出应被理解为受当前数据库覆盖范围支持的条件敏感性诊断，而非跨几何的
物理模态分支因果证明。频率仍是排序后的特征频率；本工具不进行 MAC 模态追踪。
"""
from __future__ import print_function

import argparse
import csv
import datetime
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
for folder in (os.path.join(PROJECT_ROOT, "01_CORE"),
               os.path.join(PROJECT_ROOT, "02_ABAQUS", "assembly"),
               os.path.join(PROJECT_ROOT, "02_ABAQUS", "geometry"),
               os.path.join(PROJECT_ROOT, "04_INVERSE_DESIGN")):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import config
from target_retrieval import FIXED_BAND_9D
from frequency_generation import _fixed_band_geometry_valid


MODE_COUNT = 8
GRID_COUNT = 31
INTERACTION_GRID_COUNT = 25
ANCHOR_COUNT = 40
RANDOM_SEED = 20261020
INTERACTION_PAIRS = (
    ("cutoff_thickness", "platform_height"),
    ("cutoff_thickness", "power_law"),
    ("Rou_2", "Rou_3"),
    ("Rou_4", "theta"),
)

DISPLAY = {
    "Rou_2": (1000.0, "Rou_2 (mm)"),
    "Rou_3": (1000.0, "Rou_3 (mm)"),
    "Rou_4": (1000.0, "Rou_4 (mm)"),
    "cutoff_thickness": (1000.0, "cutoff thickness (mm)"),
    "equal_t_length": (1000.0, "equal_t_length (mm)"),
    "equal_t_width": (1000.0, "equal_t_width (mm)"),
    "platform_height": (1000.0, "platform height (mm)"),
    "power_law": (1.0, "power law (-)"),
    "theta": (180.0 / math.pi, "theta (deg)"),
}


def _default_dataset():
    """读取用户当前锁定的版本化固定 15 mm 训练库。"""
    user_design = os.path.join(PROJECT_ROOT, "00_USER_DESIGN")
    if user_design not in sys.path:
        sys.path.insert(0, user_design)
    import design_request
    return design_request.FIXED_BAND_DATASET_CSV


def _load_dataset(path):
    rows, vectors, spectra = [], [], []
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("status") != "MODAL_OK":
                continue
            try:
                vector = [float(raw[key]) for key in FIXED_BAND_9D]
                modes = [float(raw["f_mode_%02d_hz" % i]) for i in range(1, MODE_COUNT + 1)]
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.all(np.isfinite(vector)) and np.all(np.isfinite(modes)) and
                    min(modes) > 0.0 and np.all(np.diff(modes) >= 0.0)):
                continue
            rows.append(raw.get("sample_id", ""))
            vectors.append(vector)
            spectra.append(modes)
    if len(rows) < 100:
        raise RuntimeError("INSUFFICIENT_VALID_FIXED_BAND_SAMPLES:%d" % len(rows))
    return rows, np.asarray(vectors, dtype=float), np.asarray(spectra, dtype=float)


def _representative_indices(x):
    """用聚类中心附近的真实几何构造多锚点，而非依赖单一基准构型。"""
    scaled = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1.0e-12)
    n_clusters = min(ANCHOR_COUNT, len(x))
    clusters = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED,
                               n_init=5, batch_size=min(1024, len(x))).fit(scaled)
    selected = []
    for center in clusters.cluster_centers_:
        order = np.argsort(np.sum((scaled - center) ** 2, axis=1))
        for index in order:
            if int(index) not in selected:
                selected.append(int(index))
                break
    return np.asarray(selected, dtype=int)


def _constraint_records(x):
    index = {key: i for i, key in enumerate(FIXED_BAND_9D)}
    r2, r3, r4 = (x[:, index[key]] for key in ("Rou_2", "Rou_3", "Rou_4"))
    theta = x[:, index["theta"]]
    platform = x[:, index["platform_height"]]
    cutoff = x[:, index["cutoff_thickness"]]
    margins = {
        "Rou_2 - Rou_3*cos(theta)": r2 - r3 * np.cos(theta),
        "Rou_4 - Rou_3*sin(theta)": r4 - r3 * np.sin(theta),
        "platform_height - cutoff_thickness": platform - cutoff,
    }
    rows = []
    for name, values in margins.items():
        rows.append({"Record_Type": "Constraint_Margin", "Parameter_A": name,
                     "Parameter_B": "", "Pearson": "", "Spearman": "",
                     "Minimum": float(np.min(values)), "P05": float(np.percentile(values, 5)),
                     "Median": float(np.median(values)), "P95": float(np.percentile(values, 95)),
                     "Maximum": float(np.max(values)), "Count": int(len(values)),
                     "Interpretation": "Must exceed 0.5 mm for Rou margins; platform must exceed cutoff."})
    for i, first in enumerate(FIXED_BAND_9D):
        for j in range(i + 1, len(FIXED_BAND_9D)):
            second = FIXED_BAND_9D[j]
            pearson = np.corrcoef(x[:, i], x[:, j])[0, 1]
            rank_a = np.argsort(np.argsort(x[:, i]))
            rank_b = np.argsort(np.argsort(x[:, j]))
            spearman = np.corrcoef(rank_a, rank_b)[0, 1]
            rows.append({"Record_Type": "Parameter_Pair", "Parameter_A": first,
                         "Parameter_B": second, "Pearson": float(pearson),
                         "Spearman": float(spearman), "Minimum": "", "P05": "",
                         "Median": "", "P95": "", "Maximum": "", "Count": int(len(x)),
                         "Interpretation": "Observational association only; not a causal coefficient."})
    return rows


def _is_feasible(vector, lower, upper):
    valid, _, _ = _fixed_band_geometry_valid(np.asarray(vector, dtype=float), lower, upper)
    return bool(valid)


def _conditional_effects(x, y, model, anchors, lower, upper):
    records, summaries = [], []
    base = model.predict(x[anchors])
    for feature_index, feature in enumerate(FIXED_BAND_9D):
        # The central 98 % of observed support avoids treating isolated extrema as
        # a design range, while still reporting the complete observed min/max later.
        scan_low, scan_high = np.percentile(x[:, feature_index], [1.0, 99.0])
        grid = np.linspace(scan_low, scan_high, GRID_COUNT)
        # Use the same anchors over the whole scan.  Without this condition the
        # mean curve could change merely because a different subset becomes
        # feasible at another grid point, which would not be a single-variable
        # effect.  A small common-anchor count is reported as limited support.
        common = []
        for anchor in anchors:
            base_vector = x[anchor].copy()
            if all(_is_feasible(np.concatenate((base_vector[:feature_index], [value],
                                                base_vector[feature_index + 1:])), lower, upper)
                   for value in grid):
                common.append(int(anchor))
        common = np.asarray(common, dtype=int)
        per_mode_means = [[] for _ in range(MODE_COUNT)]
        feasible_counts = []
        for value in grid:
            proposals = x[common].copy()
            proposals[:, feature_index] = value
            # Common-anchor screening above makes all rows valid. Keep this
            # explicit check as a guard against future geometry-rule changes.
            keep = np.asarray([_is_feasible(row, lower, upper) for row in proposals], dtype=bool)
            feasible_counts.append(int(np.count_nonzero(keep)))
            prediction = model.predict(proposals[keep]) if np.any(keep) else np.empty((0, MODE_COUNT))
            for mode in range(MODE_COUNT):
                values = prediction[:, mode] if len(prediction) else np.asarray([])
                mean = float(np.mean(values)) if len(values) else float("nan")
                p10 = float(np.percentile(values, 10)) if len(values) else float("nan")
                p90 = float(np.percentile(values, 90)) if len(values) else float("nan")
                per_mode_means[mode].append(mean)
                records.append({"Parameter": feature, "Grid_Value_Raw": float(value),
                                "Grid_Value_Display": float(value * DISPLAY[feature][0]),
                                "Display_Unit_Label": DISPLAY[feature][1],
                                "Mode_Index": mode + 1, "Feasible_Anchor_Count": int(np.count_nonzero(keep)),
                                "Anchor_Count": int(len(anchors)), "Predicted_Frequency_Mean_Hz": mean,
                                "Predicted_Frequency_P10_Hz": p10,
                                "Predicted_Frequency_P90_Hz": p90})
        for mode, means in enumerate(per_mode_means):
            means = np.asarray(means, dtype=float)
            finite = np.isfinite(means)
            if np.count_nonzero(finite) < 3:
                direction, trend = "INSUFFICIENT_FEASIBLE_POINTS", float("nan")
                low, high, delta, rel = (float("nan"),) * 4
            else:
                xv, yv = grid[finite], means[finite]
                low, high = float(yv[0]), float(yv[-1])
                delta = high - low
                rel = delta / max(abs(float(np.median(base[:, mode]))), 1.0e-9) * 100.0
                trend = float(np.corrcoef(xv, yv)[0, 1])
                if abs(trend) < 0.35:
                    direction = "NON_MONOTONIC_OR_WEAK"
                else:
                    dy = np.diff(yv)
                    sign_changes = int(np.count_nonzero(np.sign(dy[1:]) != np.sign(dy[:-1]))) if len(dy) > 1 else 0
                    if sign_changes > max(2, len(dy) // 5):
                        direction = "NON_MONOTONIC"
                    else:
                        direction = "INCREASES" if delta > 0 else "DECREASES"
            summaries.append({"Parameter": feature, "Mode_Index": mode + 1,
                              "Observed_Min_Raw": float(np.min(x[:, feature_index])),
                              "Observed_Max_Raw": float(np.max(x[:, feature_index])),
                              "Scan_P01_Raw": float(scan_low), "Scan_P99_Raw": float(scan_high),
                              "Display_Unit_Label": DISPLAY[feature][1],
                              "Frequency_At_Low_Scan_Hz": low, "Frequency_At_High_Scan_Hz": high,
                              "Conditional_Effect_Delta_Hz": delta,
                              "Conditional_Effect_Relative_To_Anchor_Median_Percent": rel,
                              "Trend_Correlation": trend, "Effect_Direction": direction,
                              "Minimum_Feasible_Anchor_Count": int(min(feasible_counts)),
                              "Maximum_Feasible_Anchor_Count": int(max(feasible_counts)),
                              "Constraint_Aware": True,
                              "Interpretation": "Surrogate conditional effect across representative feasible anchors; not physical branch tracking."})
    return records, summaries


def _local_matched_effects(x, y):
    """提供数据本身的局部匹配检查，防止只依赖代理模型曲线。"""
    result = []
    scaled = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1.0e-12)
    for feature_index, feature in enumerate(FIXED_BAND_9D):
        other = np.delete(scaled, feature_index, axis=1)
        nn = NearestNeighbors(n_neighbors=min(8, len(x))).fit(other)
        distances, neighbors = nn.kneighbors(other)
        slopes = [[] for _ in range(MODE_COUNT)]
        other_distances = []
        x_range = max(float(np.ptp(x[:, feature_index])), 1.0e-12)
        for row in range(len(x)):
            for candidate, other_distance in zip(neighbors[row, 1:], distances[row, 1:]):
                change = (x[candidate, feature_index] - x[row, feature_index]) / x_range
                if abs(change) >= 0.05:
                    other_distances.append(float(other_distance))
                    for mode in range(MODE_COUNT):
                        slopes[mode].append(float((y[candidate, mode] - y[row, mode]) / change))
                    break
        for mode, values in enumerate(slopes):
            values = np.asarray(values, dtype=float)
            result.append({"Parameter": feature, "Mode_Index": mode + 1,
                           "Matched_Pair_Count": int(len(values)),
                           "Median_Other_8D_Distance": float(np.median(other_distances)) if other_distances else float("nan"),
                           "Median_Slope_Hz_Per_Observed_Range": float(np.median(values)) if len(values) else float("nan"),
                           "P10_Slope_Hz_Per_Observed_Range": float(np.percentile(values, 10)) if len(values) else float("nan"),
                           "P90_Slope_Hz_Per_Observed_Range": float(np.percentile(values, 90)) if len(values) else float("nan"),
                           "Interpretation": "Nearest-neighbour observational check; residual changes in other parameters remain."})
    return result


def _central_reference_index(x):
    """选取九维中位数附近的一条真实样本，作为二维规则图的明确控制基准。"""
    centered = (x - np.median(x, axis=0)) / np.maximum(np.std(x, axis=0), 1.0e-12)
    return int(np.argmin(np.sum(centered ** 2, axis=1)))


def _interaction_maps(x, model, lower, upper, reference_index):
    """在同一个真实参考构型上扫描两个参数，同时保留其它七个参数。"""
    records = []
    reference = x[reference_index].copy()
    for first, second in INTERACTION_PAIRS:
        first_index, second_index = FIXED_BAND_9D.index(first), FIXED_BAND_9D.index(second)
        first_grid = np.linspace(*np.percentile(x[:, first_index], [1.0, 99.0]), INTERACTION_GRID_COUNT)
        second_grid = np.linspace(*np.percentile(x[:, second_index], [1.0, 99.0]), INTERACTION_GRID_COUNT)
        for first_value in first_grid:
            for second_value in second_grid:
                vector = reference.copy()
                vector[first_index] = first_value
                vector[second_index] = second_value
                valid, reason, _ = _fixed_band_geometry_valid(vector, lower, upper)
                prediction = model.predict(vector.reshape(1, -1))[0] if valid else [float("nan")] * MODE_COUNT
                for mode in range(MODE_COUNT):
                    records.append({
                        "Parameter_X": first, "Parameter_Y": second,
                        "X_Value_Raw": float(first_value), "Y_Value_Raw": float(second_value),
                        "X_Value_Display": float(first_value * DISPLAY[first][0]),
                        "Y_Value_Display": float(second_value * DISPLAY[second][0]),
                        "X_Display_Label": DISPLAY[first][1], "Y_Display_Label": DISPLAY[second][1],
                        "Mode_Index": mode + 1, "Geometry_Valid": bool(valid),
                        "Constraint_Result": "OK" if valid else str(reason),
                        "Predicted_Frequency_Hz": float(prediction[mode]),
                        "Reference_Index": int(reference_index),
                    })
    return records


def _plot_interactions(records, output):
    for first, second in INTERACTION_PAIRS:
        group = [row for row in records if row["Parameter_X"] == first and row["Parameter_Y"] == second]
        x_values = sorted({row["X_Value_Display"] for row in group})
        y_values = sorted({row["Y_Value_Display"] for row in group})
        x_index = {value: index for index, value in enumerate(x_values)}
        y_index = {value: index for index, value in enumerate(y_values)}
        figure, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=150, sharex=True, sharey=True)
        for mode in range(1, MODE_COUNT + 1):
            axis = axes.flat[mode - 1]
            grid = np.full((len(y_values), len(x_values)), np.nan)
            for row in group:
                if int(row["Mode_Index"]) == mode and row["Geometry_Valid"]:
                    grid[y_index[row["Y_Value_Display"]], x_index[row["X_Value_Display"]]] = row["Predicted_Frequency_Hz"]
            image = axis.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                                extent=[min(x_values), max(x_values), min(y_values), max(y_values)], cmap="viridis")
            axis.set_title("Mode %d" % mode)
            axis.set_xlabel(DISPLAY[first][1])
            axis.set_ylabel(DISPLAY[second][1])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Hz")
        figure.suptitle("Constraint-aware two-parameter map; all other parameters fixed at a central observed geometry", y=1.02)
        figure.tight_layout()
        figure.savefig(os.path.join(output, "interaction_%s__%s_modes_1_8.png" % (first, second)), bbox_inches="tight")
        plt.close(figure)


def _interaction_report(path, sample_id, reference, interaction_records):
    lines = [
        "# 参数-频率设计规则层：二维约束图", "",
        "## 用途", "",
        "本文件是反向设计前的预筛选/解释层，不替代完整 9D 代理模型或独立 Abaqus 验证。",
        "二维图的参考构型为当前训练库中最接近九维中位数的一条真实样本：`%s`。" % sample_id,
        "其余七个参数保持为该参考值；每个二维网格点均执行当前 V2 固定 15 mm 几何可行性检查。", "",
        "## 参考构型（原始单位：m、rad、无量纲）", ""
    ]
    for key, value in zip(FIXED_BAND_9D, reference):
        lines.append("- `%s` = %.12g" % (key, value))
    lines.extend(["", "## 图的读取方式", "",
                  "- 彩色区域：当前约束下有效、且代理模型给出排序模态频率的位置。",
                  "- 空白区域：几何/固定 15 mm 阻尼带不满足当前生成链约束；不是零频率，也不是代理预测失败。",
                  "- 仅能在该参考构型附近解释交互趋势；若其它七个参数改变，曲面也会改变。", "",
                  "## 各二维扫描的有效网格覆盖", ""])
    for first, second in INTERACTION_PAIRS:
        subset = [row for row in interaction_records if row["Parameter_X"] == first and row["Parameter_Y"] == second and row["Mode_Index"] == 1]
        valid = sum(1 for row in subset if row["Geometry_Valid"])
        lines.append("- `%s × %s`：%d/%d 个网格点有效。" % (first, second, valid, len(subset)))
    lines.extend(["", "## 反向设计中的使用顺序", "",
                  "1. 根据目标频率与尺寸约束，先使用二维图排除明显不可达/不合法的参数组合。",
                  "2. 将保留区间作为完整 9D 反向模型的搜索边界或初始点，而不是将二维图直接当作唯一反演方程。",
                  "3. 对最终候选继续执行 9D 代理预测、几何检查和独立 Abaqus 模态验证。"])
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def _write_csv(path, records):
    if not records:
        raise RuntimeError("NO_RECORDS_TO_WRITE:%s" % path)
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_effects(records, output):
    by_feature = {feature: [] for feature in FIXED_BAND_9D}
    for record in records:
        by_feature[record["Parameter"]].append(record)
    for feature, group in by_feature.items():
        figure, axis = plt.subplots(figsize=(8.4, 5.2), dpi=150)
        for mode in range(1, MODE_COUNT + 1):
            curve = [row for row in group if row["Mode_Index"] == mode]
            curve.sort(key=lambda row: row["Grid_Value_Raw"])
            gx = [row["Grid_Value_Display"] for row in curve]
            gy = [row["Predicted_Frequency_Mean_Hz"] for row in curve]
            p10 = [row["Predicted_Frequency_P10_Hz"] for row in curve]
            p90 = [row["Predicted_Frequency_P90_Hz"] for row in curve]
            line, = axis.plot(gx, gy, linewidth=1.25, label="Mode %d" % mode)
            axis.fill_between(gx, p10, p90, color=line.get_color(), alpha=0.08)
        axis.set_xlabel(DISPLAY[feature][1])
        axis.set_ylabel("predicted sorted frequency (Hz)")
        axis.set_title("Constraint-aware conditional effect: %s" % feature)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figure.savefig(os.path.join(output, "effect_%s_modes_1_8.png" % feature), bbox_inches="tight")
        plt.close(figure)


def _plot_heatmap(summary, output):
    matrix = np.full((len(FIXED_BAND_9D), MODE_COUNT), np.nan)
    for record in summary:
        i = FIXED_BAND_9D.index(record["Parameter"])
        matrix[i, int(record["Mode_Index"]) - 1] = abs(float(record["Conditional_Effect_Relative_To_Anchor_Median_Percent"]))
    figure, axis = plt.subplots(figsize=(10, 6.2), dpi=160)
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(MODE_COUNT))
    axis.set_xticklabels(["Mode %d" % i for i in range(1, MODE_COUNT + 1)])
    axis.set_yticks(range(len(FIXED_BAND_9D)))
    axis.set_yticklabels(FIXED_BAND_9D)
    axis.set_title("Absolute conditional frequency span (% of anchor-median frequency)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, "%.1f" % matrix[i, j], ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="absolute relative span (%)")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "parameter_effect_rank_heatmap.png"), bbox_inches="tight")
    plt.close(figure)


def _report(path, dataset, count, anchors, summary, constraint_path, local_path):
    lines = [
        "# 固定 15 mm 阻尼带 ABH：受约束单变量参数-频率影响分析", "",
        "## 分析范围", "",
        "- 数据源：`%s`。" % dataset,
        "- 有效几何：%d；代表性条件锚点：%d；使用排序后的 Mode 1–8。" % (count, anchors),
        "- 固定条件：中心平台固支、当前 V2 厚度场/几何链、横向/纵向阻尼各向内 15 mm、材料、阻尼层/约束层厚度、网格和特征值设置均保持为数据集来源的固定定义。",
        "- 本报告不运行 Abaqus；不改变当前反向设计主链。", "",
        "## 方法", "",
        "1. 训练 ExtraTrees 多输出谱代理，仅用于平滑读取既有 3445 组（以实际读入数为准）数据的局部趋势。",
        "2. 对每个参数，其他八个参数固定为 40 个代表性已观测几何；扫描该参数的观测 P01–P99 支持区间。",
        "3. 每个扫描点均调用当前固定 15 mm 生成链的代数可行性规则：数据范围、`Rou_2 - Rou_3 cos(theta) >= 0.5 mm`、`Rou_4 - Rou_3 sin(theta) >= 0.5 mm`、`platform_height > cutoff_thickness` 和固定阻尼带可行性。",
        "4. 输出扫描中各可行锚点的预测中位趋势（均值）及 P10–P90 跨锚点离散带；另给出原始数据库的局部近邻配对核查。", "",
        "## 解释边界", "",
        "- 此处的 Mode 是排序后的特征频率，不等同于已跨几何 MAC 追踪的同一物理模态分支。",
        "- 参数在实际库中存在联动；本报告通过条件锚点和约束筛除减少混杂，但不能将结果表述为无条件因果系数。",
        "- 扫描结果只在当前数据覆盖的 P01–P99 范围内适用；不可外推到未覆盖尺寸或材料/边界/阻尼条件。", "",
        "## Mode 1–8 的最大条件敏感参数", ""
    ]
    for mode in range(1, MODE_COUNT + 1):
        subset = [row for row in summary if int(row["Mode_Index"]) == mode and
                  np.isfinite(float(row["Conditional_Effect_Relative_To_Anchor_Median_Percent"]))]
        subset.sort(key=lambda row: abs(float(row["Conditional_Effect_Relative_To_Anchor_Median_Percent"])), reverse=True)
        top = subset[:3]
        formatted = "; ".join("%s: %.2f%% (%s)" %
                              (row["Parameter"], row["Conditional_Effect_Relative_To_Anchor_Median_Percent"], row["Effect_Direction"])
                              for row in top)
        lines.append("- Mode %d：%s" % (mode, formatted))
    support = {}
    for row in summary:
        support[row["Parameter"]] = int(row["Minimum_Feasible_Anchor_Count"])
    limited = ["%s (%d/%d anchors)" % (key, value, anchors)
               for key, value in support.items() if value < max(10, int(0.75 * anchors))]
    lines.extend([
        "", "## 约束覆盖警告", "",
        "为避免因扫描过程中可行锚点集合变化而产生假趋势，每个参数曲线只使用在整个 P01–P99 扫描范围内始终可行的同一组锚点。",
        ("以下参数受 Rou/角度等耦合约束影响、完整扫描的共同锚点较少：%s。它们的曲线只能作为受限可行域内的趋势，不能外推。" % "; ".join(limited)) if limited else "所有参数均有足够的共同可行锚点。",
        "", "## 文件", "",
        "- `parameter_frequency_effects_summary.csv`：九个参数 × Mode 1–8 的受约束条件效应汇总。",
        "- `conditional_parameter_frequency_curves.csv`：全部扫描点、可行锚点数量和跨锚点预测离散带。",
        "- `%s`：参数相关性与三条几何约束余量。" % os.path.basename(constraint_path),
        "- `%s`：原始数据库局部近邻配对的独立核查。" % os.path.basename(local_path),
        "- `effect_<parameter>_modes_1_8.png` 和 `parameter_effect_rank_heatmap.png`：可视化结果。",
        "", "## 使用建议", "",
        "后续若要依据单变量曲线设置设计约束，应先查看对应参数的 `Minimum_Feasible_Anchor_Count`。可行锚点少的区段受约束关系影响较大，不能作为独立调参规律。对最终新几何仍应以独立 Abaqus 模态验证为准。"
    ])
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def run(dataset=None, output_root=None):
    dataset = dataset or _default_dataset()
    if not os.path.isfile(dataset):
        raise RuntimeError("DATASET_NOT_FOUND:%s" % dataset)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_root or os.path.join(PROJECT_ROOT, "07_RESULTS", "parameter_frequency_single_variable_analysis")
    output = os.path.join(output_root, stamp)
    os.makedirs(output)
    sample_ids, x, y = _load_dataset(dataset)
    lower, upper = x.min(axis=0), x.max(axis=0)
    anchors = _representative_indices(x)
    model = ExtraTreesRegressor(n_estimators=400, random_state=RANDOM_SEED,
                                min_samples_leaf=2, max_features=1.0, n_jobs=-1)
    model.fit(x, y)
    curves, summary = _conditional_effects(x, y, model, anchors, lower, upper)
    constraints = _constraint_records(x)
    local = _local_matched_effects(x, y)
    reference_index = _central_reference_index(x)
    interactions = _interaction_maps(x, model, lower, upper, reference_index)
    curve_path = os.path.join(output, "conditional_parameter_frequency_curves.csv")
    summary_path = os.path.join(output, "parameter_frequency_effects_summary.csv")
    constraint_path = os.path.join(output, "parameter_relationships_and_constraints.csv")
    local_path = os.path.join(output, "local_matched_pair_effects.csv")
    interaction_path = os.path.join(output, "parameter_interaction_frequency_maps.csv")
    _write_csv(curve_path, curves)
    _write_csv(summary_path, summary)
    _write_csv(constraint_path, constraints)
    _write_csv(local_path, local)
    _write_csv(interaction_path, interactions)
    _plot_effects(curves, output)
    _plot_heatmap(summary, output)
    _plot_interactions(interactions, output)
    report_path = os.path.join(output, "PARAMETER_FREQUENCY_SINGLE_VARIABLE_REPORT.md")
    _report(report_path, dataset, len(x), len(anchors), summary, constraint_path, local_path)
    interaction_report = os.path.join(output, "PARAMETER_FREQUENCY_DESIGN_RULES.md")
    _interaction_report(interaction_report, sample_ids[reference_index], x[reference_index], interactions)
    manifest = {
        "status": "COMPLETED", "dataset": dataset, "valid_sample_count": int(len(x)),
        "modes": "sorted_mode_1_to_8", "anchor_count": int(len(anchors)),
        "abaqus_job_submitted": False, "material_boundary_damping_changed": False,
        "output_directory": output, "report": report_path,
        "interaction_report": interaction_report, "interaction_reference_sample_id": sample_ids[reference_index],
        "constraints": ["Rou_2-Rou_3*cos(theta)>=0.5mm", "Rou_4-Rou_3*sin(theta)>=0.5mm",
                        "platform_height>cutoff_thickness", "fixed_band_15mm_feasible"]
    }
    import json
    with open(os.path.join(output, "run_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print("PARAMETER_FREQUENCY_ANALYSIS_COMPLETED")
    print("OUTPUT_DIRECTORY=%s" % output)
    print("VALID_SAMPLES=%d" % len(x))
    print("ABAQUS_JOB_SUBMITTED=NO")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constraint-aware fixed-band ABH parameter-frequency analysis")
    parser.add_argument("--dataset", default=None, help="optional fixed-band 24-mode CSV")
    parser.add_argument("--output-root", default=None, help="optional result root")
    arguments = parser.parse_args()
    run(arguments.dataset, arguments.output_root)
