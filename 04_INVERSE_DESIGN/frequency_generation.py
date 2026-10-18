"""Fixed-damping inverse generation, geometry-group validation and bounded FE checks.

Separate from (and never writes) the accepted modal database or retrieval logic.
New validation responses are never added to the fitted model in this experiment.
"""
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import warnings
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.linalg import solve_triangular
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import config
from paths import (FREQUENCY9D_RESULTS_DIR as OUTPUT, FREQUENCY9D_RUNS_DIR as RUNS,
                   PROJECT_ROOT, DATABASE_DIR,
                   ABAQUS_TEMP_DIR, ensure_directory)
from target_retrieval import (_load_valid_modes, MODAL_DATABASE, REQUIRED_11D, retrieve,
                              FIXED_BAND_9D, FIXED_BAND_LIBRARY)
from general_geometry_validator import validate_general
from abaqus_cleanup import cleanup_abaqus_job
from damping_policy import fixed_band_width_to_rx_ry
from damping_mapping import shell_outer_dimensions

GEOMETRY = tuple(k for k in REQUIRED_11D if k not in ("rx", "ry"))
FEATURES = GEOMETRY + ("rx", "ry")
POLICY = config.FREQUENCY9D_POLICY
FIXED_BAND_TARGET_POLICY = config.FIXED_BAND_TARGET_GENERATION_POLICY
MODELS = ("A_FIXED_9D_GPR", "B_CONDITIONAL_11D_GPR", "C_HIERARCHICAL_9_PLUS_2_GPR", "D_11D_EXTRATREES")
BASE_OUTPUT = OUTPUT


def _fixed_band_training_data(dataset_csv):
    """读取固定 15 mm 样本库的 9D 几何与完整 24 阶排序频谱。"""
    if not dataset_csv or not os.path.isfile(dataset_csv):
        raise RuntimeError("FIXED_BAND_15MM_DATASET_MISSING:%s" % dataset_csv)
    rows = []
    with open(dataset_csv, newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("status") != "MODAL_OK":
                continue
            try:
                parameters = {key: float(raw[key]) for key in FIXED_BAND_9D}
                spectrum = [float(raw["f_mode_%02d_hz" % index]) for index in range(1, 25)]
                if (not np.all(np.isfinite(list(parameters.values()) + spectrum)) or
                        min(spectrum) <= 0.0 or np.any(np.diff(spectrum) < 0)):
                    continue
                rows.append({"Sample_ID": raw["sample_id"], "Parameters": parameters,
                             "Spectrum_Hz": spectrum})
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 20:
        raise RuntimeError("INSUFFICIENT_FIXED_BAND_TRAINING_GEOMETRIES")
    x = np.array([[row["Parameters"][key] for key in FIXED_BAND_9D] for row in rows])
    y = np.array([row["Spectrum_Hz"] for row in rows])
    return rows, x, y


class _FixedBandSpectrumGPR(object):
    """一个联合 24 输出 GPR；只用于固定 15 mm 样本库的实验性生成。"""
    def fit(self, x, y, max_iterations=None):
        self.input_scaler = StandardScaler().fit(x)
        # 每阶输出单独标准化，使高阶频率不会主导核参数拟合。
        self.output_scaler = StandardScaler().fit(y)
        kernel = ConstantKernel(1.0, (1.0e-2, 1.0e2)) * Matern(
            length_scale=np.ones(x.shape[1]), length_scale_bounds=(1.0e-2, 1.0e3), nu=2.5
        ) + WhiteKernel(1.0e-5, (1.0e-8, 1.0e-2))
        options = {}
        if max_iterations is not None:
            # 仅实验模型使用显式训练预算；原生产模型的优化设置保持不变。
            def bounded_optimizer(objective, initial, bounds):
                result = minimize(objective, initial, jac=True, bounds=bounds,
                                  method="L-BFGS-B", options={"maxiter": max_iterations})
                self.optimizer_result = {"Success": bool(result.success),
                                         "Message": str(result.message), "Iterations": int(result.nit)}
                return result.x, result.fun
            options["optimizer"] = bounded_optimizer
        self.model = GaussianProcessRegressor(
            kernel=kernel, alpha=1.0e-8, normalize_y=False, n_restarts_optimizer=1,
            random_state=20261013,
            **options,
        ).fit(self.input_scaler.transform(x), self.output_scaler.transform(y))
        # 局部函数不能写入缓存；预测不依赖优化器。
        if max_iterations is not None:
            self.model.optimizer = None
        return self

    def predict(self, x):
        mean_scaled, std_scaled = self.model.predict(self.input_scaler.transform(np.atleast_2d(x)), return_std=True)
        # sklearn 单输出会压缩成一维，统一为 [样本, 模态]。
        mean_scaled = np.asarray(mean_scaled).reshape(len(np.atleast_2d(x)), -1)
        mean = self.output_scaler.inverse_transform(mean_scaled)
        std_scaled = np.asarray(std_scaled)
        if std_scaled.ndim == 1:
            std_scaled = np.tile(std_scaled[:, None], (1, mean.shape[1]))
        std = std_scaled * self.output_scaler.scale_
        return mean, std


class _FixedBandAnchorDelta(object):
    """真实锚点频率 + GPR预测增量；只使用训练集锚点，不进行模态身份假设。"""

    def __init__(self, baseline, x, y, sample_ids):
        self.baseline = baseline
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.sample_ids = list(sample_ids)
        self.span = np.maximum(np.ptp(self.x, axis=0), 1.0e-12)
        self.tree = cKDTree(self.x / self.span)
        self.anchor_mean = baseline.predict(self.x)[0]
        self.scaled_x = baseline.input_scaler.transform(self.x)
        # 当前正式核为 Constant*Matern + WhiteKernel。这里只计算增量的潜在协方差，
        # 不把WhiteKernel噪声重复加在两个端点上；仍必须由独立校准残差修正。
        kernel = baseline.model.kernel_
        if not isinstance(kernel.k2, WhiteKernel):
            raise RuntimeError("UNSUPPORTED_ANCHOR_DELTA_KERNEL")
        self.signal = kernel.k1
        self.cross = self.signal(self.scaled_x, baseline.model.X_train_)

    def anchors(self, x):
        return self.tree.query(np.atleast_2d(x) / self.span)

    def predict(self, x):
        vectors = np.atleast_2d(x)
        _, anchor = self.anchors(vectors)
        raw_mean = self.baseline.predict(vectors)[0]
        mean = self.y[anchor] + raw_mean - self.anchor_mean[anchor]
        scaled = self.baseline.input_scaler.transform(vectors)
        variances = []
        # 分块避免为大量候选构造整个成对协方差矩阵。
        for start in range(0, len(vectors), 128):
            q = scaled[start:start + 128]
            a = self.scaled_x[anchor[start:start + 128]]
            difference = self.signal(q, self.baseline.model.X_train_) - self.cross[anchor[start:start + 128]]
            solved = solve_triangular(self.baseline.model.L_, difference.T, lower=True, check_finite=False)
            prior = self.signal.diag(q) + self.signal.diag(a) - 2 * np.diag(self.signal(q, a))
            variance = prior - np.sum(solved * solved, axis=0)
            if np.any(variance < -1.0e-6):
                raise RuntimeError("INVALID_INCREMENT_COVARIANCE")
            variances.extend(np.maximum(variance, 0))
        scale = np.sqrt(np.asarray(variances))[:, None] * self.baseline.output_scaler.scale_[None, :]
        # scale是待校准的增量尺度，不宣称它已经是实际频率误差的标准差。
        return mean, np.maximum(scale, 1.0e-9)


def _fixed_band_geometry_groups(x, radius):
    """用九维归一化距离的连通分量分组，近重复点不能跨数据分区。"""
    span = np.maximum(np.ptp(x, axis=0), 1.0e-12)
    parent = np.arange(len(x))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree((x - x.min(axis=0)) / span).query_pairs(radius):
        parent[root(a)] = root(b)
    return np.array([root(i) for i in range(len(x))])


def _fixed_band_calibration_factor(actual, mean, std, coverage, minimum):
    """仅用独立校准集的同排序阶次残差校准；不把它当作物理模态追踪。"""
    scores = np.abs(actual - mean) / np.maximum(std, 1.0e-9)
    if len(scores) < minimum or not np.all(np.isfinite(scores)):
        raise RuntimeError("INSUFFICIENT_OR_INVALID_CALIBRATION_DATA")
    order = int(math.ceil((len(scores) + 1) * coverage))
    if order > len(scores):
        raise RuntimeError("INSUFFICIENT_CALIBRATION_FOR_REQUESTED_COVERAGE")
    return float(np.sort(scores)[order - 1])


def _fixed_band_fit_cached(directory, name, x, y, policy):
    """缓存仅属于当前数据哈希/分组/实验策略，不写正式模型缓存。"""
    path = os.path.join(directory, name + ".pkl")
    if os.path.isfile(path):
        with open(path, "rb") as stream:
            return pickle.load(stream)
    print("MODEL_CHECK_TRAINING " + name, flush=True)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model = _FixedBandSpectrumGPR().fit(x, y, policy["optimizer_maxiter"])
    payload = {"model": model, "seconds": time.perf_counter() - start,
               "warnings": sorted(set(str(w.message) for w in captured)),
               "kernel": str(model.model.kernel_)}
    with open(path + ".pending", "wb") as stream:
        pickle.dump(payload, stream)
    os.replace(path + ".pending", path)
    return payload


def _fixed_band_selected_prediction(mean, std, target):
    # 只按预测值选模态，绝不根据测试集真实频率选择预测阶次。
    selected = np.argmin(np.abs(mean - target), axis=1)
    row = np.arange(len(mean))
    return selected, mean[row, selected], std[row, selected]


def run_fixed_band_model_check(target, additional, dataset_csv):
    """独立几何留出验证、校准和目标模型对照；此入口没有 FE 调用。"""
    policy = dict(config.FIXED_BAND_MODEL_CHECK_POLICY)
    targets = list(dict.fromkeys([float(target)] + [float(t) for t in additional]))
    if not targets or any(not np.isfinite(t) or t <= 0 for t in targets):
        raise ValueError("TARGET_FREQUENCIES_MUST_BE_FINITE_POSITIVE")
    rows, x, y = _fixed_band_training_data(dataset_csv)
    with open(dataset_csv, "rb") as stream:
        dataset_hash = hashlib.sha256(stream.read()).hexdigest()
    with open(__file__, "rb") as stream:
        code_hash = hashlib.sha256(stream.read()).hexdigest()
    identity = {"Dataset_SHA256": dataset_hash, "Code_SHA256": code_hash,
                "Policy": policy, "Targets": targets}
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = ensure_directory(os.path.join(BASE_OUTPUT, "fixed_band_15mm", "model_check", token))
    cache = ensure_directory(os.path.join(output, "experimental_cache"))
    groups = _fixed_band_geometry_groups(x, policy["near_duplicate_radius"])
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=policy["test_fraction"],
        random_state=policy["seed"]).split(x, groups=groups))
    fit_rel, cal_rel = next(GroupShuffleSplit(n_splits=1,
        test_size=policy["calibration_fraction_of_development"],
        random_state=policy["seed"] + 1).split(x[development], groups=groups[development]))
    fit, cal = development[fit_rel], development[cal_rel]
    assert not (set(groups[fit]) & set(groups[cal]) or set(groups[fit]) & set(groups[test]) or
                set(groups[cal]) & set(groups[test]))
    split_rows = [{"Sample_ID": rows[i]["Sample_ID"], "Geometry_Group": int(groups[i]),
                   "Partition": name} for name, indices in (("FIT", fit), ("CALIBRATION", cal),
                                                            ("TEST", test)) for i in indices]
    write_csv(os.path.join(output, "geometry_partitions.csv"), split_rows)
    save_json(os.path.join(output, "run_manifest.json"), dict(identity, Status="RUNNING",
        Rows=len(rows), Groups=len(set(groups)), Fit_Count=len(fit), Calibration_Count=len(cal),
        Test_Count=len(test), Abaqus_Job_Submitted=False))
    baseline = _fixed_band_fit_cached(cache, "shared_24_gpr", x[fit], y[fit], policy)
    with threadpool_limits(limits=1):
        base_cal = baseline["model"].predict(x[cal])
        base_test = baseline["model"].predict(x[test])
    existing_path = os.path.join(BASE_OUTPUT, "fixed_band_15mm", "generated",
                                 "fixed_band_15mm_exploratory_candidates.csv")
    existing = []
    if os.path.isfile(existing_path):
        with open(existing_path, encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    benchmark, per_mode, reassessed, predictions, recommendations = [], [], [], [], []
    training_details = [{"Model": "SHARED_24_GPR", "Seconds": baseline["seconds"],
                         "Kernel": baseline["kernel"], "Warnings": baseline["warnings"]}]
    for ft in targets:
        support = np.sum(np.abs(y[fit] - ft) <= ft * policy["target_training_band_fraction"], axis=0)
        modes = np.flatnonzero(support >= policy["minimum_mode_support"])
        if not len(modes):
            raise RuntimeError("NO_TRAINING_SUPPORTED_MODE_FOR_TARGET:%s" % ft)
        local_models = []
        local_seconds = 0.0
        for mode in modes:
            # 只用训练折的频率选目标相关样本，任何校准/测试频率均不可参与。
            order = np.argsort(np.abs(y[fit, mode] - ft), kind="stable")
            local = fit[order[:min(policy["local_training_rows"], len(fit))]]
            label = "target_%s_mode_%02d" % (str(ft).replace(".", "p"), mode + 1)
            payload = _fixed_band_fit_cached(cache, label, x[local], y[local, mode:mode + 1], policy)
            local_models.append(payload["model"])
            local_seconds += payload["seconds"]
            training_details.append({"Model": label, "Train_Count": len(local),
                "Seconds": payload["seconds"], "Kernel": payload["kernel"],
                "Warnings": payload["warnings"]})

        def local_predict(vectors):
            with threadpool_limits(limits=1):
                values = [model.predict(vectors) for model in local_models]
            return (np.column_stack([v[0][:, 0] for v in values]),
                    np.column_stack([v[1][:, 0] for v in values]))

        method_scores = []
        for name, indices, cp, tp, predictor, seconds in (
                ("SHARED_24_GPR", np.arange(y.shape[1]), base_cal, base_test,
                 baseline["model"].predict, baseline["seconds"]),
                ("TARGET_LOCAL_PER_MODE_GPR", modes, local_predict(x[cal]), local_predict(x[test]),
                 local_predict, local_seconds)):
            cmode, cm, cs = _fixed_band_selected_prediction(*cp, ft)
            q = _fixed_band_calibration_factor(y[cal, indices[cmode]], cm, cs,
                                               policy["coverage"], policy["minimum_calibration_rows"])
            tmode, tm, ts = _fixed_band_selected_prediction(*tp, ft)
            truth = y[test, indices[tmode]]
            absolute = np.abs(tm - truth)
            relative = absolute / truth * 100
            actual_detuning = np.abs(truth - ft) / ft * 100
            upper = (np.abs(tm - ft) + q * ts) / ft * 100
            row = {"Target_Frequency_Hz": ft, "Model": name, "Test_Count": len(test),
                "Fit_Geometries": len(fit), "Calibration_Geometries": len(cal),
                "Evaluated_Modes": ";".join(str(i + 1) for i in indices),
                "Selected_Mode_MAPE_Percent": float(np.mean(relative)),
                "Selected_Mode_MAE_Hz": float(np.mean(absolute)),
                "Selected_Mode_P90_Percent": float(np.percentile(relative, 90)),
                "Selected_Mode_Max_Error_Percent": float(np.max(relative)),
                "Raw_95_Coverage_Percent": float(np.mean(absolute <= 1.96 * ts) * 100),
                "Calibrated_95_Coverage_Percent": float(np.mean(absolute <= q * ts) * 100),
                "Calibrated_Mean_Interval_Width_Hz": float(np.mean(2 * q * ts)),
                "Calibration_Multiplier": q,
                "Model_Screened_Test_Count": int(np.sum(upper <= 2)),
                "Screened_Test_False_Accept_Count": int(np.sum((upper <= 2) & (actual_detuning > 2))),
                "Selected_Mode_Target_Hit_Count": int(np.sum(actual_detuning <= 2)),
                "Any_Observed_24_Mode_Target_Hit_Count": int(np.sum(np.min(np.abs(y[test] - ft), axis=1) / ft <= .02)),
                "Predicted_Spectrum_Inversion_Percent": float(np.mean(np.any(np.diff(tp[0], axis=1) < 0, axis=1)) * 100),
                "Training_Seconds": seconds, "Physical_Mode_Identity": "UNCONFIRMED",
                "Protocol": "FIXED_GROUP_HOLDOUT_NOT_CROSS_VALIDATION"}
            benchmark.append(row)
            method_scores.append(row)
            for column, mode in enumerate(indices):
                err = np.abs(tp[0][:, column] - y[test, mode])
                rel = err / y[test, mode] * 100
                per_mode.append({"Target_Frequency_Hz": ft, "Model": name, "Mode": int(mode + 1),
                    "Count": len(test), "MAPE_Percent": float(np.mean(rel)),
                    "MAE_Hz": float(np.mean(err)), "P90_Percent": float(np.percentile(rel, 90))})
            for k, i in enumerate(test):
                predictions.append({"Target_Frequency_Hz": ft, "Model": name,
                    "Sample_ID": rows[i]["Sample_ID"], "Selected_Sorted_Mode": int(indices[tmode[k]] + 1),
                    "Predicted_Hz": float(tm[k]), "Actual_Same_Mode_Hz": float(truth[k]),
                    "STD_Hz": float(ts[k]), "Calibrated_Radius_Hz": float(q * ts[k]),
                    "Actual_Target_Detuning_Percent": float(actual_detuning[k])})
            candidates = [c for c in existing if float(c["Target_Frequency_Hz"]) == ft]
            if candidates:
                vx = np.array([[float(c[key]) for key in FIXED_BAND_9D] for c in candidates])
                with threadpool_limits(limits=1):
                    pm, ps = predictor(vx)
                selected, mean, std = _fixed_band_selected_prediction(pm, ps, ft)
                for k, c in enumerate(candidates):
                    bound = (abs(mean[k] - ft) + q * std[k]) / ft * 100
                    reassessed.append({"Candidate_ID": c["Candidate_ID"], "Target_Frequency_Hz": ft,
                        "Model": name, "Sorted_Mode": int(indices[selected[k]] + 1),
                        "Predicted_Frequency_Hz": float(mean[k]), "STD_Hz": float(std[k]),
                        "Calibrated_Radius_Hz": float(q * std[k]), "Upper_Error_Percent": float(bound),
                        "Screen_Pass": bool(bound <= 2), "FE_Status": "NOT_RUN",
                        "Evaluation_Status": "DEVELOPMENT_RECHECK_NOT_INDEPENDENT_TEST",
                        "Original_Prediction_Preserved": True})
        b, candidate = method_scores
        recommended = (candidate["Selected_Mode_MAPE_Percent"] <= .9 * b["Selected_Mode_MAPE_Percent"] and
                       candidate["Selected_Mode_P90_Percent"] <= b["Selected_Mode_P90_Percent"] and
                       candidate["Calibrated_95_Coverage_Percent"] >= 90)
        recommendations.append({"Target_Frequency_Hz": ft, "Decision":
            "RECOMMENDED_EXPERIMENTAL_ONLY" if recommended else "KEEP_EXISTING_BASELINE",
            "Production_Model_Replaced": False})
        # 每个目标结束即保存，不必因中断重做已经完成的训练。
        for filename, values in (("model_benchmark.csv", benchmark), ("per_mode_errors.csv", per_mode),
                ("candidate_recheck.csv", reassessed), ("heldout_predictions.csv", predictions)):
            write_csv(os.path.join(output, filename), values)
        save_json(os.path.join(output, "training_details.json"), training_details)
        print("MODEL_CHECK_TARGET_COMPLETED %s" % ft, flush=True)
    report = ["# 固定15 mm反向设计模型检查", "",
        "本次零Abaqus计算。原数据库、原候选、原验证结果与生产模型不覆盖。",
        "", "## 数据与协议", "",
        "有效几何 %d；近重复合并后 %d 组。训练 %d、校准 %d、测试 %d。" %
        (len(rows), len(set(groups)), len(fit), len(cal), len(test)),
        "固定种子分组留出，不是多折Group-CV。两模型使用相同测试几何。",
        "共享24输出GPR和目标相关逐阶局部GPR均采用有限迭代预算；不能与历史全量拟合误差直接比较。",
        "目标相关模型只依据训练集选择有支持的排序阶次和频率邻域；不使用测试频率选邻域。",
        "超参数及标准化仅在训练子集拟合。校准集仅估计同阶残差/STD分位数；测试集只评价。",
        "所报主误差为模型选择的最接近目标排序阶次的预测误差，不是任意阶命中率。",
        "校准区间只针对该选阶规则的边际误差；不能保证优化器选出的新点有95%覆盖率。",
        "物理模态身份未确认；无MAC证据。比较结果用于开发，不是最终独立认证。", "",
        "## 实测对照", "",
        "|目标Hz|模型|选阶MAPE %|P90 %|原区间覆盖 %|校准覆盖 %|校准平均宽度Hz|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in benchmark:
        report.append("|%g|%s|%.3f|%.3f|%.2f|%.2f|%.3f|" % (r["Target_Frequency_Hz"],
            r["Model"], r["Selected_Mode_MAPE_Percent"], r["Selected_Mode_P90_Percent"],
            r["Raw_95_Coverage_Percent"], r["Calibrated_95_Coverage_Percent"], r["Calibrated_Mean_Interval_Width_Hz"]))
    report += ["", "## 结论与操作", ""]
    report += ["- %g Hz：%s；生产模型不替换。" % (r["Target_Frequency_Hz"], r["Decision"]) for r in recommendations]
    report += ["", "如果区间变宽但覆盖提高，说明原标准差低估误差，不应放宽2%真实验收门槛。",
        "已有候选复核不等于新FE验证，也不是独立测试：它们曾由全数据库模型选择。",
        "下次运行 `python main.py --design-mode MODEL_CHECK` 会按数据/代码哈希复用实验缓存。",
        "普通 GENERATE 仍保留原模型；修正归一化搜索和优化器退出检查，不以预测命中冒充FE成功。",
        "保留历史验证集；只有显式开发集才能在后续授权后用于修正，最终测试集必须另行冻结。",
        "", "## 文件", ""]
    report += ["- " + name for name in ("model_benchmark.csv", "per_mode_errors.csv", "candidate_recheck.csv",
        "heldout_predictions.csv", "geometry_partitions.csv", "run_manifest.json", "training_details.json")]
    report_path = os.path.join(output, "MODEL_CHECK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    result = dict(identity, Status="COMPLETED", Output_Directory=output, Report=report_path,
        Recommendations=recommendations, Abaqus_Job_Submitted=False, Production_Model_Replaced=False)
    save_json(os.path.join(output, "run_manifest.json"), dict(result, Rows=len(rows),
        Groups=len(set(groups)), Fit_Count=len(fit), Calibration_Count=len(cal), Test_Count=len(test)))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _fixed_band_load_checked_baseline(dataset_csv):
    """复用已完成检查的训练模型与原分区，严格核对数据库及实际训练数组。"""
    rows, x, y = _fixed_band_training_data(dataset_csv)
    with open(dataset_csv, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    root = os.path.join(BASE_OUTPUT, "fixed_band_15mm", "model_check")
    matches = []
    if os.path.isdir(root):
        for child in os.listdir(root):
            directory = os.path.join(root, child)
            manifest_path = os.path.join(directory, "run_manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            manifest = read_json(manifest_path)
            if manifest.get("Status") == "COMPLETED" and manifest.get("Dataset_SHA256") == digest:
                matches.append((os.path.getmtime(manifest_path), directory, manifest))
    if not matches:
        raise RuntimeError("RUN_MODEL_CHECK_FIRST_FOR_CURRENT_DATABASE")
    _, source, manifest = max(matches)
    by_id = {r["Sample_ID"]: i for i, r in enumerate(rows)}
    if len(by_id) != len(rows):
        raise RuntimeError("DUPLICATE_TRAINING_SAMPLE_IDS")
    partitions = {name: [] for name in ("FIT", "CALIBRATION", "TEST")}
    groups = {name: set() for name in partitions}
    with open(os.path.join(source, "geometry_partitions.csv"), encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            partitions[row["Partition"]].append(by_id[row["Sample_ID"]])
            groups[row["Partition"]].add(row["Geometry_Group"])
    flat = sum(partitions.values(), [])
    if sorted(flat) != list(range(len(rows))) or any(not v for v in partitions.values()):
        raise RuntimeError("INCOMPLETE_OR_DUPLICATE_PARTITIONS")
    if (groups["FIT"] & groups["CALIBRATION"] or groups["FIT"] & groups["TEST"] or
            groups["CALIBRATION"] & groups["TEST"]):
        raise RuntimeError("GEOMETRY_PARTITION_LEAKAGE")
    with open(os.path.join(source, "experimental_cache", "shared_24_gpr.pkl"), "rb") as stream:
        payload = pickle.load(stream)
    baseline = payload["model"]
    fit = partitions["FIT"]
    if (not np.allclose(baseline.input_scaler.inverse_transform(baseline.model.X_train_), x[fit],
                        rtol=1.0e-10, atol=1.0e-12) or
            not np.allclose(baseline.output_scaler.inverse_transform(baseline.model.y_train_), y[fit],
                            rtol=1.0e-10, atol=1.0e-9)):
        raise RuntimeError("CACHED_BASELINE_TRAINING_DATA_MISMATCH")
    return rows, x, y, baseline, partitions, source, manifest


def _fixed_band_anchor_metrics(name, target, cp, tp, y_cal, y_test, policy):
    """同一选阶规则下评价原模型和增量模型；测试真实频率不参与选阶。"""
    cmode, cm, cs = _fixed_band_selected_prediction(*cp, target)
    q = _fixed_band_calibration_factor(y_cal[np.arange(len(y_cal)), cmode], cm, cs,
                                       policy["coverage"], policy["minimum_calibration_rows"])
    mode, mean, scale = _fixed_band_selected_prediction(*tp, target)
    actual = y_test[np.arange(len(y_test)), mode]
    error = np.abs(actual - mean)
    relative = error / actual * 100
    upper = (np.abs(mean - target) + q * scale) / target * 100
    row = {"Target_Frequency_Hz": target, "Model": name, "Test_Count": len(y_test),
        "Selected_Mode_MAPE_Percent": float(np.mean(relative)),
        "Selected_Mode_MAE_Hz": float(np.mean(error)),
        "Selected_Mode_P90_Percent": float(np.percentile(relative, 90)),
        "Selected_Mode_Max_Error_Percent": float(np.max(relative)),
        "Calibration_Multiplier": q,
        "Calibrated_95_Coverage_Percent": float(np.mean(error <= q * scale) * 100),
        "Calibrated_Mean_Interval_Width_Hz": float(np.mean(2 * q * scale)),
        "Nominal_Screened_Test_Count": int(np.sum(upper <= 2)),
        "False_Accept_Count": int(np.sum((upper <= 2) & (np.abs(actual - target) / target > .02))),
        "Any_Actual_Mode_Target_Hit_Count": int(np.sum(np.min(np.abs(y_test - target), axis=1) / target <= .02)),
        "Predicted_Spectrum_Inversion_Percent": float(np.mean(np.any(np.diff(tp[0], axis=1) < 0, axis=1)) * 100),
        "Nonpositive_Prediction_Count": int(np.sum(tp[0] <= 0)),
        "Physical_Mode_Identity": "UNCONFIRMED",
        "Test_Role": "PREVIOUSLY_INSPECTED_DEVELOPMENT_HOLDOUT_NOT_NEW_BLIND_TEST"}
    return row, mode, mean, scale, actual, upper


def _fixed_band_local_validation_plan(candidates, targets, rows, x, dataset_hash):
    """只冻结已有候选的下一轮少量验证清单，不生成DOE或调用求解器。"""
    lookup = {r["Sample_ID"]: r for r in rows}
    span = np.maximum(np.ptp(x, axis=0), 1.0e-12)
    ledger = os.path.join(BASE_OUTPUT, "fixed_band_15mm", "generated", "fixed_band_15mm_abaqus_validation.csv")
    used = set()
    if os.path.isfile(ledger):
        with open(ledger, encoding="utf-8-sig", newline="") as stream:
            used = {r["Candidate_ID"] for r in csv.DictReader(stream)}
    plan = []
    for target in targets:
        eligible = []
        for c in candidates:
            if float(c["Target_Frequency_Hz"]) != target or c["Candidate_ID"] in used:
                continue
            if any(abs(float(c.get(k, 0)) - .015) > 1.0e-12 for k in ("Damping_Inward_X_m", "Damping_Inward_Y_m")):
                continue
            anchor = lookup.get(c["Anchor_Sample_ID"])
            if anchor is None:
                continue
            vector = np.array([float(c[k]) for k in FIXED_BAND_9D])
            av = np.array([anchor["Parameters"][k] for k in FIXED_BAND_9D])
            valid, _, _ = _fixed_band_geometry_valid(vector, x.min(axis=0), x.max(axis=0))
            distance = float(np.linalg.norm((vector - av) / span))
            nearest = float(np.min(np.linalg.norm((x - vector) / span, axis=1)))
            if (valid and nearest >= FIXED_BAND_TARGET_POLICY["minimum_new_distance"] and
                    distance <= FIXED_BAND_TARGET_POLICY["local_trust_radius"]):
                eligible.append((distance, float(c["Predicted_STD_Hz"]) / target, c, anchor, vector))
        eligible.sort(key=lambda item: (item[0], item[1], item[2]["Candidate_ID"]))
        if not eligible:
            continue
        chosen = [eligible[0]]
        # 第二个是预先锁定的复核点，优先与开发点保持几何差异，而非看未来FE结果选取。
        if len(eligible) > 1:
            chosen.append(max(eligible[1:], key=lambda item: np.linalg.norm((item[4] - chosen[0][4]) / span)))
        for number, (distance, _, candidate, anchor, vector) in enumerate(chosen):
            parameters = dict(zip(FIXED_BAND_9D, map(float, vector)))
            record = {"Target_Frequency_Hz": target, "Candidate_ID": candidate["Candidate_ID"],
                "Plan_Role": "DEVELOPMENT" if number == 0 else "LOCKED_FUTURE_CHECK",
                "Source": "EXISTING_FROZEN_CANDIDATE_NO_NEW_DOE", "Dataset_SHA256": dataset_hash,
                "Parameter_SHA256": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
                # 沿用现有单候选验证入口的字段名，未来无需转换这份清单。
                "Prediction_Source": "ORIGINAL_FROZEN_GENERATION_NOT_ANCHOR_DELTA",
                "Predicted_Frequency_Hz": float(candidate["Predicted_Frequency_Hz"]),
                "Predicted_Mode_Index": int(candidate["Predicted_Mode_Index"]),
                "Predicted_STD_Hz": float(candidate["Predicted_STD_Hz"]),
                "Nearest_Training_Distance": float(np.min(np.linalg.norm((x - vector) / span, axis=1))),
                "Anchor_Sample_ID": anchor["Sample_ID"], "Anchor_Distance": distance,
                "Anchor_Mode": int(candidate["Anchor_Mode_Index"]),
                "Anchor_Actual_Frequency_Hz": anchor["Spectrum_Hz"][int(candidate["Anchor_Mode_Index"]) - 1],
                "Damping_Inward_X_m": .015, "Damping_Inward_Y_m": .015,
                "Geometry_Precheck": "ALGEBRA_ONLY_PASS_NOT_MESH_OR_FE", "FE_Status": "NOT_SUBMITTED",
                "Actual_Frequency_Hz": "", "Actual_Target_Error_Percent": "", "Requires_FE_Authorization": True}
            record.update(parameters)
            record.update({"Anchor_" + k: anchor["Parameters"][k] for k in FIXED_BAND_9D})
            plan.append(record)
    return plan


def run_fixed_band_anchor_check(target, additional, dataset_csv):
    """复用旧模型进行一次预先定义的锚点增量对照，不训练、不提交FE。"""
    targets = list(dict.fromkeys([float(target)] + [float(t) for t in additional]))
    if any(not np.isfinite(t) or t <= 0 for t in targets):
        raise ValueError("TARGET_FREQUENCIES_MUST_BE_FINITE_POSITIVE")
    rows, x, y, baseline, partitions, source, previous = _fixed_band_load_checked_baseline(dataset_csv)
    fit, cal, test = [np.array(partitions[k]) for k in ("FIT", "CALIBRATION", "TEST")]
    policy = previous["Policy"]
    with open(__file__, "rb") as stream:
        code_hash = hashlib.sha256(stream.read()).hexdigest()
    candidate_path = os.path.join(BASE_OUTPUT, "fixed_band_15mm", "generated", "fixed_band_15mm_exploratory_candidates.csv")
    ledger_path = os.path.join(BASE_OUTPUT, "fixed_band_15mm", "generated", "fixed_band_15mm_abaqus_validation.csv")
    input_hashes = {}
    for label, path in (("Existing_Candidate_SHA256", candidate_path), ("Validation_Ledger_SHA256", ledger_path)):
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                input_hashes[label] = hashlib.sha256(stream.read()).hexdigest()
        else:
            input_hashes[label] = None
    identity = {"Dataset_SHA256": previous["Dataset_SHA256"], "Code_SHA256": code_hash,
        "Source_Model_Check": source, "Targets": targets, "Method": "NEAREST_TRAINING_ANCHOR_GPR_INCREMENT",
        "Anchor_Distance_Scaling": "TRAINING_FEATURE_RANGE_ONLY",
        "Local_Trust_Radius": FIXED_BAND_TARGET_POLICY["local_trust_radius"]}
    # 候选或台账改变时另建结果目录，禁止覆盖已经锁定的未来复核点。
    identity.update(input_hashes)
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = ensure_directory(os.path.join(BASE_OUTPUT, "fixed_band_15mm", "anchor_delta", token))
    save_json(os.path.join(output, "run_manifest.json"), dict(identity, Status="RUNNING", Abaqus_Job_Submitted=False))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        corrected = _FixedBandAnchorDelta(baseline, x[fit], y[fit], [rows[i]["Sample_ID"] for i in fit])
        original_cal, original_test = baseline.predict(x[cal]), baseline.predict(x[test])
        delta_cal, delta_test = corrected.predict(x[cal]), corrected.predict(x[test])
        anchor_error = float(np.max(np.abs(corrected.predict(x[fit[:20]])[0] - y[fit[:20]])))
    if anchor_error > 1.0e-7:
        raise RuntimeError("ANCHOR_INTERPOLATION_REGRESSION_FAILED")
    cal_distance, _ = corrected.anchors(x[cal])
    test_distance, test_anchor = corrected.anchors(x[test])
    radius = identity["Local_Trust_Radius"]
    local_cal = int(np.sum(cal_distance <= radius))
    local_test = int(np.sum(test_distance <= radius))
    # 没有近锚点真实验证对时，不把远距离校准外推成近距离设计的95%保证。
    local_support = local_cal >= policy["minimum_calibration_rows"] and local_test >= policy["minimum_calibration_rows"]
    benchmark, detail, reassessed, decisions = [], [], [], []
    candidates = []
    if os.path.isfile(candidate_path):
        with open(candidate_path, encoding="utf-8-sig", newline="") as stream:
            candidates = list(csv.DictReader(stream))
    for ft in targets:
        scores = []
        for name, cp, tp, model in (("SHARED_24_GPR", original_cal, original_test, baseline),
                                     ("ANCHOR_DELTA_GPR", delta_cal, delta_test, corrected)):
            score, mode, mean, scale, actual, upper = _fixed_band_anchor_metrics(name, ft, cp, tp, y[cal], y[test], policy)
            score.update(Local_Calibration_Count=local_cal, Local_Test_Count=local_test,
                         Local_Calibration_Supported=local_support)
            benchmark.append(score)
            scores.append(score)
            for k, i in enumerate(test):
                detail.append({"Target_Frequency_Hz": ft, "Model": name, "Sample_ID": rows[i]["Sample_ID"],
                    "Anchor_Sample_ID": rows[fit[test_anchor[k]]]["Sample_ID"],
                    "Anchor_Distance": float(test_distance[k]), "Sorted_Mode": int(mode[k] + 1),
                    "Predicted_Hz": float(mean[k]), "Actual_Same_Mode_Hz": float(actual[k]),
                    "Same_Mode_Error_Percent": float(abs(mean[k] - actual[k]) / actual[k] * 100),
                    "Calibrated_Radius_Hz": float(score["Calibration_Multiplier"] * scale[k]),
                    "Nominal_Upper_Error_Percent": float(upper[k])})
            subset = [c for c in candidates if float(c["Target_Frequency_Hz"]) == ft]
            if subset:
                vectors = np.array([[float(c[key]) for key in FIXED_BAND_9D] for c in subset])
                with threadpool_limits(limits=1):
                    cmode, cm, cs = _fixed_band_selected_prediction(*model.predict(vectors), ft)
                distance, anchors = corrected.anchors(vectors)
                for k, c in enumerate(subset):
                    bound = (abs(cm[k] - ft) + score["Calibration_Multiplier"] * cs[k]) / ft * 100
                    reassessed.append({"Candidate_ID": c["Candidate_ID"], "Target_Frequency_Hz": ft,
                        "Model": name, "Anchor_Sample_ID": rows[fit[anchors[k]]]["Sample_ID"],
                        "Anchor_Distance": float(distance[k]), "Sorted_Mode": int(cmode[k] + 1),
                        "Predicted_Hz": float(cm[k]), "Nominal_Calibrated_Upper_Error_Percent": float(bound),
                        "Nominal_Screen_Pass": bool(bound <= 2),
                        "Local_Calibration_Supported": local_support,
                        "Reliable_2Pct_Claim": False, "FE_Status": "NOT_RUN",
                        "Status": "EXPERIMENTAL_RECHECK_NOT_FE_VALIDATED"})
        b, d = scores
        improved = (d["Selected_Mode_MAPE_Percent"] <= .9 * b["Selected_Mode_MAPE_Percent"] and
                    d["Selected_Mode_P90_Percent"] <= b["Selected_Mode_P90_Percent"] and
                    d["Calibrated_95_Coverage_Percent"] >= 90 and d["Nonpositive_Prediction_Count"] == 0)
        decisions.append({"Target_Frequency_Hz": ft, "Meets_Predeclared_Improvement_Gate": improved,
            "Decision": "EXPERIMENTAL_ONLY_REQUIRES_LOCAL_EVIDENCE" if improved else "KEEP_EXISTING_BASELINE",
            "Production_Model_Replaced": False})
    for filename, values in (("anchor_delta_benchmark.csv", benchmark),
                             ("anchor_delta_heldout_predictions.csv", detail),
                             ("anchor_delta_candidate_recheck.csv", reassessed)):
        write_csv(os.path.join(output, filename), values)
    plan = _fixed_band_local_validation_plan(candidates, targets, rows, x, previous["Dataset_SHA256"])
    write_csv(os.path.join(output, "local_validation_priority.csv"), plan)
    report = ["# 真实频率锚点与局部频率增量修正", "",
        "公式：预测f(x) = 训练锚点实际f(a) + 原GPR预测f(x) - 原GPR预测f(a)。",
        "锚点只按训练集九维几何距离选择，不读取校准/测试真实频率。24阶排序频率全部保留，不等同物理模态追踪。",
        "增量协方差用于构造误差尺度，再用独立校准集残差校准；不将它直接称作可靠95%区间。", "",
        "## 对照协议", "",
        "复用上一轮%d/%d/%d训练、校准、测试几何及已训练GPR，无重新训练或FE。" % (len(fit), len(cal), len(test)),
        "该测试集已在前轮查看，因此本轮是开发对照，不是新增盲测。没有用测试集拟合、选择锚点或校准区间。",
        "预先规定实验改善门槛：MAPE至少降低10%、P90不恶化、校准覆盖至少90%、没有非正预测；不据此替换生产模型。",
        "近锚点支持检查采用当前生成半径%.3f；校准集%d点、测试集%d点落在该范围。" % (radius, local_cal, local_test),
        "锚点重构最大误差：%.6g Hz（前20个训练锚点数值回归，不是泛化精度）。" % anchor_error, "",
        "## 实测结果", "", "|目标Hz|模型|MAPE %|P90 %|校准覆盖 %|平均区间宽度Hz|",
        "|---|---|---:|---:|---:|---:|"]
    for r in benchmark:
        report.append("|%g|%s|%.3f|%.3f|%.2f|%.3f|" % (r["Target_Frequency_Hz"], r["Model"],
            r["Selected_Mode_MAPE_Percent"], r["Selected_Mode_P90_Percent"],
            r["Calibrated_95_Coverage_Percent"], r["Calibrated_Mean_Interval_Width_Hz"]))
    report += ["", "## 决定", ""]
    report += ["- %g Hz：%s。" % (r["Target_Frequency_Hz"], r["Decision"]) for r in decisions]
    report += ["", "原生成模型、正式候选、FE台账保持不变；本轮不提交Abaqus。",
        "若近锚点校准/测试不足，即使候选名义上通过2%区间筛选，也不允许声称可靠合格。",
        "如果该方法没有稳定改善，应停止继续堆叠预测修正，转而准备少量具有独立对照的目标局部开发样本；需另行授权FE。",
        "", "## 准备的最小验证清单", "",
        "local_validation_priority.csv仅复用现有候选，冻结%d个构型；不新增DOE、不提交作业。" % len(plan),
        "每个目标最多1个开发点和1个锁定复核点。先评估开发点，再决定是否授权复核；复核点不得用于拟合或选参数。",
        "每行保存完整九维参数、固定15mm带宽、已有真实锚点参数和频率、原预测及参数哈希。真实新频率保持空白。",
        "这批最多6次未来FE只是低成本局部诊断，不足以校准95%覆盖率或宣称全域达到2%；本轮FE次数为0。",
        "", "运行：在项目目录执行 `python main.py --design-mode ANCHOR_CHECK`，建议使用Anaconda解释器。",
        "", "输出：anchor_delta_benchmark.csv、anchor_delta_heldout_predictions.csv、anchor_delta_candidate_recheck.csv、local_validation_priority.csv、run_manifest.json。"]
    path = os.path.join(output, "ANCHOR_DELTA_REPORT.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    result = dict(identity, Status="COMPLETED", Elapsed_Seconds=time.perf_counter() - start,
        Recommendations=decisions, Local_Calibration_Count=local_cal, Local_Test_Count=local_test,
        Output_Directory=output, Report=path, Abaqus_Job_Submitted=False, New_GPR_Fits=0,
        Production_Model_Replaced=False, Planned_Future_Validation_Count=len(plan))
    save_json(os.path.join(output, "run_manifest.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _fixed_band_geometry_valid(vector, lower, upper, shell_size_bounds_mm=None):
    """生成阶段的代数约束；并保留距离边界的安全余量。"""
    p = dict(zip(FIXED_BAND_9D, vector))
    if np.any(vector < lower) or np.any(vector > upper):
        return False, "OUT_OF_DATA_BOUNDS", None
    if p["Rou_2"] - p["Rou_3"] * math.cos(p["theta"]) < 5.0e-4:
        return False, "ROU2_RADICAND_MARGIN_TOO_SMALL", None
    if p["Rou_4"] - p["Rou_3"] * math.sin(p["theta"]) < 5.0e-4:
        return False, "ROU4_RADICAND_MARGIN_TOO_SMALL", None
    if p["platform_height"] <= p["cutoff_thickness"]:
        return False, "PLATFORM_NOT_ABOVE_CUTOFF", None
    try:
        damping = fixed_band_width_to_rx_ry(p, 0.015, 0.015)
    except (ValueError, KeyError):
        return False, "FIXED_15MM_DAMPING_NOT_FEASIBLE", None
    full = dict(config.FIXED_BASELINE_PARAMETERS)
    full.update(p)
    dimensions = shell_outer_dimensions(full)
    if shell_size_bounds_mm is not None:
        try:
            x_min, x_max = map(float, shell_size_bounds_mm["length_x_mm"])
            y_min, y_max = map(float, shell_size_bounds_mm["width_y_mm"])
        except (KeyError, TypeError, ValueError):
            return False, "INVALID_ABH_SHELL_SIZE_CONSTRAINT", None
        # A zero lower bound deliberately means "no lower size limit".
        if not (0.0 <= x_min <= x_max and 0.0 <= y_min <= y_max):
            return False, "INVALID_ABH_SHELL_SIZE_CONSTRAINT", None
        if not x_min <= dimensions["ABH_Shell_Overall_Length_X_m"] * 1000.0 <= x_max:
            return False, "ABH_SHELL_LENGTH_X_OUT_OF_RANGE", None
        if not y_min <= dimensions["ABH_Shell_Overall_Width_Y_m"] * 1000.0 <= y_max:
            return False, "ABH_SHELL_WIDTH_Y_OUT_OF_RANGE", None
    damping.update(dimensions)
    return True, "OK", damping


def _fixed_band_generate_for_target(target, model, train_rows, train_x, train_y,
                                    lower, upper, count, seed, excluded_ids=None,
                                    shell_size_bounds_mm=None):
    """Generate only locally supported, database-external target candidates.

    A global sorted-mode surrogate can report an artificial exact target while
    being uncertain about the physical spectrum.  Generation therefore starts
    from samples whose *actual Abaqus spectrum* already contains the target,
    permits only a small normalised move, and rejects high-uncertainty output.
    """
    rng = np.random.RandomState(seed)
    width = upper - lower
    if np.any(width <= 0) or not np.all(np.isfinite(width)):
        raise ValueError("DEGENERATE_GEOMETRY_SEARCH_RANGE")
    rejections = []
    excluded_ids = set(excluded_ids or ())

    def reject(reason, anchor_row=None, anchor_mode=None, anchor_frequency=None,
               anchor_detuning=None, **metrics):
        row = {"Target_Frequency_Hz": float(target), "Rejection_Reason": reason,
               "Anchor_Sample_ID": (train_rows[anchor_row]["Sample_ID"]
                                    if anchor_row is not None else ""),
               "Anchor_Mode_Index": (anchor_mode + 1 if anchor_mode is not None else ""),
               "Anchor_Mode_Frequency_Hz": anchor_frequency if anchor_frequency is not None else "",
               "Anchor_Detuning_Percent": anchor_detuning if anchor_detuning is not None else ""}
        row.update(metrics)
        rejections.append(row)

    anchors = []
    for row, spectrum in enumerate(train_y):
        # 外轮廓约束是本轮设计的硬条件。不能让一个本身无法制造的历史
        # 样本充当局部反向优化锚点，否则优化器会反复走向超尺寸区域。
        # 这不会删除训练数据；它只限制新构型从哪些真实可行点附近出发。
        anchor_valid, _, _ = _fixed_band_geometry_valid(
            train_x[row], lower, upper, shell_size_bounds_mm)
        if not anchor_valid:
            continue
        mode = int(np.argmin(np.abs(spectrum - target)))
        detuning = float(abs(spectrum[mode] - target) / target * 100.0)
        if detuning <= FIXED_BAND_TARGET_POLICY["anchor_detuning_percent"]:
            anchors.append((detuning, row, mode))
    anchors.sort()
    if not anchors:
        reject("NO_REAL_TARGET_ANCHOR_WITHIN_1_PERCENT")
        return {"candidates": [], "rejections": rejections,
                "diagnostic": {"Target_Frequency_Hz": float(target), "Anchor_Count": 0,
                               "Optimization_Attempts": 0, "Valid_Proposals": 0,
                               "Accepted_Candidates": 0}}
    proposals = []
    # 严格 UQ 门槛拒绝、但仍符合几何/可信域/预测频差条件的点，单独
    # 保存为探索候选。它们不可被称为“设计成功”，只能由独立 Abaqus 判定。
    exploratory = []
    geometry_external_proposals = 0
    radius = float(FIXED_BAND_TARGET_POLICY["local_trust_radius"])
    minimum_new_distance = float(FIXED_BAND_TARGET_POLICY["minimum_new_distance"])
    optimization_attempts = 0
    max_anchor_attempts = int(FIXED_BAND_TARGET_POLICY.get(
        "max_anchor_attempts", 80))
    for anchor_detuning, row, selected_mode in anchors[:min(max_anchor_attempts, len(anchors))]:
        optimization_attempts += 1
        anchor = train_x[row]
        anchor_frequency = float(train_y[row, selected_mode])
        # L-BFGS-B understands box constraints only.  Scaling each coordinate
        # by sqrt(9) makes that box lie entirely inside the declared 9D
        # Euclidean trust ball, so search and acceptance use the same rule.
        coordinate_radius = radius / math.sqrt(len(lower))
        local_lower = np.maximum(lower, anchor - coordinate_radius * width)
        local_upper = np.minimum(upper, anchor + coordinate_radius * width)
        # This is a genuinely new geometry, but remains within the local
        # credibility neighbourhood of a real target-matching Abaqus sample.
        start = np.clip(anchor + rng.normal(0.0, coordinate_radius / 2.0, len(lower)) * width,
                        local_lower, local_upper)

        def objective(normalized_vector):
            # 优化器必须使用无量纲坐标，避免米级厚度与幂指数的步长失衡。
            vector = lower + normalized_vector * width
            mean, std = model.predict(vector)
            error = abs(mean[0, selected_mode] - target) / target
            uncertainty = std[0, selected_mode] / target
            # Use an upper-confidence screen, not an exact-value objective.
            return error + FIXED_BAND_TARGET_POLICY["upper_confidence_multiplier"] * uncertainty

        result = minimize(objective, (start - lower) / width, method="L-BFGS-B",
                          bounds=list(zip((local_lower - lower) / width, (local_upper - lower) / width)),
                          options={"maxiter": 80, "ftol": 1.0e-10})
        if not result.success or not np.all(np.isfinite(result.x)):
            reject("OPTIMIZER_NOT_CONVERGED", row, selected_mode, anchor_frequency,
                   anchor_detuning, Optimizer_Message=str(result.message))
            continue
        vector = lower + result.x * width
        valid, reason, damping = _fixed_band_geometry_valid(
            vector, lower, upper, shell_size_bounds_mm)
        distance = float(np.min(np.linalg.norm((vector - train_x) / width, axis=1)))
        anchor_distance = float(np.linalg.norm((vector - anchor) / width))
        if not valid:
            reject("GEOMETRY_INVALID_" + reason, row, selected_mode, anchor_frequency,
                   anchor_detuning, Nearest_Training_Distance=distance,
                   Anchor_Distance=anchor_distance)
            continue
        if distance < minimum_new_distance:
            reject("NOT_DATABASE_EXTERNAL", row, selected_mode, anchor_frequency,
                   anchor_detuning, Nearest_Training_Distance=distance,
                   Anchor_Distance=anchor_distance)
            continue
        if anchor_distance > radius + 1.0e-10:
            reject("OUTSIDE_LOCAL_TRUST_RADIUS", row, selected_mode, anchor_frequency,
                   anchor_detuning, Nearest_Training_Distance=distance,
                   Anchor_Distance=anchor_distance)
            continue
        geometry_external_proposals += 1
        mean, std = model.predict(vector)
        mode = int(np.argmin(np.abs(mean[0] - target)))
        predicted_error = float(abs(mean[0, mode] - target) / target * 100.0)
        predicted_std_percent = float(std[0, mode] / target * 100.0)
        screened_upper = predicted_error + (
            FIXED_BAND_TARGET_POLICY["upper_confidence_multiplier"] * predicted_std_percent)
        reasons = []
        if predicted_std_percent > FIXED_BAND_TARGET_POLICY["maximum_prediction_std_percent"]:
            reasons.append("PREDICTED_STD_EXCEEDS_2_PERCENT")
        if screened_upper > FIXED_BAND_TARGET_POLICY["match_percent"]:
            reasons.append("UPPER_CONFIDENCE_ERROR_EXCEEDS_2_PERCENT")
        if reasons:
            if (predicted_error <= FIXED_BAND_TARGET_POLICY["match_percent"] and
                    predicted_std_percent <= FIXED_BAND_TARGET_POLICY[
                        "exploratory_max_prediction_std_percent"]):
                proposal = dict(zip(FIXED_BAND_9D, [float(v) for v in vector]))
                proposal.update({
                    "Candidate_ID": "FB15EXP_" + hashlib.sha256(
                        json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:12],
                    "Target_Frequency_Hz": float(target),
                    "Predicted_Mode_Index": mode + 1,
                    "Predicted_Frequency_Hz": float(mean[0, mode]),
                    "Predicted_STD_Hz": float(std[0, mode]),
                    "Predicted_Error_Percent": predicted_error,
                    "Predicted_STD_Percent": predicted_std_percent,
                    "Predicted_Upper_Confidence_Error_Percent": screened_upper,
                    "Nearest_Training_Distance": distance,
                    "Anchor_Distance": anchor_distance,
                    "Anchor_Sample_ID": train_rows[row]["Sample_ID"],
                    "Anchor_Mode_Index": selected_mode + 1,
                    "Anchor_Mode_Frequency_Hz": float(train_y[row, selected_mode]),
                    "Anchor_Detuning_Percent": anchor_detuning,
                    "Local_Trust_Radius": radius, "Geometry_Valid": True,
                    "New_Geometry": True, "Data_Library": FIXED_BAND_LIBRARY,
                    "Damping_Inward_X_m": 0.015, "Damping_Inward_Y_m": 0.015,
                    "rx_for_v2_builder": damping["rx"], "ry_for_v2_builder": damping["ry"],
                    "ABH_Shell_Overall_Length_X_m": damping["ABH_Shell_Overall_Length_X_m"],
                    "ABH_Shell_Overall_Width_Y_m": damping["ABH_Shell_Overall_Width_Y_m"],
                    "Damping_Configuration": damping["Damping_Configuration"],
                    "Prediction_Status": "EXPLORATORY_UQ_NOT_PASSED_REQUIRES_ABAQUS",
                    "Uncertainty_Calibration_Status": "UNCALIBRATED_NOT_A_95_PERCENT_GUARANTEE",
                    "Physical_Mode_Identity": "UNCONFIRMED",
                    "Optimizer_Success": True,
                    "Strict_Rejection_Reason": ";".join(reasons),
                    "Predicted_Ordered_Frequencies_Hz": json.dumps(
                        [float(value) for value in mean[0]]),
                })
                exploratory.append(proposal)
            reject(";".join(reasons), row, selected_mode, anchor_frequency, anchor_detuning,
                   Predicted_Mode_Index=mode + 1, Predicted_Frequency_Hz=float(mean[0, mode]),
                   Predicted_Error_Percent=predicted_error,
                   Predicted_STD_Percent=predicted_std_percent,
                   Predicted_Upper_Confidence_Error_Percent=screened_upper,
                   Nearest_Training_Distance=distance, Anchor_Distance=anchor_distance)
            continue
        proposal = dict(zip(FIXED_BAND_9D, [float(v) for v in vector]))
        proposal.update({
            "Candidate_ID": "FB15_" + hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:12],
            "Target_Frequency_Hz": float(target), "Predicted_Mode_Index": mode + 1,
            "Predicted_Frequency_Hz": float(mean[0, mode]),
            "Predicted_STD_Hz": float(std[0, mode]),
            "Predicted_Error_Percent": predicted_error,
            "Predicted_STD_Percent": predicted_std_percent,
            "Predicted_Upper_Confidence_Error_Percent": screened_upper,
            "Nearest_Training_Distance": distance,
            "Anchor_Distance": anchor_distance,
            "Anchor_Sample_ID": train_rows[row]["Sample_ID"],
            "Anchor_Mode_Index": selected_mode + 1,
            "Anchor_Mode_Frequency_Hz": float(train_y[row, selected_mode]),
            "Anchor_Detuning_Percent": anchor_detuning,
            "Local_Trust_Radius": radius, "Geometry_Valid": True,
            "New_Geometry": True, "Data_Library": FIXED_BAND_LIBRARY,
            "Damping_Inward_X_m": 0.015, "Damping_Inward_Y_m": 0.015,
            "rx_for_v2_builder": damping["rx"], "ry_for_v2_builder": damping["ry"],
            "ABH_Shell_Overall_Length_X_m": damping["ABH_Shell_Overall_Length_X_m"],
            "ABH_Shell_Overall_Width_Y_m": damping["ABH_Shell_Overall_Width_Y_m"],
            "Damping_Configuration": damping["Damping_Configuration"],
            "Prediction_Status": "LOCAL_TRUST_REGION_SCREENED_REQUIRES_ABAQUS",
            "Uncertainty_Calibration_Status": "UNCALIBRATED_NOT_A_95_PERCENT_GUARANTEE",
            "Physical_Mode_Identity": "UNCONFIRMED",
            "Optimizer_Success": True,
            "Predicted_Ordered_Frequencies_Hz": json.dumps([float(v) for v in mean[0]]),
        })
        proposals.append(proposal)

    selected = []
    for candidate in sorted(proposals, key=lambda item: (item["Predicted_Error_Percent"],
                                                          item["Predicted_STD_Hz"])):
        vector = np.array([candidate[key] for key in FIXED_BAND_9D])
        normal = (vector - lower) / width
        if all(np.linalg.norm(normal - (np.array([old[key] for key in FIXED_BAND_9D]) - lower) / width) >=
               FIXED_BAND_TARGET_POLICY["minimum_pair_distance"]
               for old in selected):
            selected.append(candidate)
        else:
            reject("PAIR_DIVERSITY_FILTER", train_rows.index(next(item for item in train_rows
                   if item["Sample_ID"] == candidate["Anchor_Sample_ID"])),
                   candidate["Anchor_Mode_Index"] - 1, candidate["Anchor_Mode_Frequency_Hz"],
                   candidate["Anchor_Detuning_Percent"],
                   Predicted_STD_Percent=candidate["Predicted_STD_Percent"],
                   Predicted_Upper_Confidence_Error_Percent=candidate["Predicted_Upper_Confidence_Error_Percent"])
        if len(selected) >= count:
            break
    for index, candidate in enumerate(selected, 1):
        candidate["Generation_Priority"] = index
    exploratory_selected = []
    previously_validated_skipped = 0
    for candidate in sorted(exploratory, key=lambda item: (
            item["Predicted_STD_Percent"], item["Predicted_Error_Percent"],
            item["Nearest_Training_Distance"])):
        if candidate["Candidate_ID"] in excluded_ids:
            previously_validated_skipped += 1
            reject("ALREADY_IN_INDEPENDENT_VALIDATION_LEDGER")
            continue
        vector = np.array([candidate[key] for key in FIXED_BAND_9D])
        normal = (vector - lower) / width
        if all(np.linalg.norm(normal - (
                np.array([old[key] for key in FIXED_BAND_9D]) - lower) / width) >=
               FIXED_BAND_TARGET_POLICY["minimum_pair_distance"]
               for old in exploratory_selected):
            exploratory_selected.append(candidate)
        if len(exploratory_selected) >= FIXED_BAND_TARGET_POLICY[
                "exploratory_candidate_count"]:
            break
    for index, candidate in enumerate(exploratory_selected, 1):
        candidate["Exploratory_Priority"] = index
    return {"candidates": selected, "exploratory_candidates": exploratory_selected,
            "rejections": rejections,
            "diagnostic": {"Target_Frequency_Hz": float(target), "Anchor_Count": len(anchors),
                           "Optimization_Attempts": optimization_attempts,
                           "Geometry_External_Proposals": geometry_external_proposals,
                           "UQ_Passed_Proposals": len(proposals),
                           "Exploratory_Candidates": len(exploratory_selected),
                           "Previously_Validated_Candidates_Skipped": previously_validated_skipped,
                           "Accepted_Candidates": len(selected)}}


def _fixed_band_validated_candidate_ids(output_dir):
    """Return candidate IDs with a completed solver record; never re-propose them."""
    path = os.path.join(output_dir, "fixed_band_15mm_abaqus_validation.csv")
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return {str(row.get("Candidate_ID", "")).strip() for row in csv.DictReader(stream)
                if str(row.get("Candidate_ID", "")).strip() and
                str(row.get("Status", "")).strip() == "SUCCESS"}


def _fixed_band_far_target(target, model, rows, x, y, policy, seed,
                           shell_size_bounds_mm=None):
    """远距离探索：全库排斥距离 + 有界锚点邻域；不是已验证可信域。"""
    lower, span = x.min(axis=0), np.ptp(x, axis=0)
    if np.any(span <= 0):
        raise ValueError("DEGENERATE_GEOMETRY_RANGE")
    normalized = (x - lower) / span
    tree = cKDTree(normalized)
    rng = np.random.RandomState(seed)
    nearest_mode = np.argmin(abs(y - target), axis=1)
    # 锚点只按训练数据库实际频谱选取，允许全部24个排序模态。
    anchor_ids = np.argsort(abs(y[np.arange(len(y)), nearest_mode] - target))[:min(120, len(x))]
    anchor_choice = rng.choice(anchor_ids, policy["proposal_count"])
    direction = rng.normal(size=(len(anchor_choice), x.shape[1]))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = rng.uniform(policy["minimum_database_distance"] * 1.1,
                         policy["maximum_anchor_distance"], len(direction))
    proposals = normalized[anchor_choice] + direction * radius[:, None]
    valid_box = np.all((proposals >= 0) & (proposals <= 1), axis=1)
    proposals, anchor_choice = proposals[valid_box], anchor_choice[valid_box]
    distance = tree.query(proposals)[0]
    outside = distance >= policy["minimum_database_distance"]
    proposals, anchor_choice = proposals[outside], anchor_choice[outside]
    feasible = [_fixed_band_geometry_valid(
        lower + q * span, lower, lower + span, shell_size_bounds_mm)[0] for q in proposals]
    proposals, anchor_choice = proposals[feasible], anchor_choice[feasible]
    diagnostic = {"Target_Frequency_Hz": target, "Random_Proposals": policy["proposal_count"],
                  "Algebraically_Feasible_Far_Proposals": len(proposals), "Optimization_Attempts": 0,
                  "Optimizer_Failed": 0, "Constraint_Rejected": 0, "Prediction_Rejected": 0}
    if not len(proposals):
        return [], [], diagnostic

    def mean_only(vectors):
        scaled = model.input_scaler.transform(np.atleast_2d(vectors))
        return model.output_scaler.inverse_transform(model.model.predict(scaled))

    with threadpool_limits(limits=1):
        means = np.concatenate([mean_only(lower + proposals[i:i+128] * span)
                                for i in range(0, len(proposals), 128)])
        order = np.argsort(np.min(abs(means - target), axis=1))
        # 多起点先去除相近起点，避免只围绕一个设计簇优化。
        starts = []
        for index in order:
            if all(np.linalg.norm(proposals[index] - proposals[j]) >= policy["minimum_pair_distance"] for j in starts):
                starts.append(index)
            if len(starts) >= policy["optimization_starts"]:
                break
        accepted = []
        for number, index in enumerate(starts):
            anchor = normalized[anchor_choice[index]]
            mode = int(np.argmin(abs(means[index] - target)))
            def objective(q):
                return float(((mean_only(lower + q * span)[0, mode] - target) / target) ** 2)
            constraints = [
                {"type": "ineq", "fun": lambda q: tree.query(q)[0] - policy["minimum_database_distance"]},
                {"type": "ineq", "fun": lambda q: policy["maximum_anchor_distance"] - np.linalg.norm(q - anchor)},
            ]
            fit = minimize(objective, proposals[index], method="SLSQP", bounds=[(0., 1.)] * x.shape[1],
                           constraints=constraints, options={"maxiter": policy["optimizer_maxiter"], "ftol": 1e-12})
            diagnostic["Optimization_Attempts"] += 1
            if not fit.success:
                diagnostic["Optimizer_Failed"] += 1
                continue
            vector = lower + fit.x * span
            distance, nearest = tree.query(fit.x)
            valid, reason, damping = _fixed_band_geometry_valid(
                vector, lower, lower + span, shell_size_bounds_mm)
            if (not valid or distance < policy["minimum_database_distance"] - 1e-8 or
                    np.linalg.norm(fit.x - anchor) > policy["maximum_anchor_distance"] + 1e-8):
                diagnostic["Constraint_Rejected"] += 1
                continue
            mean, std = model.predict(vector)
            mode = int(np.argmin(abs(mean[0] - target)))
            error, uncertainty = abs(mean[0, mode] - target) / target * 100, std[0, mode] / target * 100
            if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.min(mean) <= 0 or
                    error > policy["prediction_tolerance_percent"] or
                    uncertainty > policy["maximum_prediction_std_percent"]):
                diagnostic["Prediction_Rejected"] += 1
                continue
            parameters = dict(zip(FIXED_BAND_9D, map(float, vector)))
            candidate = dict(parameters)
            candidate.update({
                "Candidate_ID": "FB15EXP_" + hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:12],
                "Target_Frequency_Hz": target, "Predicted_Mode_Index": mode + 1,
                "Predicted_Frequency_Hz": float(mean[0, mode]), "Predicted_STD_Hz": float(std[0, mode]),
                "Predicted_Error_Percent": float(error), "Prediction_STD_Percent": float(uncertainty),
                "Nearest_Training_Distance": float(distance), "Nearest_Database_Sample_ID": rows[nearest]["Sample_ID"],
                "Anchor_Distance": float(np.linalg.norm(fit.x - anchor)),
                "Nearest_Database_Mode_Index": int(nearest_mode[nearest]) + 1,
                "Nearest_Database_Frequency_Hz": float(y[nearest, nearest_mode[nearest]]),
                "Geometry_Valid": True, "Geometry_Check_Level": "ALGEBRAIC_ONLY_NOT_MESH_OR_FE",
                "New_Geometry": True, "Data_Library": FIXED_BAND_LIBRARY,
                "Damping_Inward_X_m": 0.015, "Damping_Inward_Y_m": 0.015,
                "rx_for_v2_builder": damping["rx"], "ry_for_v2_builder": damping["ry"],
                "ABH_Shell_Overall_Length_X_m": damping["ABH_Shell_Overall_Length_X_m"],
                "ABH_Shell_Overall_Width_Y_m": damping["ABH_Shell_Overall_Width_Y_m"],
                "Damping_Configuration": damping["Damping_Configuration"],
                "Prediction_Status": "FAR_EXPLORATORY_REQUIRES_ABAQUS",
                "Physical_Mode_Identity": "UNCONFIRMED", "Abaqus_Validated": False,
                "Uncertainty_Calibration_Status": "UNCALIBRATED_NOT_A_95_PERCENT_GUARANTEE",
                "Spectrum_Inversion": bool(np.any(np.diff(mean[0]) < 0)),
                "Boundary_Margin_Normalized": float(np.min(np.minimum(fit.x, 1 - fit.x))),
            })
            accepted.append((candidate, fit.x, int(nearest)))
            if (number + 1) % 6 == 0:
                print("FAR_SEARCH target=%g starts=%d/%d accepted=%d" % (target, number+1, len(starts), len(accepted)), flush=True)
    selected, differences, positions = [], [], []
    for candidate, position, nearest in sorted(accepted, key=lambda item: (item[0]["Spectrum_Inversion"], item[0]["Prediction_STD_Percent"], item[0]["Predicted_Error_Percent"])):
        if any(np.linalg.norm(position - q) < policy["minimum_pair_distance"] for q in positions):
            continue
        selected.append(candidate)
        positions.append(position)
        for j, key in enumerate(FIXED_BAND_9D):
            scale, unit = (180 / math.pi, "deg") if key == "theta" else ((1., "dimensionless") if key == "power_law" else (1000., "mm"))
            differences.append({"Candidate_ID": candidate["Candidate_ID"], "Target_Frequency_Hz": target,
                "Nearest_Database_Sample_ID": rows[nearest]["Sample_ID"], "Parameter": key, "Unit": unit,
                "Candidate_Value": candidate[key] * scale, "Database_Value": float(x[nearest,j]) * scale,
                "Difference": (candidate[key] - float(x[nearest,j])) * scale,
                "Difference_Fraction_of_Database_Range": (candidate[key] - float(x[nearest,j])) / span[j]})
        if len(selected) >= policy["candidate_count"]:
            break
    diagnostic["Selected_Candidates"] = len(selected)
    return selected, differences, diagnostic


def run_fixed_band_far_generation(target, additional, dataset_csv, policy,
                                  shell_size_bounds_mm=None):
    """单独保存远距离探索成果；不覆盖生产模型、不提交有限元。"""
    targets = list(dict.fromkeys([float(target)] + list(map(float, additional))))
    if any(not np.isfinite(t) or t <= 0 for t in targets):
        raise ValueError("INVALID_TARGET_FREQUENCY")
    if not 0 < policy["minimum_database_distance"] < policy["maximum_anchor_distance"]:
        raise ValueError("INVALID_FAR_DISTANCE_POLICY")
    for key in ("candidate_count", "proposal_count", "optimization_starts", "optimizer_maxiter", "training_maxiter"):
        if isinstance(policy[key], bool) or int(policy[key]) != policy[key] or policy[key] <= 0:
            raise ValueError("INVALID_FAR_SEARCH_COUNT:" + key)
    for key in ("minimum_pair_distance", "prediction_tolerance_percent", "maximum_prediction_std_percent"):
        if not np.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError("INVALID_FAR_SEARCH_THRESHOLD:" + key)
    rows, x, y = _fixed_band_training_data(dataset_csv)
    identity = hashlib.sha256(x.tobytes() + y.tobytes() + str(policy["training_maxiter"]).encode() + b"FAR_V1").hexdigest()
    root = ensure_directory(os.path.join(BASE_OUTPUT, "fixed_band_15mm", "far_generated"))
    output = ensure_directory(os.path.join(root, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    cache = ensure_directory(os.path.join(root, "model_cache", identity[:16]))
    print("FAR_SEARCH_TRAIN_OR_LOAD geometries=%d; ABAQUS=DISABLED" % len(rows), flush=True)
    fitted = _fixed_band_fit_cached(cache, "experimental_spectrum", x, y, {"optimizer_maxiter": policy["training_maxiter"]})
    candidates, differences, diagnostics = [], [], []
    for i, frequency in enumerate(targets):
        print("FAR_SEARCH_START target=%g" % frequency, flush=True)
        found, delta, diagnostic = _fixed_band_far_target(
            frequency, fitted["model"], rows, x, y, policy, policy["seed"] + i,
            shell_size_bounds_mm)
        candidates.extend(found); differences.extend(delta); diagnostics.append(diagnostic)
        write_csv(os.path.join(output, "far_candidates.csv"), candidates)
        write_csv(os.path.join(output, "nearest_geometry_differences.csv"), differences)
        print(json.dumps(diagnostic), flush=True)
    user_prediction_path = write_fixed_band_user_prediction_table(candidates, "GENERATE_FAR")
    summary = {"Status": "FAR_CANDIDATES_NOT_FE_VALIDATED" if candidates else "NO_FAR_CANDIDATES",
               "Abaqus_Job_Submitted": False, "Training_Geometries": len(rows),
               "Data_Array_SHA256": identity, "Policy": policy, "Targets": targets,
               "Diagnostics": diagnostics, "Candidate_Count": len(candidates), "Output_Directory": output,
               "ABH_Shell_Size_Constraint_mm": shell_size_bounds_mm,
               "User_Prediction_Table": user_prediction_path,
               "Model_Training_Warnings": fitted["warnings"],
               "Distance_Normalization": {k: {"min": float(x[:,j].min()), "max": float(x[:,j].max())} for j,k in enumerate(FIXED_BAND_9D)}}
    save_json(os.path.join(output, "run_manifest.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def run_fixed_band_frequency_generation(mode, target, additional, dataset_csv,
                                        shell_size_bounds_mm=None):
    """固定 15 mm 数据库的训练和多解生成；本函数绝不提交 Abaqus。"""
    if mode != "GENERATE":
        raise RuntimeError("FIXED_BAND_15MM_CURRENTLY_SUPPORTS_GENERATE_ONLY")
    targets = list(dict.fromkeys([float(target)] + [float(item) for item in additional]))
    rows, x, y = _fixed_band_training_data(dataset_csv)
    lower, upper = x.min(axis=0), x.max(axis=0)
    with threadpool_limits(limits=1):
        model = _FixedBandSpectrumGPR().fit(x, y)
    output = ensure_directory(os.path.join(BASE_OUTPUT, "fixed_band_15mm", "generated"))
    validated_ids = _fixed_band_validated_candidate_ids(output)
    candidates, exploratory_candidates, rejections, diagnostics = [], [], [], []
    for index, target_value in enumerate(targets):
        generated = _fixed_band_generate_for_target(
            target_value, model, rows, x, y, lower, upper,
            count=FIXED_BAND_TARGET_POLICY["candidate_count"], seed=20261013 + index,
            excluded_ids=validated_ids, shell_size_bounds_mm=shell_size_bounds_mm)
        candidates.extend(generated["candidates"])
        exploratory_candidates.extend(generated.get("exploratory_candidates", []))
        rejections.extend(generated["rejections"])
        diagnostics.append(generated["diagnostic"])
    write_csv(os.path.join(output, "fixed_band_15mm_new_abh_candidates.csv"), candidates)
    exploratory_path = os.path.join(output, "fixed_band_15mm_exploratory_candidates.csv")
    write_csv(exploratory_path, exploratory_candidates)
    rejection_path = os.path.join(output, "fixed_band_15mm_generation_rejections.csv")
    write_csv(rejection_path, rejections)
    rejection_summary = []
    for target_value in targets:
        target_rows = [row for row in rejections if row["Target_Frequency_Hz"] == float(target_value)]
        counts = {}
        for row in target_rows:
            reason = row["Rejection_Reason"]
            counts[reason] = counts.get(reason, 0) + 1
        for reason, count_value in sorted(counts.items()):
            rejection_summary.append({"Target_Frequency_Hz": float(target_value),
                                      "Rejection_Reason": reason,
                                      "Rejected_Attempts": count_value})
    rejection_summary_path = os.path.join(output, "fixed_band_15mm_generation_rejection_summary.csv")
    write_csv(rejection_summary_path, rejection_summary)
    user_prediction_path = write_fixed_band_user_prediction_table(
        candidates + exploratory_candidates, "GENERATE_LOCAL")
    summary = {
        # 空候选是可信域/UQ筛选的正常结果，不能伪装成已经生成但尚未验证。
        "Status": ("GENERATED_NOT_ABAQUS_VALIDATED" if candidates else
                   ("EXPLORATORY_CANDIDATES_REQUIRE_ABAQUS_VALIDATION"
                    if exploratory_candidates else
                    "NO_CANDIDATES_PASSED_LOCAL_UQ_GATE")),
        "Abaqus_Job_Submitted": False,
        "Data_Library": FIXED_BAND_LIBRARY, "Training_Geometries": len(rows),
        "Target_Frequencies_Hz": targets, "Generated_Candidates": len(candidates),
        "Exploratory_Candidates": len(exploratory_candidates),
        "Previously_Validated_Candidate_IDs_Excluded": len(validated_ids),
        "Generation_Method": "LOCAL_TARGET_MATCHING_ANCHOR_PLUS_UQ_HARD_GATE",
        "Generation_Policy": FIXED_BAND_TARGET_POLICY,
        "ABH_Shell_Size_Constraint_mm": shell_size_bounds_mm,
        "Target_Diagnostics": diagnostics,
        "Output": os.path.join(output, "fixed_band_15mm_new_abh_candidates.csv"),
        "Exploratory_Output": exploratory_path,
        "Rejection_Output": rejection_path,
        "Rejection_Summary_Output": rejection_summary_path,
        "User_Prediction_Table": user_prediction_path,
    }
    save_json(os.path.join(output, "fixed_band_15mm_generation_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def save_json(path, value):
    with open(path + ".pending", "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
    os.replace(path + ".pending", path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ["Status"]
    with open(path + ".pending", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(path + ".pending", path)


def _fixed_band_user_view_directory():
    """日常查看入口：只保留预测表和验证表，不替代技术台账。"""
    return ensure_directory(os.path.join(
        BASE_OUTPUT, "fixed_band_15mm", "用户查看"))


def write_fixed_band_user_prediction_table(candidate_rows, generation_source):
    """将本轮生成结果压缩为一张面向用户的预测表。"""
    fields = ("Candidate_ID", "Target_Frequency_Hz", "Predicted_Frequency_Hz",
              "Predicted_Mode_Index", "Predicted_STD_Hz", "Prediction_STD_Percent",
              "Nearest_Training_Distance", "Nearest_Database_Sample_ID",
              "Prediction_Status", "Geometry_Valid", "New_Geometry",
              "Damping_Inward_X_m", "Damping_Inward_Y_m",
              "ABH_Shell_Overall_Length_X_m", "ABH_Shell_Overall_Width_Y_m") + tuple(FIXED_BAND_9D)
    rows = []
    for raw in candidate_rows:
        row = {field: raw.get(field, "") for field in fields}
        # 旧局部候选以Predicted_Raw_Mode命名；用户表统一显示为预测模态阶次。
        if not row["Predicted_Mode_Index"]:
            row["Predicted_Mode_Index"] = raw.get("Predicted_Raw_Mode", "")
        row["Generation_Source"] = generation_source
        rows.append(row)
    path = os.path.join(_fixed_band_user_view_directory(), "预测表.csv")
    write_csv(path, rows)
    return path


def write_fixed_band_user_validation_table(ledger_path, candidate_ids=None):
    """从完整FE台账导出一张可直接判断频率是否命中的验证表。"""
    if not os.path.isfile(ledger_path):
        return None
    with open(ledger_path, newline="", encoding="utf-8-sig") as stream:
        source = list(csv.DictReader(stream))
    if candidate_ids is not None:
        allowed = {str(value).strip() for value in candidate_ids}
        source = [row for row in source if str(row.get("Candidate_ID", "")).strip() in allowed]
    fields = ("Candidate_ID", "Status", "Target_Frequency_Hz", "Predicted_Frequency_Hz",
              "Predicted_Raw_Mode", "Abaqus_Frequency_Hz", "Abaqus_Raw_Mode",
              "Frequency_Error_Percent", "Frequency_Match_PASS", "Mode_Match_Status",
              "Positive_Mode_Count", "FE_Elapsed_Seconds", "Nearest_Database_Geometry_Distance",
              "Geometry_Valid", "Mesh_Valid", "ODB_Valid", "Failure_Reason",
              "Damping_Inward_X_m", "Damping_Inward_Y_m",
              "ABH_Shell_Overall_Length_X_m", "ABH_Shell_Overall_Width_Y_m") + tuple(FIXED_BAND_9D)
    rows = [{field: raw.get(field, "") for field in fields} for raw in source]
    path = os.path.join(_fixed_band_user_view_directory(), "验证表.csv")
    write_csv(path, rows)
    return path


def write_fixed_band_frequency_qualified_pool(validation_rows, candidate_ids=None):
    """导出真实 Abaqus 频率通过的候选池，供后续耦合减振筛选使用。

    这里的合格只表示独立 ABH 在目标频率的最近排序模态满足 ±2%。它不暗示
    Host--RABH 已产生减振效果，也不会自行启动任何耦合、PSD 或疲劳作业。
    """
    allowed = None
    if candidate_ids is not None:
        allowed = {str(value).strip() for value in candidate_ids}

    pool = []
    for raw in validation_rows:
        candidate_id = str(raw.get("Candidate_ID", "")).strip()
        if allowed is not None and candidate_id not in allowed:
            continue
        if str(raw.get("Status", "")).upper() != "SUCCESS":
            continue
        passed = str(raw.get("Frequency_Match_PASS", "")).strip().lower()
        if passed not in ("true", "1", "yes"):
            continue
        row = dict(raw)
        row["Frequency_Selection_Status"] = "FREQUENCY_QUALIFIED_FOR_COUPLED_SCREENING"
        row["Coupled_Validation_Status"] = "NOT_RUN"
        row["Dynamic_Response_Status"] = "NOT_RUN"
        pool.append(row)

    pool.sort(key=lambda item: (float(item.get("Target_Frequency_Hz", 0.0)),
                                float(item.get("Frequency_Error_Percent", "inf"))))
    for index, row in enumerate(pool, 1):
        row["Frequency_Qualified_Rank"] = index

    output_dir = _fixed_band_user_view_directory()
    fields = (
        "Frequency_Qualified_Rank", "Frequency_Selection_Status",
        "Coupled_Validation_Status", "Dynamic_Response_Status", "Candidate_ID",
        "Target_Frequency_Hz", "Abaqus_Frequency_Hz", "Abaqus_Raw_Mode",
        "Frequency_Error_Percent", "Predicted_Frequency_Hz", "Predicted_Raw_Mode",
        "RABH_Assembly_Mass", "Nearest_Database_Geometry_Distance",
        "ABH_Shell_Overall_Length_X_m", "ABH_Shell_Overall_Width_Y_m",
        "Damping_Inward_X_m", "Damping_Inward_Y_m",
    ) + tuple(FIXED_BAND_9D)
    pool_path = os.path.join(output_dir, "频率合格候选池.csv")
    write_csv(pool_path, [{field: row.get(field, "") for field in fields} for row in pool])

    by_target = {}
    for row in pool:
        target = row.get("Target_Frequency_Hz", "")
        by_target[target] = by_target.get(target, 0) + 1
    report_path = os.path.join(output_dir, "频率合格候选池说明.md")
    lines = [
        "# 固定 15 mm 阻尼带：频率合格候选池",
        "",
        "本表只保留独立 Abaqus 模态计算成功，且目标频率误差不超过 2% 的新构型。",
        "频率合格不等价于耦合减振性能合格；Host--RABH 耦合、动态响应、PSD 和疲劳均尚未运行。",
        "",
        "## 数量",
        "",
        "- 总数：%d" % len(pool),
    ]
    for target in sorted(by_target, key=lambda item: float(item)):
        lines.append("- %s Hz：%d 个" % (target, by_target[target]))
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return {"Frequency_Qualified_Pool": pool_path,
            "Frequency_Qualified_Pool_Report": report_path,
            "Frequency_Qualified_Count": len(pool)}


def read_json(path):
    if not os.path.isfile(path): return None
    with open(path, encoding="utf-8") as stream: return json.load(stream)


def load_training_data():
    modes = _load_valid_modes()
    samples = {}
    spectra = {}
    for row in modes:
        sid = row["Sample_ID"]
        samples[sid] = row
        spectra.setdefault(sid, {})[row["Mode_Index"]] = row["Mode_Frequency_Hz"]
    ids = sorted(samples)
    x = np.array([[samples[s][k] for k in FEATURES] for s in ids])
    # Complete first eight ordered modes avoid right-censoring by the 500 Hz cap.
    y = np.array([[spectra[s][j] for j in range(1, 9)] for s in ids])
    if not np.all(np.isfinite(y)) or not np.all(y > 0) or np.any(np.diff(y, axis=1)<0):
        raise ValueError("INVALID_ORDERED_FREQUENCY_SPECTRA")
    # Canonical geometry grouping also catches accidental duplicate ABH_IDs.
    groups = np.array([hashlib.sha256(json.dumps(list(v[:9]), separators=(",", ":")).encode()).hexdigest()[:16] for v in x])
    ref_rows = [i for i,s in enumerate(ids) if s.endswith("_D02")]
    ref = x[ref_rows[0], -2:]
    if len(ref_rows) != 15 or not np.allclose(x[ref_rows, -2:], ref, atol=1e-12, rtol=0):
        raise ValueError("D02_FIXED_REFERENCE_NOT_SUPPORTED_BY_EXPECTED_15_GEOMETRIES")
    fixed = np.all(np.abs(x[:, -2:]-ref)<1e-12, axis=1)
    bounds = read_json(config.GENERAL_DESIGN_BOUNDS_FILE)["bounds"]
    lower = np.array([bounds[k]["lower"] for k in FEATURES])
    width = np.array([bounds[k]["upper"]-bounds[k]["lower"] for k in FEATURES])
    return {"x": x, "y": y, "groups": groups, "ids": ids, "samples": samples,
            "fixed": fixed, "ref": ref, "lower": lower, "width": width,
            "mode_rows": len(modes), "raw_modes": modes, "bounds": bounds}


class ForwardModel:
    """Log-frequency regression with explicit Hz uncertainty transformation."""
    def __init__(self, kind): self.kind = kind

    def fit(self, x, log_y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        if self.kind == "TREE":
            self.tree = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2,
                                            random_state=POLICY["seed"], n_jobs=1).fit(z, log_y)
        else:
            self.models = []
            for j in range(log_y.shape[1]):
                kernel = ConstantKernel(1., (.01, 100.))*Matern(np.ones(x.shape[1]), (.05, 100.), nu=2.5)+WhiteKernel(1e-5, (1e-8, .1))
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
                                              n_restarts_optimizer=0, random_state=POLICY["seed"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(z, log_y[:, j])
                self.models.append(gp)
        return self

    def predict_log(self, x):
        z = self.scaler.transform(np.atleast_2d(x))
        if self.kind == "TREE":
            values = np.array([t.predict(z) for t in self.tree.estimators_])
            return values.mean(axis=0), values.std(axis=0)
        pairs = [gp.predict(z, return_std=True) for gp in self.models]
        return np.array([p[0] for p in pairs]).T, np.array([p[1] for p in pairs]).T


class SpectrumModel:
    def __init__(self, name, ref): self.name, self.ref = name, ref

    def fit(self, x, y, groups):
        ly = np.log(y)
        fixed = np.all(np.abs(x[:, -2:]-self.ref)<1e-12, axis=1)
        if self.name == MODELS[0]:
            self.base = ForwardModel("GP").fit(x[fixed, :9], ly[fixed])
        elif self.name in (MODELS[1], MODELS[3]):
            self.base = ForwardModel("TREE" if self.name==MODELS[3] else "GP").fit(x, ly)
        else:
            references = {g: ly[i] for i,g in enumerate(groups) if fixed[i]}
            paired = np.array([i for i,g in enumerate(groups) if g in references])
            delta = np.array([ly[i]-references[groups[i]] for i in paired])
            self.correction = ForwardModel("GP").fit(x[paired], delta)
            xr = x.copy(); xr[:, -2:] = self.ref
            correction = self.correction.predict_log(x)[0]-self.correction.predict_log(xr)[0]
            adjusted = ly-correction
            geom_x, geom_y = [], []
            for g in np.unique(groups):
                mask = groups==g
                geom_x.append(x[mask, :9][0]); geom_y.append(adjusted[mask].mean(axis=0))
            self.base = ForwardModel("GP").fit(np.array(geom_x), np.array(geom_y))
        return self

    def predict_log(self, x):
        x = np.atleast_2d(x)
        if self.name in (MODELS[1], MODELS[3]): return self.base.predict_log(x)
        mu, sigma = self.base.predict_log(x[:, :9])
        if self.name == MODELS[2]:
            xr = x.copy(); xr[:, -2:] = self.ref
            a, sa = self.correction.predict_log(x); b, sb = self.correction.predict_log(xr)
            mu += a-b
            away = np.any(np.abs(x[:, -2:]-self.ref)>1e-12, axis=1)
            sigma[away] = np.sqrt(sigma[away]**2+sa[away]**2+sb[away]**2)
        return mu, sigma

    def predict(self, x):
        mu, s = self.predict_log(x)
        # GP normalized y is already inverse-scaled by sklearn; no second scaling.
        return np.exp(mu), np.exp(mu+.5*s*s)*np.sqrt(np.expm1(s*s))


def metric_row(name, scope, actual, predicted, log_std, groups, selection):
    rel = np.abs(predicted-actual)/actual*100
    sigma = np.maximum(log_std, 1e-8)
    residual = np.log(actual)-np.log(predicted)
    return {"Model":name,"Scope":scope,"Subset":selection,"N":int(actual.size),
            "Geometry_Groups":int(len(set(groups))),"MAPE_Percent":float(rel.mean()),
            "MAE_Hz":float(np.mean(np.abs(predicted-actual))),"P90_Relative_Error_Percent":float(np.percentile(rel,90)),
            "Max_Relative_Error_Percent":float(rel.max()),"Mean_Log_STD":float(sigma.mean()),
            "Coverage_68_Percent":float(100*np.mean(np.abs(residual)<=sigma)),
            "Coverage_95_Percent":float(100*np.mean(np.abs(residual)<=1.96*sigma)),
            "Gaussian_NLL_Log_Frequency":float(np.mean(.5*np.log(2*np.pi*sigma*sigma)+.5*(residual/sigma)**2)),
            "Uncertainty_Use":"DIAGNOSTIC_ONLY_NOT_SOLE_ACCEPTANCE_CRITERION"}


def compare_surrogates(data):
    x,y,g = data["x"],data["y"],data["groups"]
    splits = list(GroupKFold(n_splits=POLICY["cv_folds"]).split(x,y,g))
    rows, diagnostics = [], {}
    for name in MODELS:
        p=np.full_like(y,np.nan); s=np.full_like(y,np.nan)
        for train,test in splits:
            assert not set(g[train]) & set(g[test])
            fit=SpectrumModel(name,data["ref"]).fit(x[train],y[train],g[train])
            mu, sigma = fit.predict_log(x[test]); p[test]=np.exp(mu); s[test]=sigma
        scopes = [("COMMON_FIXED_D02",data["fixed"])]
        if name!=MODELS[0]: scopes.append(("ALL_DAMPING_CONDITIONS",np.ones(len(x),dtype=bool)))
        for scope,mask in scopes:
            a,b,std=y[mask],p[mask],s[mask]
            rows.append(metric_row(name,scope,a,b,std,g[mask],"ALL_MODES_1_8"))
            for j in range(8): rows.append(metric_row(name,scope,a[:,j],b[:,j],std[:,j],g[mask],"MODE_%d"%(j+1)))
            for lo,hi in ((0,50),(50,100),(100,200),(200,500)):
                band=(a>=lo)&(a<hi)
                if band.any(): rows.append(metric_row(name,scope,a[band],b[band],std[band],g[mask],"BAND_%d_%d_HZ"%(lo,hi)))
        fixed=data["fixed"]
        diagnostics[name]={"P90_By_Mode":np.percentile(abs(p[fixed]-y[fixed])/y[fixed],90,axis=0).tolist(),
                           "OOF_Predictions_Hz":p.tolist(),"OOF_Log_STD":s.tolist()}
        print("GROUP_CV_COMPLETED "+name,flush=True)
    eligible=[r for r in rows if r["Scope"]=="COMMON_FIXED_D02" and r["Subset"]=="ALL_MODES_1_8"]
    chosen=min(eligible,key=lambda r:(r["MAPE_Percent"],r["P90_Relative_Error_Percent"]))["Model"]
    write_csv(os.path.join(OUTPUT,"frequency_surrogate_comparison.csv"),rows)
    model=SpectrumModel(chosen,data["ref"]).fit(x,y,g)
    return model,rows,diagnostics


def generate_candidates(target, model, data, diagnostics):
    """Trust-region multistart search. All generated coordinates are new geometry."""
    low,width=data["lower"][:9],data["width"][:9]
    z=(data["x"][:,:9]-low)/width
    centers=np.unique(z,axis=0)
    xp=np.column_stack((centers*width+low,np.tile(data["ref"],(len(centers),1))))
    means,unused=model.predict(xp)
    starts=sorted((abs(means[i,j]-target)/target,i,j) for i in range(len(centers)) for j in range(8))[:24]
    rng=np.random.default_rng(POLICY["seed"]+int(target*100))
    pool=[]
    p90=np.array(diagnostics[model.name]["P90_By_Mode"])
    def parameters(t): return dict(zip(FEATURES,np.r_[low+width*t,data["ref"]]))
    def constraints(t):
        p=parameters(t)
        return np.array([(p["Rou_2"]**2-(p["Rou_3"]*math.cos(p["theta"]))**2)/.04,
                         (p["Rou_4"]**2-(p["Rou_3"]*math.sin(p["theta"]))**2)/.04,
                         (p["platform_height"]-p["cutoff_thickness"])/.008])-1e-7
    for unused,i,j in starts:
        center=centers[i]
        for repeat in range(2):
            lower=np.maximum(0,center-POLICY["local_halfwidth"])
            upper=np.minimum(1,center+POLICY["local_halfwidth"])
            start=np.clip(center+rng.normal(0,.04,9),lower,upper)
            def objective(t):
                m,s=model.predict(np.array(list(parameters(t).values())))
                return ((m[0,j]-target)/target)**2+.0002*(s[0,j]/target)**2+.00002*np.sum((t-center)**2)
            if model.name==MODELS[3]:
                # Trees are discontinuous: rank randomized trust-region proposals.
                proposals=np.clip(center+rng.normal(0,.08,(250,9)),lower,upper)
            else:
                result=minimize(objective,start,method="SLSQP",bounds=list(zip(lower,upper)),
                                constraints=[{"type":"ineq","fun":constraints},
                                             {"type":"ineq","fun":lambda t:np.min(np.linalg.norm(centers-t,axis=1))-POLICY["min_database_distance"]}],
                                options={"maxiter":65,"ftol":1e-10})
                proposals=[result.x]
            for proposal in proposals:
                p=parameters(proposal); valid,reason,derived=validate_general(p)
                distances=np.linalg.norm(centers-proposal,axis=1)
                if not valid or distances.min()<POLICY["min_database_distance"]-1e-6 or distances.min()>POLICY["max_support_distance"]:continue
                if np.sum(np.abs(proposal-centers[np.argmin(distances)])>=.02)<2:continue
                m,s=model.predict(np.array(list(p.values())))
                mode=int(np.argmin(abs(m[0]-target)))
                error=float(abs(m[0,mode]-target)/target*100)
                # Posterior uncertainty is only one component; distance and OOF
                # error remain explicit and no 2% predictive guarantee is made.
                relative_std=float(s[0,mode]/target)
                if relative_std>0.5 or error>POLICY["match_percent"]:continue
                score=error+10*relative_std+2*float(distances.min())+p90[mode]
                sid="G9_"+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]
                pool.append(dict(p,Candidate_ID=sid,Target_Frequency_Hz=target,Predicted_Raw_Mode=mode+1,
                                 Predicted_Frequency_Hz=float(m[0,mode]),Predicted_STD_Hz=float(s[0,mode]),
                                 Predicted_Error_Percent=error,Group_CV_P90_Relative_Error_Percent=float(p90[mode]*100),
                                 Nearest_Database_Geometry_Distance=float(distances.min()),Geometry_Valid=True,New_Geometry=True,
                                 Prediction_Status="EXPLORATORY_REQUIRES_ABAQUS",Surrogate=model.name,
                                 Damping_Reference="D02",Footprint_Area=derived["Footprint_Area"],Selection_Score=score,
                                 Predicted_Ordered_Frequencies_Hz=json.dumps(m[0].tolist())))
    selected=[]
    for row in sorted(pool,key=lambda r:r["Selection_Score"]):
        v=(np.array([row[k] for k in GEOMETRY])-low)/width
        if all(np.linalg.norm(v-(np.array([old[k] for k in GEOMETRY])-low)/width)>=POLICY["min_pair_distance"] for old in selected):
            selected.append(row)
        if len(selected)>=POLICY["candidate_count"]:break
    for i,r in enumerate(selected):r["Validation_Priority"]=i+1
    return selected


def validate_one(candidate, state, state_dir, runtime=None):
    # Optional isolated campaign namespace; historical default budget/state is unchanged.
    runtime = runtime or {}
    run_root = runtime.get("run_root", RUNS)
    output_dir = runtime.get("output_dir", OUTPUT)
    output_name = runtime.get("output_name", "new_abh_abaqus_validation.csv")
    policy = runtime.get("policy", POLICY)
    import time
    started = time.perf_counter()
    sid=candidate["Candidate_ID"]
    if sid in state["Jobs"]:
        req=state["Jobs"][sid]["Request"]
        result=read_json(req["Result_Path"])
        if result:
            record=summarize_validation(candidate,result)
            state["Validation"][sid]=record
            return record
        return {"Candidate_ID":sid,"Status":"INCOMPLETE_PREVIOUS_RUN_NOT_RESUBMITTED","Target_Frequency_Hz":candidate["Target_Frequency_Hz"]}
    if len(state["Jobs"])>=policy["max_jobs"]:raise RuntimeError("12_JOB_BUDGET_EXHAUSTED")
    # Historical D02 validation had a finite campaign quota.  New fixed-band
    # validation is deliberately one-candidate-per-launch and keeps a
    # separate ledger, so it must not count earlier candidates as a reason to
    # block a later manual validation.
    if policy.get("enforce_global_request_limit", True):
        prior_requests=sum(name.endswith("_request.json") for root,dirs,files in os.walk(os.path.join(run_root,"_state")) for name in files)
        if prior_requests>=policy["max_jobs"]:raise RuntimeError("GLOBAL_12_ISOLATED_JOB_BUDGET_EXHAUSTED")
    if shutil.disk_usage(run_root).free< policy["minimum_free_gb"]*1024**3:raise RuntimeError("INSUFFICIENT_DISK_SPACE")
    run_dir=os.path.join(run_root,sid)
    if os.path.exists(run_dir):raise RuntimeError("REFUSING_TO_REUSE_UNREGISTERED_RUN_DIRECTORY")
    req={"Candidate":candidate,"Run_Directory":run_dir,"Job_Name":"freq9_"+sid[3:],
         "Result_Path":os.path.join(state_dir,sid+"_result.json")}
    if runtime.get("resource_policy"):
        req["Resource_Policy"]=runtime["resource_policy"]
    request_path=os.path.join(state_dir,sid+"_request.json");save_json(request_path,req)
    state["Jobs"][sid]={"Status":"RUNNING","Request":req}
    save_json(os.path.join(state_dir,"checkpoint.json"),state)
    env=os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"]="1"
    env["PYTHONPATH"]=os.pathsep.join([PROJECT_ROOT,os.path.join(PROJECT_ROOT,"01_CORE"),DATABASE_DIR])
    for name in ("TEMP","TMP","ABAQUS_TEMP","ABAQUS_SCRATCH"):env[name]=ensure_directory(ABAQUS_TEMP_DIR)
    command=["abaqus","cae","noGUI="+os.path.join(DATABASE_DIR,"v2_pilot_replay_worker.py"),"--","--design-request",request_path]
    with open(os.path.join(state_dir,sid+"_worker.log"),"a",encoding="utf-8") as log:
        process=subprocess.Popen(["cmd","/c",subprocess.list2cmdline(command)],cwd=PROJECT_ROOT,env=env,stdout=log,stderr=log)
        state["Jobs"][sid]["PID"]=process.pid;save_json(os.path.join(state_dir,"checkpoint.json"),state)
        safety_abort=None; emergency_count=0
        try:
            resources=runtime.get("resource_policy",{})
            if not resources:
                process.wait(timeout=policy["job_timeout_seconds"])
            else:
                deadline=time.monotonic()+policy["job_timeout_seconds"]
                while process.poll() is None:
                    if time.monotonic()>=deadline:
                        raise subprocess.TimeoutExpired(command,policy["job_timeout_seconds"])
                    try:process.wait(timeout=min(5,max(0.1,deadline-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        memory=_windows_memory_snapshot()
                        memory["Process_Tree"]=_owned_process_snapshot(process.pid)
                        emergency_count=_resource_emergency_count(memory,resources,emergency_count)
                        memory["Resource_Emergency_Consecutive_Samples"]=emergency_count
                        with open(os.path.join(state_dir,sid+"_resource_monitor.jsonl"),"a",encoding="utf-8") as monitor:
                            monitor.write(json.dumps(memory)+"\n")
                        emergency_count=_resource_emergency_count(memory,resources,emergency_count)
                        if emergency_count>=resources.get("emergency_consecutive_samples",3):
                            safety_abort="RESOURCE_SAFETY_ABORT"
                            save_json(os.path.join(state_dir,sid+"_resource_abort.json"),memory)
                            # Only the child tree created by this invocation is targeted.
                            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=log,stderr=log)
                            process.wait(timeout=30)
                            break
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=log,stderr=log)
            process.wait(timeout=30)
        except Exception:
            if process.poll() is None:
                subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=log,stderr=log)
                process.wait(timeout=30)
            raise
        state["Jobs"][sid]["Worker_Exit_Code"]=process.poll()
    result=read_json(req["Result_Path"]) or {"Status":"FAILED","Failure_Reason":safety_abort or "NO_FINAL_EXTRACTION_WORKER_EXITED_OR_TIMEOUT"}
    if runtime and result["Status"] != "SUCCESS":
        log_path=os.path.join(run_dir,req["Job_Name"]+".log")
        if os.path.isfile(log_path):
            diagnostic=open(log_path,encoding="utf-8",errors="replace").read()
            if "insufficient memory" in diagnostic.lower():
                result["Failure_Reason"] += "; CONFIRMED_SOLVER_MEMORY_LIMIT_INSUFFICIENT"
            elif "ReadFile from pipe failed" in diagnostic:
                result["Failure_Reason"] += "; SOLVER_PROCESS_PIPE_FAILURE_REQUIRES_RESOURCE_CHECK"
    state["Jobs"][sid]["Status"]=result["Status"]
    record=summarize_validation(candidate,result)
    record["FE_Elapsed_Seconds"] = time.perf_counter()-started
    record["Worker_Exit_Code"]=process.returncode
    if result["Status"]=="SUCCESS":
        record["Frequency_Upper_Limit_Hz"]=result.get("Frequency_Upper_Limit_Hz")
        record["Highest_Extracted_Frequency_Hz"]=max(result["All_Positive_Frequencies_Hz"])
        record["Resource_Policy"]=json.dumps(result.get("Resource_Policy"))
    state["Validation"][sid]=record
    write_csv(os.path.join(output_dir,output_name),list(state["Validation"].values()))
    save_json(os.path.join(state_dir,"checkpoint.json"),state)
    if result["Status"]=="SUCCESS" and record.get("Geometry_Valid") and not runtime.get("preserve_large_artifacts"):
        # JSON plus a readable validation CSV are durable before cleanup.
        with open(os.path.join(output_dir,output_name),encoding="utf-8-sig") as h:
            assert any(r["Candidate_ID"]==sid for r in csv.DictReader(h))
        from abaqus_cleanup import write_manifest
        write_manifest(os.path.join(state_dir,sid+"_cleanup_inventory.md"),[
            {"directory":run_dir,"allowed_root":run_root,"reason":"Successful JSON and readable CSV verified"}])
        cleanup=cleanup_abaqus_job(run_dir,run_root,True)
        state["Freed_Bytes"]+=cleanup["freed_bytes"]
        save_json(os.path.join(state_dir,"checkpoint.json"),state)
    print("INDEPENDENT_VALIDATION_COMPLETED "+sid+" "+record["Status"],flush=True)
    return record


def summarize_validation(candidate, result):
    row={"Candidate_ID":candidate["Candidate_ID"],"Target_Frequency_Hz":candidate["Target_Frequency_Hz"],
         "Predicted_Frequency_Hz":candidate["Predicted_Frequency_Hz"],"Predicted_Raw_Mode":candidate["Predicted_Raw_Mode"],
         "Status":result["Status"],"Geometry_Valid":result.get("Geometry_Valid",False),"New_Geometry":True,
         "Source":"NEW_ABAQUS_ISOLATED_MODAL","Used_For_Training":False,
         "Failure_Reason":result.get("Failure_Reason","")}
    for key in ("Predicted_STD_Hz","Nearest_Database_Geometry_Distance","Group_CV_P90_Relative_Error_Percent","Surrogate_Mode_Order_Warning"):
        row[key]=candidate.get(key)
    row.update({k:candidate[k] for k in FEATURES})
    if result["Status"]=="SUCCESS":
        f=np.array(result["All_Positive_Frequencies_Hz"])
        target=candidate["Target_Frequency_Hz"]; j=int(np.argmin(abs(f-target)))
        branch=candidate["Predicted_Raw_Mode"]-1
        row.update({"Abaqus_Frequency_Hz":float(f[j]),"Abaqus_Raw_Mode":j+1,
                    "Frequency_Error_Percent":float(abs(f[j]-target)/target*100),
                    "Same_Ordered_Mode_Frequency_Hz":float(f[branch]) if branch<len(f) else None,
                    "Surrogate_Same_Mode_Error_Percent":float(abs(f[branch]-candidate["Predicted_Frequency_Hz"])/f[branch]*100) if branch<len(f) else None,
                    "Mode_Match_Status":"SAME_SORTED_MODE_NOT_PHYSICAL_ID" if branch==j else "NEAREST_MODE_DIFFERS_BRANCH_AMBIGUOUS",
                    "Frequency_Match_PASS":bool(abs(f[j]-target)/target*100<=POLICY["match_percent"]),
                    "Independent_Geometry_Validation":True,"Positive_Mode_Count":len(f),
                    "RABH_Assembly_Mass":result["masses"]["Total_Attachment_Mass"]})
        for field in ("Thickness_Field_Implementation","Geometry_Builder_Version","Section_Valid","Tie_Valid","BC_Valid","Mesh_Valid","ODB_Valid"):
            row[field]=result.get(field)
    return row


def rebuild_fixed_band_validation_ledger():
    """Rebuild the fixed-band validation table from per-candidate checkpoints.

    A single validation has its own checkpoint.  The durable ledger must merge
    those checkpoints rather than allowing one single-candidate write to
    replace another candidate's result.  This performs no Abaqus work.
    """
    output_dir = ensure_directory(os.path.join(
        BASE_OUTPUT, "fixed_band_15mm", "generated"))
    state_root = os.path.join(RUNS, "fixed_band_15mm_validation", "_state")
    records = {}
    if os.path.isdir(state_root):
        for name in os.listdir(state_root):
            checkpoint = os.path.join(state_root, name, "checkpoint.json")
            state = read_json(checkpoint)
            if not state:
                continue
            for candidate_id, record in state.get("Validation", {}).items():
                records[candidate_id] = record
    path = os.path.join(output_dir, "fixed_band_15mm_abaqus_validation.csv")
    write_csv(path, [records[key] for key in sorted(records)])
    return path, len(records)


def validate_saved_candidate(candidate_id, candidate_csv=None):
    """只验证一个已生成的新构型；不会重新训练或生成其它候选。"""
    global OUTPUT
    OUTPUT = BASE_OUTPUT
    candidate_csv = candidate_csv or os.path.join(OUTPUT, "new_abh_candidates.csv")
    if not os.path.isfile(candidate_csv):
        raise RuntimeError("CANDIDATE_CSV_NOT_FOUND: " + candidate_csv)

    with open(candidate_csv, newline="", encoding="utf-8-sig") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if str(row.get("Candidate_ID", "")).strip() == candidate_id]
    if len(matches) != 1:
        raise RuntimeError("CANDIDATE_ID_NOT_FOUND_OR_NOT_UNIQUE: " + candidate_id)

    # CSV 中的数字读入后是文本；恢复既有验证函数需要的数值类型。
    candidate = dict(matches[0])
    numeric_fields = set(FEATURES) | {
        "Target_Frequency_Hz", "Predicted_Raw_Mode", "Predicted_Frequency_Hz",
        "Predicted_STD_Hz", "Predicted_Error_Percent",
        "Group_CV_P90_Relative_Error_Percent", "Nearest_Database_Geometry_Distance",
        "Footprint_Area", "Selection_Score", "Validation_Priority",
    }
    for field in numeric_fields:
        if field not in candidate or candidate[field] in (None, ""):
            continue
        value = float(candidate[field])
        candidate[field] = int(value) if field in ("Predicted_Raw_Mode", "Validation_Priority") else value

    # 已有成功记录只返回既有 Abaqus 证据，绝不重复提交同一构型。
    validation_csv = os.path.join(OUTPUT, "new_abh_abaqus_validation.csv")
    if os.path.isfile(validation_csv):
        with open(validation_csv, newline="", encoding="utf-8-sig") as stream:
            previous = [row for row in csv.DictReader(stream)
                        if str(row.get("Candidate_ID", "")).strip() == candidate_id
                        and row.get("Status") == "SUCCESS"]
        if previous:
            result = dict(previous[-1])
            result.update({"Status": "ALREADY_VALIDATED_NO_RESUBMIT",
                           "Candidate_ID": candidate_id,
                           "Abaqus_Job_Submitted": False,
                           "Validation_CSV": validation_csv})
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return result

    state_dir = ensure_directory(os.path.join(RUNS, "_single_candidate_validation", candidate_id))
    checkpoint = os.path.join(state_dir, "checkpoint.json")
    state = read_json(checkpoint) or {
        "Status": "STARTED", "Jobs": {}, "Validation": {}, "Freed_Bytes": 0,
        "Candidate_CSV": os.path.abspath(candidate_csv), "Candidate_ID": candidate_id,
    }
    try:
        record = validate_one(candidate, state, state_dir)
        state["Status"] = record.get("Status", "UNKNOWN")
    except Exception as exc:
        import traceback
        state.update({"Status": "STOPPED_WITH_EVIDENCE", "Error": str(exc),
                      "Traceback": traceback.format_exc()})
        record = {"Candidate_ID": candidate_id, "Status": state["Status"],
                  "Error": state["Error"]}
    finally:
        save_json(checkpoint, state)
        # 不能用单候选 state 覆盖历史验证台账；仅替换同 ID 的旧记录。
        existing_rows = []
        if os.path.isfile(validation_csv):
            with open(validation_csv, newline="", encoding="utf-8-sig") as stream:
                existing_rows = [row for row in csv.DictReader(stream)
                                 if str(row.get("Candidate_ID", "")).strip() != candidate_id]
        existing_rows.extend(state.get("Validation", {}).values())
        write_csv(validation_csv, existing_rows)

    submitted = bool(state.get("Jobs", {}).get(candidate_id, {}).get("PID"))
    result = dict(record, Abaqus_Job_Submitted=submitted,
                  Validation_CSV=os.path.join(OUTPUT, "new_abh_abaqus_validation.csv"))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def validate_fixed_band_saved_candidate(candidate_id, candidate_csv=None):
    """Validate exactly one new 15 mm-band geometry with the V2 FE chain.

    This is intentionally separate from the retired D02 validation campaign.
    It never retrains the surrogate or regenerates candidates.  The physical
    damping widths are fixed at 15 mm in X and Y; derived ``rx``/``ry`` are
    calculated only because the verified V2 builder requires them.
    """
    output_dir = ensure_directory(os.path.join(
        BASE_OUTPUT, "fixed_band_15mm", "generated"))
    if candidate_csv:
        candidate_csv = candidate_csv
    elif str(candidate_id).startswith("FB15EXP_"):
        # 探索候选与严格候选分表保存；只凭 ID 即可选择正确来源，避免用户
        # 因填写错 CSV 而误以为候选不存在。
        candidate_csv = os.path.join(
            output_dir, "fixed_band_15mm_exploratory_candidates.csv")
    else:
        candidate_csv = os.path.join(
            output_dir, "fixed_band_15mm_new_abh_candidates.csv")
    if not os.path.isfile(candidate_csv):
        raise RuntimeError("FIXED_BAND_CANDIDATE_CSV_NOT_FOUND: " + candidate_csv)

    with open(candidate_csv, newline="", encoding="utf-8-sig") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if str(row.get("Candidate_ID", "")).strip() == candidate_id]
    if len(matches) != 1:
        raise RuntimeError("FIXED_BAND_CANDIDATE_ID_NOT_FOUND_OR_NOT_UNIQUE: " + candidate_id)

    raw = matches[0]
    candidate = {key: raw[key] for key in FIXED_BAND_9D}
    for key in FIXED_BAND_9D:
        candidate[key] = float(candidate[key])
    # The only damping design inputs for this workflow are the two fixed
    # physical 15 mm widths.  They are not optimisation variables.
    band_x = float(raw.get("Damping_Inward_X_m", 0.015))
    band_y = float(raw.get("Damping_Inward_Y_m", 0.015))
    if abs(band_x - 0.015) > 1.0e-12 or abs(band_y - 0.015) > 1.0e-12:
        raise RuntimeError("FIXED_BAND_15MM_VALIDATION_REQUIRES_X15MM_Y15MM")
    damping = fixed_band_width_to_rx_ry(candidate, band_x, band_y)
    full_geometry = dict(config.FIXED_BASELINE_PARAMETERS)
    full_geometry.update(candidate)
    shell_dimensions = shell_outer_dimensions(full_geometry)
    candidate.update({
        "Candidate_ID": candidate_id,
        "Target_Frequency_Hz": float(raw["Target_Frequency_Hz"]),
        "Predicted_Frequency_Hz": float(raw["Predicted_Frequency_Hz"]),
        "Predicted_STD_Hz": float(raw.get("Predicted_STD_Hz", "nan")),
        "Predicted_Raw_Mode": int(float(raw["Predicted_Mode_Index"])),
        "Nearest_Database_Geometry_Distance": float(raw.get("Nearest_Training_Distance", "nan")),
        "Damping_Inward_X_m": band_x,
        "Damping_Inward_Y_m": band_y,
        "Damping_Configuration": damping["Damping_Configuration"],
        "rx": damping["rx"],
        "ry": damping["ry"],
        "ABH_Shell_Overall_Length_X_m": shell_dimensions["ABH_Shell_Overall_Length_X_m"],
        "ABH_Shell_Overall_Width_Y_m": shell_dimensions["ABH_Shell_Overall_Width_Y_m"],
        "Validation_Model": "FIXED_BAND_15MM",
        "Data_Library": FIXED_BAND_LIBRARY,
    })

    validation_csv = os.path.join(output_dir, "fixed_band_15mm_abaqus_validation.csv")
    if os.path.isfile(validation_csv):
        with open(validation_csv, newline="", encoding="utf-8-sig") as stream:
            old = [row for row in csv.DictReader(stream)
                   if str(row.get("Candidate_ID", "")).strip() == candidate_id
                   and row.get("Status") == "SUCCESS"]
        if old:
            write_fixed_band_user_validation_table(validation_csv, [candidate_id])
            result = dict(old[-1])
            result.update({"Status": "ALREADY_VALIDATED_NO_RESUBMIT",
                           "Abaqus_Job_Submitted": False,
                           "Validation_CSV": validation_csv})
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return result

    run_root = ensure_directory(os.path.join(RUNS, "fixed_band_15mm_validation"))
    state_dir = ensure_directory(os.path.join(run_root, "_state", candidate_id))
    checkpoint = os.path.join(state_dir, "checkpoint.json")
    state = read_json(checkpoint) or {
        "Status": "STARTED", "Jobs": {}, "Validation": {}, "Freed_Bytes": 0,
        "Candidate_CSV": os.path.abspath(candidate_csv), "Candidate_ID": candidate_id,
        "Validation_Model": "FIXED_BAND_15MM",
    }
    # This campaign has its own one-at-a-time ledger.  It must not inherit the
    # exhausted historical D02 campaign budget.
    fixed_policy = dict(POLICY)
    fixed_policy["max_jobs"] = 1
    fixed_policy["enforce_global_request_limit"] = False
    runtime = {"run_root": run_root, "output_dir": output_dir,
               "output_name": os.path.basename(validation_csv),
               "policy": fixed_policy, "preserve_large_artifacts": True}
    try:
        record = validate_one(candidate, state, state_dir, runtime=runtime)
        state["Status"] = record.get("Status", "UNKNOWN")
    except Exception as exc:
        import traceback
        state.update({"Status": "STOPPED_WITH_EVIDENCE", "Error": str(exc),
                      "Traceback": traceback.format_exc()})
        record = {"Candidate_ID": candidate_id, "Status": state["Status"],
                  "Error": state["Error"]}
    finally:
        # Persist physical band inputs and builder-derived ratios with every
        # result, so 15 mm can never be mistaken for a fixed rx/ry ratio.
        record.update({"Damping_Inward_X_m": band_x,
                       "Damping_Inward_Y_m": band_y,
                       "rx_for_v2_builder": damping["rx"],
                       "ry_for_v2_builder": damping["ry"],
                       "ABH_Shell_Overall_Length_X_m": shell_dimensions["ABH_Shell_Overall_Length_X_m"],
                       "ABH_Shell_Overall_Width_Y_m": shell_dimensions["ABH_Shell_Overall_Width_Y_m"],
                       "Damping_Configuration": damping["Damping_Configuration"],
                       "Validation_Model": "FIXED_BAND_15MM"})
        state.setdefault("Validation", {})[candidate_id] = record
        save_json(checkpoint, state)
        # Rebuild from all one-candidate checkpoints so an individual run
        # cannot overwrite an earlier validation result.
        validation_csv, _ = rebuild_fixed_band_validation_ledger()
        write_fixed_band_user_validation_table(validation_csv, [candidate_id])

    submitted = bool(state.get("Jobs", {}).get(candidate_id, {}).get("PID"))
    result = dict(record, Abaqus_Job_Submitted=submitted,
                  Validation_CSV=validation_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def validate_fixed_band_saved_candidates_batch(candidate_ids, candidate_csv=None,
                                                retry_previously_attempted=False):
    """按用户明确给出的顺序，逐个验证固定 15 mm 阻尼带候选。

    本函数不生成候选、不训练模型。每次调用底层单候选验证入口，因此保持既有
    V2 几何、材料、网格和 Abaqus 提交流程不变。完成一个候选后立即保存批次
    台账；中断后再次运行会根据正式验证台账跳过已经成功或已尝试的 ID。
    """
    ids = [str(item).strip() for item in candidate_ids if str(item).strip()]
    if not ids:
        raise RuntimeError("BATCH_CANDIDATE_IDS_REQUIRED")
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise RuntimeError("BATCH_CANDIDATE_IDS_NOT_UNIQUE:" + ",".join(duplicates))

    output_dir = ensure_directory(os.path.join(
        BASE_OUTPUT, "fixed_band_15mm", "generated"))
    if candidate_csv:
        source_csv = candidate_csv
    else:
        source_csv = os.path.join(output_dir, "fixed_band_15mm_exploratory_candidates.csv")
    if not os.path.isfile(source_csv):
        raise RuntimeError("FIXED_BAND_BATCH_CANDIDATE_CSV_NOT_FOUND:" + source_csv)

    with open(source_csv, newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))
    # 在提交任何 Abaqus Job 前验证候选CSV物理字段。曾出现把远距离候选复制
    # 到旧表后因字段列顺序不同而错位的情况；绝不能把rx/ry误作15 mm带宽。
    required_fields = set(FIXED_BAND_9D) | {
        "Candidate_ID", "Target_Frequency_Hz", "Predicted_Frequency_Hz",
        "Predicted_Mode_Index", "Damping_Inward_X_m", "Damping_Inward_Y_m",
    }
    missing_fields = sorted(required_fields - set(source_rows[0] if source_rows else {}))
    if missing_fields:
        raise RuntimeError("FIXED_BAND_CANDIDATE_CSV_SCHEMA_MISSING_FIELDS:" + ",".join(missing_fields))
    source_ids = [str(row.get("Candidate_ID", "")).strip() for row in source_rows]
    missing = [item for item in ids if source_ids.count(item) != 1]
    if missing:
        raise RuntimeError("BATCH_CANDIDATE_ID_NOT_FOUND_OR_NOT_UNIQUE:" + ",".join(missing))
    selected_rows = [next(row for row in source_rows if str(row.get("Candidate_ID", "")).strip() == item)
                     for item in ids]
    malformed = []
    for row in selected_rows:
        try:
            band_x = float(row["Damping_Inward_X_m"])
            band_y = float(row["Damping_Inward_Y_m"])
        except (TypeError, ValueError):
            malformed.append(str(row.get("Candidate_ID", "")) + ":NON_NUMERIC_DAMPING_WIDTH")
            continue
        if abs(band_x - 0.015) > 1.0e-12 or abs(band_y - 0.015) > 1.0e-12:
            malformed.append("%s:X=%g,Y=%g" % (row["Candidate_ID"], band_x, band_y))
    if malformed:
        raise RuntimeError("FIXED_BAND_CANDIDATE_CSV_SCHEMA_OR_DAMPING_MISMATCH:" + ";".join(malformed))

    ledger_path = os.path.join(output_dir, "fixed_band_15mm_abaqus_validation.csv")
    existing = {}
    if os.path.isfile(ledger_path):
        with open(ledger_path, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                item = str(row.get("Candidate_ID", "")).strip()
                if item:
                    existing[item] = dict(row)

    batch_csv = os.path.join(output_dir, "fixed_band_15mm_batch_validation_summary.csv")
    checkpoint_path = os.path.join(output_dir, "fixed_band_15mm_batch_validation_checkpoint.json")
    checkpoint = read_json(checkpoint_path) or {}
    checkpoint.update({
        "Status": "RUNNING", "Candidate_CSV": os.path.abspath(source_csv),
        "Candidate_IDs": ids, "Retry_Previously_Attempted": bool(retry_previously_attempted),
        "Started_UTC": checkpoint.get("Started_UTC", datetime.now(timezone.utc).isoformat()),
    })
    records = []

    for sequence, candidate_id in enumerate(ids, 1):
        prior = existing.get(candidate_id)
        if prior and prior.get("Status") == "SUCCESS":
            result = dict(prior, Status="ALREADY_VALIDATED_NO_RESUBMIT",
                          Abaqus_Job_Submitted=False)
            action = "SKIPPED_SUCCESS_ALREADY_RECORDED"
        elif prior and not retry_previously_attempted:
            result = dict(prior, Abaqus_Job_Submitted=False)
            action = "SKIPPED_PREVIOUSLY_ATTEMPTED"
        else:
            action = "SUBMITTED_SEQUENTIALLY"
            try:
                result = validate_fixed_band_saved_candidate(candidate_id, source_csv)
            except Exception as exc:
                # 保留当前批次其它候选继续执行；错误写入批次台账供人工排查。
                result = {"Candidate_ID": candidate_id, "Status": "BATCH_ENTRY_ERROR",
                          "Error": str(exc), "Abaqus_Job_Submitted": False}
                action = "NOT_SUBMITTED_ENTRY_ERROR"

        row = dict(result)
        row.update({"Batch_Sequence": sequence, "Batch_Action": action,
                    "Batch_Candidate_CSV": os.path.abspath(source_csv),
                    "Retry_Previously_Attempted": bool(retry_previously_attempted)})
        records.append(row)
        checkpoint.update({"Completed_Count": len(records), "Last_Candidate_ID": candidate_id,
                           "Last_Status": row.get("Status", "UNKNOWN"), "Records": records})
        save_json(checkpoint_path, checkpoint)
        write_csv(batch_csv, records)

    submitted = sum(bool(row.get("Abaqus_Job_Submitted")) for row in records)
    checkpoint.update({"Status": "COMPLETED", "Completed_UTC": datetime.now(timezone.utc).isoformat(),
                       "Abaqus_Jobs_Submitted": submitted})
    save_json(checkpoint_path, checkpoint)
    user_validation_path = write_fixed_band_user_validation_table(ledger_path, ids)
    qualified_pool = write_fixed_band_frequency_qualified_pool(records, ids)
    payload = {"Status": "BATCH_VALIDATION_COMPLETED", "Requested_Candidates": len(ids),
               "Abaqus_Jobs_Submitted": submitted, "Batch_Summary_CSV": batch_csv,
               "Checkpoint": checkpoint_path, "Validation_Ledger": ledger_path,
               "User_Validation_Table": user_validation_path,
                "Frequency_Qualified_Pool": qualified_pool,
               "Sequential": True}
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_report(state, data, comparison, candidates):
    chosen=next((r for r in comparison if r["Model"]==state.get("Selected_Model") and
                 r["Scope"]=="COMMON_FIXED_D02" and r["Subset"]=="ALL_MODES_1_8"),{})
    lines=["# 9D 固定阻尼多解频率反向设计", "", "状态："+state["Status"],
           "错误/停止原因："+state.get("Error","无"), "", "## 1. 上轮成果保留与暂停",
           "旧 V4 耦合/动态任务已停用；不重新提交旧算例、不读取旧 ODB 做新计算。"]
    for job in state["Previous_Task"]["Jobs"]:
        lines.append("- %s / %s：%s；保留 `%s`。"%(job["Candidate"],job["Kind"],job["Status"],job["Result_Path"]))
    lines += ["旧任务没有完成局部新候选优化；未完成动态结果不作为本阶段训练标签。",
              "", "## 2. 固定阻尼与数据",
              "D02：rx=%.15g，ry=%.15g；15组基础几何均有相同参数，位于既有范围中部。D04并非共同固定构形。"%tuple(data["ref"]),
              "保留 V2 几何、3 mm阻尼层、约束层及独立夹持边界定义；材料参数以当前 config 的 ABH_Frequency_Dataset_V2_2 对齐值为准。",
              "数据库：%d个样本、%d组几何、%d条模态。以几何坐标哈希分组，整组阻尼方案只进入同一折。"%(len(data["ids"]),len(set(data["groups"])),data["mode_rows"]),
              "预测前8个排序固有频率（全部79样本完整）；其余605条高阶模态仅用于数据库参照，因500 Hz截断造成样本不齐，不混入完整谱训练。",
              "排序编号不是固定物理模态；未声称解决mode veering/crossing。新算例同时报告同排序模态误差与全部模态最近目标误差。",
              "", "## 3. 真实分组验证与模型选择",
              "A只用D02；B用11维条件GPR；C用成对阻尼差异校正，再在组平均固定参考谱上拟合9维GPR；D为ExtraTrees对照。",
              "所有scaler、核超参数和分层校正仅在各训练折拟合。公平选择基于相同15个D02组的5折外推结果，其他阻尼79点指标单独列出。",
              "模型选择使用这轮CV，不能再将选择后的CV称为独立最终性能证明；新Abaqus验证始终未参与训练。",
              "|模型|共同D02 MAPE %|MAE Hz|P90 %|最大相对误差 %|95%覆盖率 %|",
              "|---|---:|---:|---:|---:|---:|"]
    for r in comparison:
        if r["Scope"]=="COMMON_FIXED_D02" and r["Subset"]=="ALL_MODES_1_8":
            lines.append("|%s|%.3f|%.3f|%.3f|%.3f|%.1f|"%(r["Model"],r["MAPE_Percent"],r["MAE_Hz"],r["P90_Relative_Error_Percent"],r["Max_Relative_Error_Percent"],r["Coverage_95_Percent"]))
    lines += ["当前选择："+state.get("Selected_Model","未完成"),
              "GPR以log(Hz)训练，normalize_y仅由sklearn处理一次；STD按lognormal公式转换到Hz，不对STD加mean。树的树间离散度和分层组合STD都不是校准置信区间。",
              "所选模型共同D02验证的95%%区间覆盖率为%s%%；不能把小STD等同于达到±2%%目标。"%chosen.get("Coverage_95_Percent","NOT_AVAILABLE"),
              "uncertainty不是唯一筛点依据：同时限制归一化几何距离、联合几何有效性、组CV误差及候选多样性。",
              "", "## 4. 数据库外新构形与Abaqus验证",
              "多起点连续空间搜索；新几何距数据库至少%.3f，候选间至少%.3f（按既有物理bounds归一化的9维欧氏距离）；至少两个变量变化2%%范围。"%(POLICY["min_database_distance"],POLICY["min_pair_distance"]),
              "所有候选预测在提交前固定。以下±2%只表示频率匹配，不表示动态减振。"]
    warnings_count=sum(c.get("Surrogate_Mode_Order_Warning")=="PREDICTED_BRANCH_ORDER_INVERSION" for c in candidates)
    lines.append("%d个候选预测谱存在局部排序交叉，保留警告，不强制赋予物理模态身份。"%warnings_count)
    for target in state["Targets"]:
        generated=[c for c in candidates if c["Target_Frequency_Hz"]==target]
        tested=[v for v in state["Validation"].values() if v["Target_Frequency_Hz"]==target]
        passed=[v for v in tested if v.get("Frequency_Match_PASS")]
        lines.append("### %.2f Hz：生成%d，验证%d，±2%%合格%d"%(target,len(generated),len(tested),len(passed)))
        for v in tested:
            lines.append("- %s：预测 %.5f Hz；Abaqus %s Hz；目标误差 %s%%；%s。"%(v["Candidate_ID"],v["Predicted_Frequency_Hz"],v.get("Abaqus_Frequency_Hz","未获得"),v.get("Frequency_Error_Percent","未获得"),v.get("Mode_Match_Status",v["Status"])))
            if v.get("Frequency_Match_PASS"):
                lines.append("  9维参数："+json.dumps({k:v[k] for k in GEOMETRY},ensure_ascii=False)+"；rx/ry固定为D02。")
    lines += ["", "## 5. 限制、接口和结论",
              "共使用%d/12个新独立模态作业。先试探3点；若全都失败或目标误差>5%%，停止剩余求解。5%%是本任务显式试探停止阈值，不是论文标准。"%len(state["Jobs"]),
              "相比数据库检索，新增连续参数空间内数据库外几何生成、预测不确定性记录和真实独立模态复核；未保证任意频率均可生成合格设计。",
              "34组独立几何对9维空间仍稀疏。建议以后按独立几何增加数据，重点覆盖误差大/距离远区域；本轮不自动开展DOE。",
              "只有多个Abaqus新点均通过时才能称多个合格构形。未验证候选只保留预测，不算成功。",
              "接口：python main.py默认RETRIEVAL；--design-mode GENERATE不运行Abaqus；VALIDATE仅验证已冻结候选；GENERATE_AND_VALIDATE显式启用独立模态。",
              "用户只需在00_USER_DESIGN/design_request.py修改TARGET_FREQUENCY_HZ；新增目标输出使用独立子目录，原50 Hz结果不覆盖。本阶段全局12次独立求解预算不会因更换目标重置。",
              "旧FULL耦合入口已按本次研究优先级禁用。V2几何、正式数据库和原检索排序不变。",
              "10–400动态/PSD、Host耦合、fe-safe、疲劳、其他工程结构：本轮均未运行。",
              "35 Hz原检索回归："+str(state.get("Retrieval_35Hz","NOT_CHECKED")),
              "", "## 文件",
              "- frequency_surrogate_comparison.csv",
              "- new_abh_candidates.csv",
              "- new_abh_abaqus_validation.csv"]
    with open(os.path.join(OUTPUT,"9D_FREQUENCY_INVERSE_DESIGN_REPORT.md"),"w",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")


def run_frequency_design(mode, target=50., additional=()):
    """Run only the requested isolated workflow, with persistent 12-job budget."""
    if mode not in ("GENERATE","VALIDATE","GENERATE_AND_VALIDATE"):raise ValueError("UNKNOWN_DESIGN_MODE")
    global OUTPUT
    targets=list(dict.fromkeys([float(target)]+[float(t) for t in additional]))
    default_request=targets==[50.,34.41,129.30,213.63]
    tag="target_"+str(float(target)).replace(".","p")+"_"+hashlib.sha256(json.dumps(targets).encode()).hexdigest()[:8]
    OUTPUT=BASE_OUTPUT if default_request else os.path.join(BASE_OUTPUT,tag)
    ensure_directory(OUTPUT); ensure_directory(RUNS)
    root_state=ensure_directory(os.path.join(RUNS,"_state"))
    state_dir=root_state if default_request else ensure_directory(os.path.join(root_state,tag))
    import msvcrt
    lock=open(os.path.join(root_state,"controller.lock"),"a+b")
    lock.seek(0);lock.write(b"0");lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    checkpoint=os.path.join(state_dir,"checkpoint.json")
    state=read_json(checkpoint) or {"Status":"STARTED","Jobs":{},"Validation":{},"Freed_Bytes":0,
                                "Targets":targets}
    data=load_training_data(); rows=[]; candidates=[]
    try:
        if state["Targets"]!=targets:raise ValueError("REQUEST_DIFFERS_FROM_FROZEN_CHECKPOINT")
        state.pop("Error",None)
        cache=os.path.join(root_state,"model_bundle.pkl")
        dbhash=hashlib.sha256(open(MODAL_DATABASE,"rb").read()).hexdigest()
        with threadpool_limits(limits=1):
            if os.path.isfile(cache):
                with open(cache,"rb") as f: bundle=pickle.load(f)
                if bundle["Database_SHA256"]!=dbhash:raise RuntimeError("TRAINING_DATABASE_CHANGED")
                model,rows,diagnostics=bundle["model"],bundle["rows"],bundle["diagnostics"]
            else:
                model,rows,diagnostics=compare_surrogates(data)
                with open(cache,"wb") as f:pickle.dump({"Database_SHA256":dbhash,"model":model,"rows":rows,"diagnostics":diagnostics},f)
            state["Selected_Model"]=model.name
            write_csv(os.path.join(OUTPUT,"frequency_surrogate_comparison.csv"),rows)
            frozen=os.path.join(state_dir,"candidates_before_abaqus.json")
            if os.path.isfile(frozen):candidates=read_json(frozen)
            elif mode=="VALIDATE":raise RuntimeError("NO_FROZEN_CANDIDATES_RUN_GENERATE_FIRST")
            else:
                for t in state["Targets"]:
                    candidates.extend(generate_candidates(t,model,data,diagnostics))
                    print("GENERATION_COMPLETED %.2f"%t,flush=True)
                save_json(frozen,candidates)
            for candidate in candidates:
                spectrum=np.array(json.loads(candidate["Predicted_Ordered_Frequencies_Hz"]))
                candidate["Surrogate_Mode_Order_Warning"]="PREDICTED_BRANCH_ORDER_INVERSION" if np.any(np.diff(spectrum)<0) else "NONE"
            write_csv(os.path.join(OUTPUT,"new_abh_candidates.csv"),candidates)
            if mode!="GENERATE":
                primary=[c for c in candidates if c["Target_Frequency_Hz"]==target]
                if len(primary)<3:raise RuntimeError("FEWER_THAN_THREE_TRUST_REGION_CANDIDATES_NO_FE_SUBMISSION")
                probes=[validate_one(c,state,state_dir) for c in primary[:3]]
                if all(p.get("Frequency_Error_Percent",float("inf"))>POLICY["probe_stop_percent"] for p in probes):
                    state["Status"]="STOPPED_INITIAL_PROBES_UNRELIABLE"
                    state["Error"]="All first three new points exceed 5% target error or failed; no further jobs"
                else:
                    for c in primary[3:6]:validate_one(c,state,state_dir)
                    for t in targets[1:]:
                        for c in [c for c in candidates if c["Target_Frequency_Hz"]==t][:2]:validate_one(c,state,state_dir)
                    state["Status"]="COMPLETED_BOUNDED_INDEPENDENT_VALIDATION"
            else:state["Status"]="GENERATED_NOT_ABAQUS_VALIDATED"
        from types import SimpleNamespace
        req=SimpleNamespace(TARGET_FREQUENCY_HZ=35.,FREQUENCY_TOLERANCE_PERCENT=2.,TARGET_DIRECTION="Z",USE_MASS_FILTER=False,MAX_MASS=None)
        ids={r["Sample_ID"] for r in retrieve(req)}
        state["Retrieval_35Hz"]="PASS" if {"FORMAL11D_B1_A009_D02","FORMAL11D_B1_A014_D01"}<=ids else "FAIL"
    except Exception as exc:
        import traceback
        state.update({"Status":"STOPPED_WITH_EVIDENCE","Error":str(exc),"Traceback":traceback.format_exc()})
    finally:
        save_json(checkpoint,state)
        write_csv(os.path.join(OUTPUT,"new_abh_abaqus_validation.csv"),list(state["Validation"].values()))
        write_report(state,data,rows,candidates)
        lock.close()
    print(json.dumps({"Status":state["Status"],"Error":state.get("Error"),"New_Isolated_Jobs":len(state["Jobs"]),"Output":OUTPUT},ensure_ascii=False),flush=True)
    return state


def audit_manual_input(reference, manual):
    """Strict ordered physical-keyword comparison, allowing two export roundoffs."""
    import re
    def physical(path):
        rows=[]
        for line in open(path,encoding="latin-1"):
            s=line.strip().upper()
            if not s or s.startswith("**"):continue
            if s.startswith("*OUTPUT"):break  # Only terminal output requests differ.
            s=re.sub(r'(\*TIE, NAME=)CONSTRAINT-1-1(?=,|$)',r'\1CONSTRAINT-1',s)
            rows.append(s)
        return rows
    a,b=physical(reference),physical(manual)
    if len(a)!=len(b):raise RuntimeError("MANUAL_PHYSICAL_BLOCK_LENGTH_DIFFERS")
    context=""; rounding=[]
    for i,(x,y) in enumerate(zip(a,b)):
        if x.startswith("*"):context=x
        if x==y:continue
        if context!="*NODAL THICKNESS":raise RuntimeError("MANUAL_PHYSICAL_DIFFERENCE_AT_%d:%s/%s"%(i,x,y))
        xa,xb=x.split(','),y.split(',')
        if xa[0]!=xb[0] or len(xa)!=2 or len(xb)!=2:raise RuntimeError("THICKNESS_MAPPING_CHANGED")
        delta=abs(float(xa[1])-float(xb[1]))
        if delta>1.01e-8 or delta/abs(float(xa[1]))>5e-6:raise RuntimeError("THICKNESS_EXPORT_DIFF_TOO_LARGE")
        rounding.append({"Node":xa[0],"Old":float(xa[1]),"Manual":float(xb[1]),"Difference_m":delta})
    return {"Equivalent_Within_Export_Precision":True,"Physical_Lines_Compared":len(a),
            "Thickness_Rounding":rounding,"Output_Difference":"History ALLKE/ALLSE omitted; S/U output order changed",
            "Reference_SHA256":hashlib.sha256(open(reference,"rb").read()).hexdigest(),
            "Manual_SHA256":hashlib.sha256(open(manual,"rb").read()).hexdigest()}


def _staged_process(command, cwd, logpath, monitor_policy=None):
    """Run a single owned process tree to completion, never retry."""
    import time
    env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
    env["PYTHONPATH"]=os.pathsep.join([PROJECT_ROOT,os.path.join(PROJECT_ROOT,"01_CORE"),DATABASE_DIR])
    for key in ("TEMP","TMP","ABAQUS_TEMP","ABAQUS_SCRATCH"):env[key]=ensure_directory(ABAQUS_TEMP_DIR)
    start=time.monotonic(); samples=[]; emergency_count=0
    with open(logpath,"w",encoding="utf-8") as log:
        p=subprocess.Popen(["cmd","/c",subprocess.list2cmdline(command)],cwd=cwd,env=env,stdout=log,stderr=log)
        try:
            while p.poll() is None:
                if time.monotonic()-start>config.TARGET_DRIVEN_POLICY["job_timeout_seconds"]:raise RuntimeError("STAGED_TIMEOUT")
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if monitor_policy:
                        s=_windows_memory_snapshot()
                        s["Disk_Free_GB"]=shutil.disk_usage(cwd).free/1024**3
                        # Preserve the resource observation even if process telemetry fails.
                        save_json(logpath+".latest_resource.json",s)
                        s["Process_Tree"]=_owned_process_snapshot(p.pid);samples.append(s)
                        emergency_count=_resource_emergency_count(s,monitor_policy,emergency_count)
                        s["Resource_Emergency_Consecutive_Samples"]=emergency_count
                        with open(logpath+".resources.jsonl","a",encoding="utf-8") as h:h.write(json.dumps(s)+"\n")
                        if s["Disk_Free_GB"]<1.0:raise RuntimeError("DISK_SAFETY_ABORT:"+str(s))
                        if emergency_count>=monitor_policy.get("emergency_consecutive_samples",3):
                            raise RuntimeError("RESOURCE_SAFETY_ABORT:"+str(s))
            if p.returncode:raise RuntimeError("STAGED_PROCESS_EXIT_%s:%s"%(p.returncode,logpath))
        except Exception:
            if p.poll() is None:
                subprocess.run(["taskkill","/PID",str(p.pid),"/T","/F"],stdout=log,stderr=log);p.wait(timeout=30)
            raise
    return {"Elapsed_Seconds":time.monotonic()-start,"Command":command,"Resource_Samples":samples}


def _staged_worker(request, folder):
    path=os.path.join(folder,request["Stage"].lower()+"_request.json");save_json(path,request)
    worker=os.path.join(DATABASE_DIR,"v2_pilot_replay_worker.py")
    command=(["abaqus","cae","noGUI="+worker,"--"] if request["Stage"]=="BUILD" else ["abaqus","python",worker])
    _staged_process(command+["--design-request",path],PROJECT_ROOT,os.path.join(folder,request["Stage"].lower()+".log"))
    result=read_json(request["Result_Path"])
    if not result or result["Status"] not in ("INP_READY","SUCCESS"):raise RuntimeError("STAGED_WORKER_FAILED:"+str(result))
    return result


def run_staged_frequency_validation(mode="standard", candidate_id=None, retry_authorization=None):
    """Reuse manual evidence, then serial frozen cases; total historical budget 12."""
    import msvcrt,re
    from paths import TARGET_DRIVEN_RUNS_DIR as root, RESULTS_DIR
    sd=ensure_directory(os.path.join(root,"staged_automation"));cp=os.path.join(sd,"checkpoint.json")
    lock=open(os.path.join(root,"_state","controller.lock"),"a+b");lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=read_json(cp) or {"Prior_Failed_Submissions":4,"Manual_Submissions":1,"Jobs":{},"Results":{},"Status":"PREPARED"}
    try:
        if mode not in ("standard","manual_equivalent_single","user_selected_single","extract_only"):raise ValueError("UNKNOWN_SUBMISSION_MODE")
        if mode=="manual_equivalent_single":
            if candidate_id!="TD_T_bb51617d747b":raise ValueError("SINGLE_MODE_REQUIRES_AUTHORIZED_CANDIDATE")
            existing=state["Jobs"].get(candidate_id)
            if not existing or "Build_Result" not in existing:raise RuntimeError("SINGLE_MODE_REQUIRES_EXISTING_INP")
            if existing.get("Submitted"):
                token="TD_T_bb51617d747b_EXTRA_ONE_20261012"
                if retry_authorization!=token or state.get("Consumed_Retry_Authorization")==token:
                    raise RuntimeError("SINGLE_MODE_ALREADY_SUBMITTED_NO_RETRY")
                if not existing.get("Stop_Reason","").startswith("MONITOR_ENCODING_FAILURE"):
                    raise RuntimeError("RETRY_NOT_FOR_EXPECTED_MONITOR_FAILURE")
                policy=dict(config.STAGED_ISOLATED_RESOURCES,minimum_available_memory_gb=0.0,minimum_commit_headroom_gb=0.0)
                _recovery_resource_check(policy)
                source=existing["Build_Result"]["INP_Path"]
                if hashlib.sha256(open(source,"rb").read()).hexdigest()!=existing["INP_SHA256"]:
                    raise RuntimeError("RETRY_INPUT_CHANGED")
                folder=os.path.join(sd,candidate_id,"authorized_retry_1")
                if os.path.exists(folder):raise RuntimeError("RETRY_DIRECTORY_ALREADY_EXISTS")
                ensure_directory(os.path.join(folder,"run"))
                request=dict(existing["Request"],Run_Directory=os.path.join(folder,"run"),
                             Job_Name="auto_"+candidate_id+"_retry1",Result_Path=os.path.join(folder,"build_result.json"))
                state["Jobs"][candidate_id+"__attempt1"]=existing
                state["Jobs"][candidate_id]={"Submitted":False,"Request":request,
                    "Build_Result":existing["Build_Result"],"INP_SHA256":existing["INP_SHA256"]}
                state["Consumed_Retry_Authorization"]=token
                save_json(cp,state)
        if mode in ("user_selected_single","extract_only") and not candidate_id:
            raise ValueError("USER_SELECTED_CANDIDATE_ID_REQUIRED")
        state["Submission_Mode"]=mode
        state.pop("Error",None);state["Status"]="RUNNING"
        historical=read_json(os.path.join(root,"_state","mesh_memory_checkpoint.json"))
        six=read_json(os.path.join(root,"_state","single_memory6gb_checkpoint.json"))
        if historical["Prior_Submissions"]+len(historical["Attempts"])!=4 or six.get("Submitted"):
            raise RuntimeError("HISTORICAL_BUDGET_REQUIRES_RECONCILIATION")
        frozen=os.path.join(root,"_state","frozen_candidates.json")
        if hashlib.sha256(open(frozen,"rb").read()).hexdigest()!=read_json(os.path.join(root,"_state","checkpoint.json"))["Frozen_SHA256"]:raise RuntimeError("FROZEN_CHANGED")
        candidates=read_json(frozen);r=next(c for c in candidates if c["Candidate_ID"]=="TD_R_466597d04ebc")
        if mode in ("user_selected_single","extract_only"):
            selected=next((c for c in candidates if c["Candidate_ID"]==candidate_id),None)
            if selected is None:raise RuntimeError("USER_SELECTED_CANDIDATE_NOT_FROZEN")
            if not selected.get("FE_Selected"):raise RuntimeError("USER_SELECTED_CANDIDATE_NOT_FE_SELECTED")
            if mode=="user_selected_single" and state["Jobs"].get(candidate_id,{}).get("Submitted"):
                raise RuntimeError("USER_SELECTED_CANDIDATE_ALREADY_SUBMITTED_NO_RETRY")
            if mode=="extract_only" and not state["Jobs"].get(candidate_id,{}).get("Submitted"):
                raise RuntimeError("EXTRACT_ONLY_REQUIRES_SUBMITTED_CANDIDATE")
        old=os.path.join(historical["Attempts"][0]["Run_Directory"],"freq9_R_466597d04ebc.inp")
        manual=r"E:\temp\Job-4"
        state["Manual_Equivalence"]=audit_manual_input(old,manual+".inp")
        com=open(manual+".com",encoding="latin-1").read()
        for pattern in (r"'cpus':1,",r"'memory':'90%',",r"'mp_mode':THREADS,",r"'lanczos':ON,"):
            if not re.search(pattern,com):raise RuntimeError("MANUAL_COMMAND_POLICY_DIFFERS")
        if state["Results"].get(r["Candidate_ID"],{}).get("Raw",{}).get("Extraction_Schema")!=2:
            copied=ensure_directory(os.path.join(sd,"manual_Job4_extract"))
            for ext in (".odb",".log",".msg"):
                dest=os.path.join(copied,"Job-4"+ext)
                if not os.path.isfile(dest):shutil.copy2(manual+ext,dest)
            req={"Stage":"EXTRACT","Candidate":r,"Run_Directory":copied,"ODB_Path":os.path.join(copied,"Job-4.odb"),"Result_Path":os.path.join(copied,"result.json")}
            raw=_staged_worker(req,copied)
            state["Results"][r["Candidate_ID"]]={"Candidate":r,"Raw":raw,"Evaluation":summarize_validation(r,raw),"Provenance":"MANUAL_JOB4_REUSED_NO_RESUBMIT"}
            save_json(cp,state)
        fair=[c for c in candidates if c["Trial_Class"]=="FAIR_9" and c["Candidate_ID"]!=r["Candidate_ID"]]
        # Give all three target frequencies evidence before completing per-target trios.
        fair.sort(key=lambda c:(0 if c["Target_Frequency_Hz"]!=34.41 and c["Method"]=="T" else 1,c["Target_Frequency_Hz"],c["Priority"]))
        if mode=="manual_equivalent_single":fair=[c for c in fair if c["Candidate_ID"]==candidate_id]
        if mode in ("user_selected_single","extract_only"):fair=[selected]
        resources=dict(config.STAGED_ISOLATED_RESOURCES)
        if mode in ("manual_equivalent_single","user_selected_single"):
            # User-authorized single-job override only; keep conflict/disk checks
            # and the existing in-run 2 GiB RAM / 3 GiB commit emergency guard.
            resources.update(minimum_available_memory_gb=0.0,minimum_commit_headroom_gb=0.0)
        for c in fair:
            sid=c["Candidate_ID"]
            if sid in state["Results"]:continue
            submissions=sum(bool(j.get("Submitted")) for j in state["Jobs"].values())
            j=state["Jobs"].get(sid)
            if 5+submissions>=12 and not (j and j.get("Submitted")):break
            folder=ensure_directory(os.path.join(sd,sid));run=os.path.join(folder,"run")
            req={"Stage":"BUILD","Candidate":c,"Run_Directory":run,"Job_Name":"auto_"+sid,
                 "Result_Path":os.path.join(folder,"build_result.json")}
            if not j:
                if os.path.exists(run):raise RuntimeError("UNREGISTERED_RUN_PATH_EXISTS")
                state["Jobs"][sid]={"Submitted":False,"Request":req};save_json(cp,state);j=state["Jobs"][sid]
            else:
                req=j["Request"];run=req["Run_Directory"];folder=os.path.dirname(req["Result_Path"])
            if "Build_Result" not in j:
                # Geometry generation does not submit FE; leave CAE before solving.
                built=_staged_worker(req,folder);j["Build_Result"]=built
                j["INP_SHA256"]=hashlib.sha256(open(built["INP_Path"],"rb").read()).hexdigest();save_json(cp,state)
            built=j["Build_Result"];scratch=ensure_directory(os.path.join(folder,"scratch"))
            if hashlib.sha256(open(built["INP_Path"],"rb").read()).hexdigest()!=j["INP_SHA256"]:
                raise RuntimeError("GENERATED_INPUT_CHANGED")
            command=["abaqus","job="+req["Job_Name"],"input="+built["INP_Path"],"cpus=1","mp_mode=threads","memory=90%","scratch="+scratch,"interactive"]
            if j.get("Submitted") and mode!="extract_only":
                log_path=os.path.join(run,req["Job_Name"]+".log")
                if (not os.path.isfile(log_path) or " COMPLETED" not in open(log_path,errors="replace").read() or
                        os.path.exists(os.path.join(run,req["Job_Name"]+".lck"))):
                    raise RuntimeError("PREVIOUS_SUBMISSION_NOT_COMPLETE_NO_AUTOMATIC_RETRY")
            elif not j.get("Submitted"):
                try:j["Preflight"]=_recovery_resource_check(resources)
                except RuntimeError as exc:
                    state.update(Status="PRECONDITION_BLOCKED",Error=str(exc));break
                if any(os.path.exists(os.path.join(run,req["Job_Name"]+ext)) for ext in (".odb",".lck",".sta",".msg")):
                    raise RuntimeError("STALE_SOLVER_OUTPUT_BEFORE_SUBMISSION")
                import time
                j.update(Submitted=True,Status="SUBMITTED",Command=command,Submitted_Epoch=time.time(),Submission_Mode=mode,Resource_Policy=resources);save_json(cp,state)
                j["Execution"]=_staged_process(command,run,os.path.join(folder,"solver_console.log"),resources)
            actual_com=open(os.path.join(run,req["Job_Name"]+".com"),encoding="latin-1").read()
            for pattern in (r"'cpus':1,",r"'memory':'90%',",r"'mp_mode':THREADS,"):
                if not re.search(pattern,actual_com):raise RuntimeError("ACTUAL_SOLVER_POLICY_DIFFERS")
            extract=dict(req,Stage="EXTRACT",Result_Path=os.path.join(folder,"result.json"),Build_Result=built,ODB_Path=os.path.join(run,req["Job_Name"]+".odb"),Submitted_Epoch=j.get("Submitted_Epoch"))
            raw=_staged_worker(extract,folder)
            state["Results"][sid]={"Candidate":c,"Raw":raw,"Evaluation":summarize_validation(c,raw),"Provenance":"NEW_STAGED_AUTOMATION"}
            j["Status"]="SUCCESS";save_json(cp,state)
            from abaqus_cleanup import write_manifest
            write_manifest(os.path.join(folder,"cleanup_inventory.md"),[{"directory":run,"allowed_root":sd,"reason":"Archived complete JSON and modal spectrum"}])
            # No deletion before final evidence review. Retain ODB for this campaign.
        else:state["Status"]="COMPLETED"
        if 5+sum(bool(j.get("Submitted")) for j in state["Jobs"].values())>=12:state["Status"]="BUDGET_EXHAUSTED"
    except Exception as exc:
        state.update(Status="STOPPED_WITH_EVIDENCE",Error=str(exc))
        if candidate_id in state["Jobs"] and state["Jobs"][candidate_id].get("Submitted"):
            state["Jobs"][candidate_id].update(Status="STOPPED_WITH_EVIDENCE",Stop_Reason=str(exc))
    finally:
        save_json(cp,state);lock.close()
        rows=[]
        for v in state["Results"].values():
            rows.append(dict(v["Evaluation"],FE_SOLVER_SUCCESS=True,TARGET_FREQUENCY_MATCHED=v["Evaluation"].get("Frequency_Match_PASS"),Provenance=v["Provenance"]))
        if rows:
            write_csv(os.path.join(sd,"validation.csv"),rows)
            for target in config.TARGET_DRIVEN_POLICY["targets"]:
                selected=[row for row in rows if row["Target_Frequency_Hz"]==target]
                if selected:write_csv(os.path.join(sd,"target_%s_validation.csv"%str(target).replace('.','p')),selected)
        report=["# Automated frequency design — final execution report","", "STATUS = "+state["Status"],
                "Submission mode = "+mode,
                "Single-candidate modes skip only fixed startup RAM/commit gates, retain disk/conflict checks, log low-resource warnings, and stop only after sustained dual-critical RAM/commit exhaustion; no next candidate is selected.",
                "Manual Job-4 equivalence: "+json.dumps(state.get("Manual_Equivalence"),ensure_ascii=False),
                "Total FE submissions: %d/12"%(5+sum(bool(j.get("Submitted")) for j in state["Jobs"].values())),
                "Error: "+str(state.get("Error","NONE")),"", "|Candidate|Target Hz|Actual ODB Hz|Mode|Error %|Solver success|Matched|", "|---|---|---|---|---|---|---|"]
        for row in rows:report.append("|%s|%s|%s|%s|%s|True|%s|"%(row["Candidate_ID"],row["Target_Frequency_Hz"],row.get("Abaqus_Frequency_Hz"),row.get("Abaqus_Raw_Mode"),row.get("Frequency_Error_Percent"),row.get("Frequency_Match_PASS")))
        report += ["", "State and full parameters: "+cp,"Validation CSV: "+os.path.join(sd,"validation.csv"),
            "Resume: python -B -c \"import project_bootstrap; project_bootstrap.activate(); from frequency_generation import run_staged_frequency_validation; run_staged_frequency_validation()\"",
            "No retraining, no resampling, no physical-model changes. Manual original ODB is preserved."]
        report += ["", "## 实际证据与限制",
            "Job-4原始INP/ODB/COM/LOG/MSG位于E:/temp，未改写。两个June日期的exception文件不属于本次October成功求解，不混入本次故障统计。",
            "物理输入在346825行比较中，除两处节点厚度1e-8 m导出舍入及Tie重命名外一致；节点/单元/材料/截面/BC/500Hz求解定义一致。采用导出精度内等价，不称逐字节相同。输出历史ALLKE/ALLSE缺失不影响此频率核验。",
            "根因边界：旧2GB失败有明确限额不足；4GB运行出现整体资源压力；手动90%作业成功。尚未以对照试验证明CAE驻留是唯一根因，不能保证90%自动提交必然成功。",
            "新架构：BUILD只生成INP并退出CAE；调度器独立abaqus job=... input=... cpus=1 mp_mode=threads memory=90% scratch=... interactive；EXTRACT用abaqus python读取ODB。",
            "标准模式提交阈值RAM>=8GiB、提交余量>=14GiB、磁盘>=10GiB；单候选模式仅跳过前两项固定启动门槛。运行中RAM<2GiB或提交余量<3GiB只记录警告；仅当RAM<0.5GiB且提交余量<1GiB连续3次时停止自己的进程树。memory=90%不作为进程树硬上限。",
            "原单进程run_frame路径为其它历史调用保留；新入口不调用其提交/等待分支，不做未经审计的全局删除。",
            "修改文件：01_CORE/config.py；02_ABAQUS/assembly/frame_runner.py；03_DATABASE/v2_pilot_replay_worker.py；04_INVERSE_DESIGN/frequency_generation.py。备份：07_RESULTS/inverse_design/target_driven/history/pre_staged_submission。未新增Python文件。"]
        for sid,v in state["Results"].items():
            raw=v["Raw"]
            report += ["", "### "+sid, "Source = "+v["Provenance"],
                "9D/D02 parameters = "+json.dumps({k:v["Candidate"][k] for k in FEATURES}),
                "ODB frequencies Hz = "+str(raw["All_Positive_Frequencies_Hz"]),
                "Masses kg = "+str(raw["masses"]),
                "Nodes / elements = %s / %s"%(raw["Node_Count"],raw["Element_Count"]),
                "Solver wallclock / reported memory peak = %s s / %s GB"%(raw.get("Solver_Wallclock_Seconds"),raw.get("Solver_Memory_Peak_GB")),
                "Input warnings / analysis warnings / errors = %s / %s / %s"%(raw.get("Input_Warnings"),raw.get("Analysis_Warnings"),raw.get("Error_Count"))]
        for sid,j in state["Jobs"].items():
            report += ["", "### 自动阶段："+sid,"FE Submitted = "+str(j["Submitted"]),
                "INP SHA256 = "+str(j.get("INP_SHA256")),"Build result = "+str(j.get("Build_Result")),
                "Logs = "+os.path.dirname(j["Request"]["Result_Path"])]
        new=sum(bool(j.get("Submitted")) for j in state["Jobs"].values())
        report += ["", "## 当前完成程度",
            "本轮自动新增FE=%d；已登记人工Job-4=1；历史失败=4；总数=%d/12；剩余=%d。"%(new,5+new,7-new),
            "手动作业无全过程RAM/提交监控，不伪造其最低提交余量。自动未提交时的运行时间/峰值/运行安全状态为NOT_AVAILABLE。",
            "34.41 Hz已有独立FE证据但R未命中±2%；其它目标以结果表为准，不能用预测值填补空缺。",
            "新自动端到端成功数=%d；仅BUILD成功不能宣称流程跑通。"%sum(v["Provenance"]=="NEW_STAGED_AUTOMATION" for v in state["Results"].values()),
            "续跑优先未覆盖129.30/213.63Hz的T候选，再补冻结R/A/T；剩余7次不足完成全部8个未验证公平候选，不扩大预算。",
            "资源不足时请用户自行保存并关闭不必要后台应用；不自动关闭程序、不更改分页文件。",
            "成功后先写清理清单；当前原始成功证据及失败诊断均未自动删除。Host耦合/PSD/RMS/fe-safe/新训练：NOT_RUN。"]
        with open(os.path.join(RESULTS_DIR,"AUTOMATED_FREQUENCY_DESIGN_FINAL_REPORT.md"),"w",encoding="utf-8") as h:h.write("\n".join(report)+"\n")
    print(json.dumps({
        "Status": state["Status"],
        "Candidate_ID": candidate_id,
        "Error": state.get("Error"),
        "Total_FE_Submissions": 5+sum(bool(j.get("Submitted")) for j in state["Jobs"].values()),
        "Report": os.path.join(RESULTS_DIR,"AUTOMATED_FREQUENCY_DESIGN_FINAL_REPORT.md"),
    }, ensure_ascii=False), flush=True)
    return state


def run_single_memory6gb_experiment():
    """Explicit single-R continuation; never loops over candidates or retries."""
    import msvcrt
    from paths import TARGET_DRIVEN_RUNS_DIR as root, TARGET_DRIVEN_RESULTS_DIR as out
    sd=os.path.join(root,"_state"); cp=os.path.join(sd,"single_memory6gb_checkpoint.json")
    lock=open(os.path.join(sd,"controller.lock"),"a+b")
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=read_json(cp) or {"Status":"PREPARED","Submitted":False}
    try:
        if state.get("Submitted"):raise RuntimeError("SINGLE_ATTEMPT_ALREADY_RESERVED_NO_RETRY")
        original=read_json(os.path.join(sd,"checkpoint.json"))
        recovery=read_json(os.path.join(sd,"recovery_checkpoint.json"))
        memory=read_json(os.path.join(sd,"mesh_memory_checkpoint.json"))
        if len(original["Jobs"])!=2 or len(recovery["Attempts"])!=3 or memory["Prior_Submissions"]!=3 or len(memory["Attempts"])!=1:
            raise RuntimeError("SUBMISSION_LEDGER_REQUIRES_RECONCILIATION")
        if memory["Status"]!="STOPPED":raise RuntimeError("PRIOR_CONTROLLER_NOT_STOPPED")
        frozen=os.path.join(sd,"frozen_candidates.json")
        if hashlib.sha256(open(frozen,"rb").read()).hexdigest()!=original["Frozen_SHA256"]:
            raise RuntimeError("FROZEN_CANDIDATES_CHANGED")
        candidate=next(c for c in read_json(frozen) if c["Candidate_ID"]=="TD_R_466597d04ebc")
        policy=dict(config.MESH_MEMORY_EXPERIMENT_RESOURCES)
        old_attempt=memory["Attempts"][0]
        inp=os.path.join(old_attempt["Run_Directory"],"freq9_R_466597d04ebc.inp")
        policy["expected_inp_sha256"]=hashlib.sha256(open(inp,"rb").read()).hexdigest()
        if policy["expected_inp_sha256"]!=old_attempt["INP_SHA256"]:raise RuntimeError("PRIOR_INPUT_HASH_CHANGED")
        runroot=ensure_directory(os.path.join(root,"single_memory6gb_attempt_05"))
        localdir=ensure_directory(os.path.join(runroot,"_state"))
        runpath=os.path.join(runroot,candidate["Candidate_ID"])
        if os.path.exists(runpath):raise RuntimeError("FRESH_RUN_PATH_ALREADY_EXISTS")
        state.update(Prior_Submissions=4,Remaining_Budget=8,Resource_Policy=policy,
                     Run_Directory=runpath,Candidate=candidate,Preflight=_windows_memory_snapshot())
        state["Preflight"]["E_Free_GB"]=shutil.disk_usage(root).free/1024**3
        save_json(cp,state)
        try:state["Preflight"]=_recovery_resource_check(policy)
        except RuntimeError as exc:
            state.update(Status="PRECONDITION_BLOCKED",Error=str(exc));save_json(cp,state)
            return state
        state.update(Status="RESERVED_SINGLE_SUBMISSION",Submitted=True,Remaining_Budget=7)
        save_json(cp,state)
        local={"Jobs":{},"Validation":{},"Freed_Bytes":0}
        result=validate_one(candidate,local,localdir,{"run_root":runroot,"output_dir":localdir,
            "output_name":"validation.csv","policy":config.TARGET_DRIVEN_POLICY,
            "resource_policy":policy,"preserve_large_artifacts":True})
        req=local["Jobs"][candidate["Candidate_ID"]]["Request"]
        raw=read_json(req["Result_Path"]) or {}
        telemetry=os.path.join(localdir,candidate["Candidate_ID"]+"_resource_monitor.jsonl")
        samples=[json.loads(line) for line in open(telemetry,encoding="utf-8")] if os.path.isfile(telemetry) else []
        safe=bool(samples) and all(s["Available_RAM_GB"]>=2 and s["Commit_Headroom_GB"]>=3 for s in samples)
        state.update(Result=result,Raw_Result_Path=req["Result_Path"],Resource_Monitor=telemetry,
            Lowest_Available_RAM_GB=min((s["Available_RAM_GB"] for s in samples),default=None),
            Lowest_Commit_Headroom_GB=min((s["Commit_Headroom_GB"] for s in samples),default=None))
        state["Status"]=("SOLVER_SUCCESS_RESOURCE_SAFE" if safe else "SOLVER_SUCCESS_RESOURCE_UNSAFE") if raw.get("Status")=="SUCCESS" else "SOLVER_FAILED"
        # Keep the fresh ODB until the requested complete evidence review is done.
    except Exception as exc:
        state.update(Status="STOPPED_WITH_EVIDENCE",Error=str(exc))
    finally:
        save_json(cp,state);lock.close()
        print(json.dumps(state,ensure_ascii=False),flush=True)
    return state


def run_mesh_memory_experiment():
    """One explicitly authorized original-mesh campaign; no candidate generation.

    The prior three submissions remain charged. A failure stops the campaign.
    The unchanged mesh avoids claiming unproven coarse-grid accuracy.
    """
    import msvcrt
    from paths import TARGET_DRIVEN_RUNS_DIR as root, TARGET_DRIVEN_RESULTS_DIR as out
    sd=os.path.join(root,"_state")
    cp=os.path.join(sd,"mesh_memory_checkpoint.json")
    lock=open(os.path.join(sd,"controller.lock"),"a+b")
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=read_json(cp)
    try:
        prior=read_json(os.path.join(sd,"recovery_checkpoint.json"))
        frozen=os.path.join(sd,"frozen_candidates.json")
        original=read_json(os.path.join(sd,"checkpoint.json"))
        if hashlib.sha256(open(frozen,"rb").read()).hexdigest()!=original["Frozen_SHA256"]:
            raise RuntimeError("FROZEN_CANDIDATES_CHANGED")
        if state is None:
            if len(prior["Attempts"])!=3:raise RuntimeError("PRIOR_BUDGET_NOT_THREE")
            state={"Attempts":[],"Status":"PREPARED","Prior_Submissions":3,
                   "Candidates":read_json(frozen),"Resource_Policy":config.MESH_MEMORY_EXPERIMENT_RESOURCES}
        if state.get("Stopped_On_Failure"):raise RuntimeError("PREVIOUS_EXPERIMENT_FAILED_NO_RESUBMISSION")
        if any(a["Status"]=="RESERVED" for a in state["Attempts"]):
            raise RuntimeError("UNRESOLVED_ATTEMPT_REQUIRES_RECONCILIATION")
        for c in [c for c in state["Candidates"] if c["Trial_Class"]=="FAIR_9"]:
            if any(a["Candidate_ID"]==c["Candidate_ID"] for a in state["Attempts"]):continue
            if 3+len(state["Attempts"])>=12:break
            snapshot=_recovery_resource_check(state["Resource_Policy"])
            number=4+len(state["Attempts"])
            runroot=ensure_directory(os.path.join(root,"memory4gb_attempt_%02d"%number))
            localdir=ensure_directory(os.path.join(runroot,"_state"))
            attempt={"Number":number,"Candidate_ID":c["Candidate_ID"],"Status":"RESERVED",
                     "Resource_Snapshot":snapshot,"Mesh_Version":"ORIGINAL_V2_3P5_3P0_EDGE4"}
            state["Attempts"].append(attempt);state["Status"]="RUNNING";save_json(cp,state)
            local={"Jobs":{},"Validation":{},"Freed_Bytes":0}
            result=validate_one(c,local,localdir,{"run_root":runroot,"output_dir":localdir,
                "output_name":"validation.csv","policy":config.TARGET_DRIVEN_POLICY,
                "resource_policy":state["Resource_Policy"]})
            req=local["Jobs"][c["Candidate_ID"]]["Request"]
            inp=os.path.join(req["Run_Directory"],req["Job_Name"]+".inp")
            attempt.update(Status=result["Status"],Result=result,Run_Directory=req["Run_Directory"],
                           Result_Path=req["Result_Path"],Freed_Bytes=local["Freed_Bytes"],
                           INP_SHA256=hashlib.sha256(open(inp,"rb").read()).hexdigest() if os.path.isfile(inp) else None)
            save_json(cp,state)
            if result["Status"]!="SUCCESS":
                state["Stopped_On_Failure"]=True
                raise RuntimeError("SOLVER_FAILURE_STOP:"+result.get("Failure_Reason",""))
            if result.get("Frequency_Upper_Limit_Hz",0)<c["Target_Frequency_Hz"]:
                state["Stopped_On_Failure"]=True;raise RuntimeError("FREQUENCY_COVERAGE_FAILED")
        state["Status"]="COMPLETE"
    except Exception as exc:
        state=state or {"Attempts":[],"Prior_Submissions":3}
        state.update(Status="STOPPED",Error=str(exc))
    finally:
        save_json(cp,state);lock.close()
        lines=["# Mesh and memory optimization — execution evidence", "",
               "STATUS = "+state["Status"],
               "Original V2 mesh unchanged: shell 3.5 mm; solid 3.0 mm; selected edges 4 divisions.",
               "Experiment: 1 CPU, 4096 MB, automatic analysis memory override disabled.",
               "Entry gates: available RAM >=6 GiB, commit headroom >=10 GiB, E free >=10 GiB; no existing Abaqus solver.",
               "No coarse mesh was promoted; no change to 500 Hz eigenfrequency ceiling, geometry, materials, BC or Tie.",
               "|Attempt|Candidate|Status|Nearest Hz|Error %|Elapsed seconds|", "|---|---|---|---|---|---|"]
        for a in state["Attempts"]:
            v=a.get("Result",{})
            lines.append("|%s|%s|%s|%s|%s|%s|"%(a["Number"],a["Candidate_ID"],a["Status"],v.get("Abaqus_Frequency_Hz","NA"),v.get("Frequency_Error_Percent","NA"),v.get("FE_Elapsed_Seconds","NA")))
        lines += ["", "Total submissions including prior failures: %d/12"%(3+len(state["Attempts"])),
                  "Remaining budget: %d"%(9-len(state["Attempts"])),
                  "Error: "+str(state.get("Error","NONE")),
                  "Detailed immutable per-attempt requests/results and original-mesh INPs are in E:/ABH_Damping_Frequency_Design_frequency9d_runs/target_driven/memory4gb_attempt_*/."]
        with open(os.path.join(out,"MESH_MEMORY_OPTIMIZATION_REPORT.md"),"w",encoding="utf-8") as h:h.write("\n".join(lines)+"\n")
    print(json.dumps({"Status":state["Status"],"New_Submissions":len(state["Attempts"]),"Error":state.get("Error")}),flush=True)
    return state


# Target-driven experiment: isolated from the production retrieval/model cache.
class TargetLocalModel:
    """Training-anchor corrected log GP for one supported ordered-mode branch.

    Anchor response is an existing training observation, never a new FE outcome.
    Ordered-mode identity is not a physical MAC correspondence.
    """
    def __init__(self, data, anchor, mode, allowed_ids=None):
        policy = config.TARGET_DRIVEN_POLICY
        allowed = set(allowed_ids or data["ids"])
        records = [r for r in data["raw_modes"]
                   if r["Sample_ID"] in allowed and r["Mode_Index"] == mode]
        self.anchor = anchor
        self.mode = mode
        self.ax = np.array([anchor[k] for k in FEATURES])
        self.af = float(anchor["Mode_Frequency_Hz"])
        distances = {}
        for r in records:
            key = tuple(r[k] for k in GEOMETRY)
            distances[key] = float(np.linalg.norm((np.array(key)-self.ax[:9])/data["width"][:9]))
        if len(distances) < policy["minimum_local_groups"]:
            raise ValueError("INSUFFICIENT_LOCAL_GEOMETRIES")
        selected = set(sorted(distances, key=distances.get)[:policy["local_groups"]])
        records = [r for r in records if tuple(r[k] for k in GEOMETRY) in selected]
        x = np.array([[r[k] for k in FEATURES] for r in records])
        y = np.log([r["Mode_Frequency_Hz"] for r in records])[:, None]
        self.gp = ForwardModel("GP").fit(x, y)
        self.anchor_log = float(self.gp.predict_log(self.ax)[0][0, 0])
        self.group_count = len(selected)
        self.rows = len(records)

    def predict(self, x):
        mu, std = self.gp.predict_log(np.atleast_2d(x))
        f = np.exp(mu[:, 0]-self.anchor_log+np.log(self.af))
        return f, f*np.sqrt(np.expm1(std[:, 0]**2))


def _target_anchors(data, target, allowed_ids=None):
    allowed = set(allowed_ids or data["ids"])
    ref = data["ref"]
    # All recorded modes, not only modes 1..8; fixed-D02 anchors avoid an
    # unobserved damping transfer at the anchor itself.
    rows = [r for r in data["raw_modes"] if r["Sample_ID"] in allowed
            and abs(r["rx"]-ref[0]) < 1e-12 and abs(r["ry"]-ref[1]) < 1e-12]
    rows.sort(key=lambda r: abs(r["Mode_Frequency_Hz"]-target)/target)
    chosen = []
    for r in rows:
        if any(q["Sample_ID"] == r["Sample_ID"] for q in chosen):
            continue
        chosen.append(r)
        if len(chosen) == 6:
            break
    return chosen


def _target_offline_check(data):
    """Three geometry folds, training-only anchors; diagnostic, not tuning."""
    rows = []
    for fold, (train, test) in enumerate(GroupKFold(3).split(data["x"], groups=data["groups"]), 1):
        assert not set(data["groups"][train]) & set(data["groups"][test])
        baseline = SpectrumModel(MODELS[1], data["ref"]).fit(
            data["x"][train], data["y"][train], data["groups"][train])
        allowed = [data["ids"][i] for i in train]
        for target in config.TARGET_DRIVEN_POLICY["targets"]:
            anchors = _target_anchors(data, target, allowed)
            local = []
            for a in anchors:
                try:
                    local.append(TargetLocalModel(data, a, a["Mode_Index"], allowed))
                except ValueError:
                    pass
            for i in test:
                if not data["fixed"][i]:
                    continue
                x = data["x"][i]; sid = data["ids"][i]
                actual = {r["Mode_Index"]:r["Mode_Frequency_Hz"] for r in data["raw_modes"] if r["Sample_ID"] == sid}
                amean, _ = baseline.predict(x)
                aj = int(np.argmin(abs(amean[0]-target)))
                nearest = min(anchors, key=lambda a: np.linalg.norm((x[:9]-np.array([a[k] for k in GEOMETRY]))/data["width"][:9]))
                predictions = [("R", nearest["Mode_Index"], nearest["Mode_Frequency_Hz"]), ("A", aj+1, amean[0,aj])]
                if local:
                    options = [(abs(m.predict(x)[0][0]-target), m.mode, m.predict(x)[0][0]) for m in local]
                    _, mode, frequency = min(options)
                    predictions.append(("T", mode, frequency))
                for method, mode, frequency in predictions:
                    rows.append({"Fold":fold,"Target":target,"Method":method,"Sample_ID":sid,
                        "Predicted_Mode":mode,"Predicted_Frequency":float(frequency),
                        "Same_Mode_Error_Percent":abs(frequency-actual[mode])/actual[mode]*100 if mode in actual else None,
                        "Actual_Nearest_Target_Error_Percent":min(abs(f-target)/target*100 for f in actual.values()),
                        "Protocol":"HELD_OUT_GEOMETRY_DIAGNOSTIC_NOT_CANDIDATE_HIT_RATE"})
    return rows


def _target_generate(data, baseline, target, method, historical_x):
    import time
    started = time.perf_counter(); policy = config.TARGET_DRIVEN_POLICY
    low, width = data["lower"][:9], data["width"][:9]
    centers = np.unique((data["x"][:,:9]-low)/width, axis=0)
    known = np.vstack((centers, (historical_x-low)/width))
    anchors = _target_anchors(data, target)
    rng = np.random.default_rng(policy["seed"]+int(target*100))
    models = {}
    if method == "T":
        for i, a in enumerate(anchors):
            try:
                models[i] = TargetLocalModel(data, a, a["Mode_Index"])
            except ValueError:
                pass
    def parameters(z):
        return dict(zip(FEATURES, np.r_[low+width*z,data["ref"]]))
    def constraint(z):
        p = parameters(z)
        return np.array([p["Rou_2"]-abs(p["Rou_3"]*math.cos(p["theta"]))-1e-4,
            p["Rou_4"]-abs(p["Rou_3"]*math.sin(p["theta"]))-1e-4,
            p["platform_height"]-p["cutoff_thickness"]-1e-6,
            np.min(np.linalg.norm(known-z,axis=1))-policy["min_geometry_distance"]])
    pool = []
    for start_id in range(policy["starts"]):
        ai = start_id % len(anchors); a = anchors[ai]
        center = (np.array([a[k] for k in GEOMETRY])-low)/width
        lower, upper = np.maximum(0,center-policy["local_halfwidth"]), np.minimum(1,center+policy["local_halfwidth"])
        direction = rng.normal(size=9); direction /= np.linalg.norm(direction)
        start = np.clip(center+direction*(.09+.04*rng.random()),lower,upper)
        mode = int(a["Mode_Index"])
        if method == "T" and ai not in models:
            continue
        if method == "A":
            mu, _ = baseline.predict(np.r_[low+width*center,data["ref"]])
            mode = int(np.argmin(abs(mu[0]-target)))+1
        def predict(z):
            x = np.r_[low+width*z,data["ref"]]
            if method == "A":
                mu, std = baseline.predict(x)
                return float(mu[0,mode-1]),float(std[0,mode-1])
            if method == "T":
                mu, std = models[ai].predict(x)
                return float(mu[0]),float(std[0])
            return float(a["Mode_Frequency_Hz"]), None
        if method == "R":
            z = start
        else:
            def objective(z):
                f, std = predict(z)
                return ((f-target)/target)**2+2e-4*(std/target)**2+2e-5*np.sum((z-center)**2)
            opt = minimize(objective,start,method="SLSQP",bounds=list(zip(lower,upper)),
                constraints=[{"type":"ineq","fun":constraint}],options={"maxiter":65,"ftol":1e-10})
            z = opt.x
        p = parameters(z); valid, reason, derived = validate_general(p)
        distances = np.linalg.norm(known-z,axis=1); distance = float(distances.min())
        if not valid or np.any(constraint(z)<-1e-7) or not .08-1e-6<=distance<=policy["trust_radius"]:
            continue
        if np.sum(abs(z-known[int(distances.argmin())])>=.02)<2:
            continue
        f, std = predict(z); error = abs(f-target)/target*100
        if error > policy["match_percent"]:
            continue
        # One common ranking independent of uncalibrated method-specific STD.
        score = error+2*distance
        sid = "TD_"+method+"_"+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]
        row = dict(p,Candidate_ID=sid,Method=method,Target_Frequency_Hz=target,
            Predicted_Frequency_Hz=f,Predicted_Raw_Mode=mode,Predicted_STD_Hz=std,
            Predicted_Error_Percent=error,Prediction_Kind="CONSTANT_TRAINING_NEIGHBOR" if method=="R" else "SURROGATE",
            Seed_Sample_ID=a["Sample_ID"],Seed_Mode=a["Mode_Index"],Seed_Frequency_Hz=a["Mode_Frequency_Hz"],
            Nearest_Database_Geometry_Distance=float(np.min(np.linalg.norm(centers-z,axis=1))),
            Nearest_All_Previous_FE_Geometry_Distance=distance,
            Local_Training_Geometries=models[ai].group_count if method=="T" else "",
            Geometry_Valid=True,Mesh_Valid="NOT_TESTED",New_Geometry=True,
            Selection_Score=score,Physical_Mode_Identity="UNCONFIRMED",FE_Selected=False,
            Trial_Class="UNVALIDATED_POOL",Footprint_Area=derived["Footprint_Area"])
        pool.append(row)
    selected = []
    for row in sorted(pool,key=lambda r:r["Selection_Score"]):
        z = (np.array([row[k] for k in GEOMETRY])-low)/width
        if all(np.linalg.norm(z-(np.array([r[k] for k in GEOMETRY])-low)/width)>=policy["min_pair_distance"] for r in selected):
            row["Priority"] = len(selected)+1; selected.append(row)
        if len(selected)>=policy["candidate_count"]:
            break
    elapsed = time.perf_counter()-started
    for row in selected:
        row["Generation_Seconds"] = elapsed
    return selected


def _target_report(state, output):
    """Only actual FE results count as successes; planned/unrun rows never do."""
    candidates = state.get("Candidates",[])
    validations = list(state.get("Validation",{}).values())
    by_id = {c["Candidate_ID"]:c for c in candidates}
    enriched = []
    for row in validations:
        c = by_id[row["Candidate_ID"]]
        enriched.append({**c, **row, "FE_Status":row["Status"],
            "Algebraic_Geometry_Valid":c["Geometry_Valid"],
            "Abaqus_Geometry_Valid":row.get("Geometry_Valid") if row["Status"]=="SUCCESS" else None,
            "Geometry_Valid":row.get("Geometry_Valid") if row["Status"]=="SUCCESS" else None,
            "Actual_Frequency_Error_Percent":row.get("Frequency_Error_Percent")})
    write_csv(os.path.join(output,"target_driven_abaqus_validation.csv"),enriched)
    efficiency = []
    for trial in ("FAIR_9","EXPLORATION"):
        for target in list(config.TARGET_DRIVEN_POLICY["targets"])+["ALL"]:
            for method in ("R","A","T"):
                rows = [r for r in enriched if r["Trial_Class"]==trial and r["Method"]==method
                        and (target=="ALL" or r["Target_Frequency_Hz"]==target)]
                passed = [r for r in rows if r.get("Frequency_Match_PASS") and r["FE_Status"]=="SUCCESS"]
                n = len(rows); success = len(passed)
                from scipy.spatial.distance import pdist
                bounds = read_json(config.GENERAL_DESIGN_BOUNDS_FILE)["bounds"]
                widths = np.array([bounds[k]["upper"]-bounds[k]["lower"] for k in GEOMETRY])
                distances = pdist(np.array([[r[k] for k in GEOMETRY] for r in passed])/widths) if success>1 else []
                efficiency.append({"Trial_Class":trial,"Target_Frequency_Hz":target,"Method":method,
                    "FE_Calls":n,"Solver_Success":sum(r["FE_Status"]=="SUCCESS" for r in rows),
                    "Qualified_New_Geometries":success,"Frequency_Match_Success_Percent":100*success/n if n else None,
                    "FE_Calls_Per_Qualified_Design":n/success if success else None,
                    "Qualified_Min_Pair_Distance":float(min(distances)) if len(distances) else None,
                    "Qualified_Database_Anchor_Clusters":len(set(r["Seed_Sample_ID"] for r in passed)),
                    "FE_Seconds":sum(r["FE_Elapsed_Seconds"] for r in rows) if rows and all("FE_Elapsed_Seconds" in r for r in rows) else None,
                    "Generation_Seconds":sum(max([c["Generation_Seconds"] for c in candidates if c["Method"]==method and c["Target_Frequency_Hz"]==t] or [0]) for t in (config.TARGET_DRIVEN_POLICY["targets"] if target=="ALL" else [target]))})
    write_csv(os.path.join(output,"method_efficiency_comparison.csv"),efficiency)
    lines = ["# Target-driven inverse design", "", "Status: "+state.get("Status","PREPARING"),
        "固定D02；旧79工况/1237模态、旧12点FE、生产GPR/PCA及盲测角色保持。原4组远场补样暂停。",
        "## 方法与预注册", "R：固定D02数据库最近目标模态锚点，归一化几何小扰动，预测为锚点频率（不是新构型FE）。",
        "A：原11D条件GPR，仅前8阶；保持基线模型。",
        "T：用全部模态寻找D02目标邻近锚点；按已有排序阶次收集最近24组几何、至少12组才拟合log-GPR；用训练锚点校正局部偏差。搜索可用第9阶及以上。",
        "T的阶次只是局部排序标识，PHYSICAL_MODE_IDENTITY=UNCONFIRMED；截断缺失的阶次不补零。局部训练只读取旧数据库，不使用新FE或盲测响应。",
        "共同规则：原bounds/V2 validator，距全部旧FE几何>=0.08，可信域<=0.35，至少2维改变2%范围；每目标方法48起点，优化最多65迭代；预测频差<=2%，按预测频差+2×归一化距离排序，候选间距>=0.12。R不使用代理梯度。",
        "首批每目标每方法1点，共9点；每目标T的下一多样化候选预先标记探索，共最多3点，不纳入公平比较。所有候选和预测在FE前冻结。",
        "离线3折按基础几何分组，选择训练锚点、局部邻域和模态不读取测试频率；仅诊断，不用于调参。离线旧几何最近目标误差不能当新构型成功率。",
        "", "## 离线诊断（非FE候选命中率）", "|方法|指定排序频率MAPE %|有效/预期条数|", "|---|---:|---:|"]
    for method in ("R","A","T"):
        values = [r["Same_Mode_Error_Percent"] for r in state.get("Offline",[]) if r["Method"]==method and r["Same_Mode_Error_Percent"] is not None]
        lines.append("|%s|%s|%d/45|"%(method,"%.3f"%np.mean(values) if values else "NA",len(values)))
    lines += ["", "## 首批等FE预算真实结果", "|目标Hz|方法|FE合格/调用|实际最近频率Hz|误差%|实际阶次|", "|---|---|---|---|---|---|"]
    for r in enriched:
        if r["Trial_Class"]=="FAIR_9":
            lines.append("|%s|%s|%s/1|%s|%s|%s|"%(r["Target_Frequency_Hz"],r["Method"],int(bool(r.get("Frequency_Match_PASS"))),r.get("Abaqus_Frequency_Hz","NA"),r.get("Frequency_Error_Percent","NA"),r.get("Abaqus_Raw_Mode","NA")))
    lines += ["", "## 探索结果（不混入公平比较）"]
    for r in enriched:
        if r["Trial_Class"]=="EXPLORATION":
            lines.append("- %s %sHz：%s；实际%s Hz，误差%s%%。"%(r["Candidate_ID"],r["Target_Frequency_Hz"],r["FE_Status"],r.get("Abaqus_Frequency_Hz","NA"),r.get("Frequency_Error_Percent","NA")))
    lines += ["", "## 结论与限制"]
    if state.get("Resource_Evidence"):
        lines += ["本轮因资源不足中止，未获得新的有效FE频谱。不能判定R/A/T真实频率匹配率高低，也不能把未运行点写成匹配失败。",
            "资源证据：`"+json.dumps(state["Resource_Evidence"],ensure_ascii=False)+"`",
            "首个R作业输入处理完成后在Lanczos阶段异常退出，日志出现ReadFile from pipe failed；其直接崩溃根因尚未确认。后续A作业期间可用内存已极低，按资源停止条件主动中止本轮所属进程树。",
            "目前只能确认每个目标的三种方法都生成了数据库外代数可行候选，不能确认本轮生成多个真实±2%合格构型。",
            "下一步首先解决计算资源/求解进程退出问题，再按已冻结协议完成验证；当前没有证据要求更换局部算法或额外补样。不会自动恢复，也不改变V2几何、网格或材料。"]
    for e in efficiency:
        if e["Target_Frequency_Hz"]=="ALL" and e["Trial_Class"]=="FAIR_9":
            lines.append("- 方法%s：合格%s/%s，调用/合格=%s。"%(e["Method"],e["Qualified_New_Geometries"],e["FE_Calls"],e["FE_Calls_Per_Qualified_Design"]))
    for target in config.TARGET_DRIVEN_POLICY["targets"]:
        qualified = [r for r in enriched if r["Target_Frequency_Hz"]==target and r.get("Frequency_Match_PASS") and r["FE_Status"]=="SUCCESS"]
        lines.append("- %sHz：本轮真实合格新构型%d个。"%(target,len(qualified)))
    lines += ["9个公平FE样本仅每方法3点，不能据此声称全频段优势或稳定普适设计；探索结果单列。",
        "候选按R/A/T固定顺序执行共同排序和跨方法0.12距离排重；后续方法可能跳过与前一方法过近的首位候选，优先级及参数均在FE前冻结。",
        "离线R有1个所选排序模态在测试样本500 Hz内缺失，误差留空，没有当成零误差。不同方法离线可用分母注明，不能据此给出严格胜负。",
        "设计簇数量使用已知数据库锚点ID作邻域代理，不是物理模态簇或可靠全局聚类；同时报告真实9D归一化两两距离。",
        "同频命中不证明指定排序阶次预测准确，更不证明物理模态追踪成功。",
        "若T不优于R，应优先保留简单扰动和受控局部校验，不扩大复杂模型搜索。若各方法在同一目标失败，结合已保存实际频谱和距离定位，再决定针对性补样，不能用完整谱MAPE替代本轮目标成功率。",
        "新增FE调用：%d/12；释放大型临时文件%.3f GB。"%(len(state.get("Jobs",{})),state.get("Freed_Bytes",0)/1024**3),
        "系统异常："+str(state.get("Error","NONE")),
        "原始候选/冻结协议、完整频谱JSON和checkpoint保存在本轮E盘独立_state目录；不加入原训练库。",
        "输出：target_driven_candidates.csv；target_driven_abaqus_validation.csv；method_efficiency_comparison.csv。",
        "Host / PSD / RMS / fe-safe / 新DOE：NOT_RUN。"]
    with open(os.path.join(output,"TARGET_DRIVEN_INVERSE_DESIGN_REPORT.md"),"w",encoding="utf-8") as h:
        h.write("\n".join(lines)+"\n")


def run_target_driven(validate=False):
    """Prepare once, then explicit bounded isolated FE; never call old workflow."""
    import msvcrt
    from paths import TARGET_DRIVEN_RESULTS_DIR, TARGET_DRIVEN_RUNS_DIR
    output = ensure_directory(TARGET_DRIVEN_RESULTS_DIR)
    root = ensure_directory(TARGET_DRIVEN_RUNS_DIR); state_dir=ensure_directory(os.path.join(root,"_state"))
    lock=open(os.path.join(state_dir,"controller.lock"),"a+b")
    lock.seek(0);lock.write(b"0");lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    checkpoint=os.path.join(state_dir,"checkpoint.json")
    state=read_json(checkpoint) or {"Jobs":{},"Validation":{},"Freed_Bytes":0,"Status":"PREPARING"}
    try:
        data=load_training_data()
        model_path=os.path.join(RUNS,"_state","model_bundle.pkl")
        protected=[MODAL_DATABASE,model_path,os.path.join(BASE_OUTPUT,"new_abh_abaqus_validation.csv"),
            os.path.join(BASE_OUTPUT,"new_abh_candidates.csv"),os.path.join(os.path.dirname(BASE_OUTPUT),"generalization_diagnosis","NEXT_SAMPLING_PLAN.csv")]
        hashes={p:hashlib.sha256(open(p,"rb").read()).hexdigest() for p in protected}
        if "Source_Hashes" in state and hashes!=state["Source_Hashes"]:
            raise RuntimeError("PROTECTED_SOURCE_CHANGED")
        state["Source_Hashes"]=hashes
        with open(model_path,"rb") as h:bundle=pickle.load(h)
        baseline=bundle["model"]
        if baseline.name!=MODELS[1]:raise RuntimeError("BASELINE_MODEL_NOT_ORIGINAL_11D_GPR")
        frozen=os.path.join(state_dir,"frozen_candidates.json")
        with threadpool_limits(limits=1):
            if os.path.isfile(frozen):
                raw=open(frozen,"rb").read()
                if hashlib.sha256(raw).hexdigest()!=state["Frozen_SHA256"]:raise RuntimeError("FROZEN_CANDIDATES_CHANGED")
                state["Candidates"]=json.loads(raw)
            else:
                state["Offline"]=_target_offline_check(data)
                with open(protected[2],encoding="utf-8-sig") as h:history=list(csv.DictReader(h))
                hx=np.array([[float(r[k]) for k in GEOMETRY] for r in history])
                candidates=[]
                for target in config.TARGET_DRIVEN_POLICY["targets"]:
                    selected_at_target=[]
                    for method in ("R","A","T"):
                        rows=_target_generate(data,baseline,target,method,hx)
                        if not rows:raise RuntimeError("NO_SUPPORTED_CANDIDATE_%s_%s"%(target,method))
                        for row in rows:
                            z=np.array([row[k] for k in GEOMETRY])/data["width"][:9]
                            if all(np.linalg.norm(z-v)>=.12 for v in selected_at_target):
                                row.update(FE_Selected=True,Trial_Class="FAIR_9")
                                selected_at_target.append(z);break
                        else:raise RuntimeError("NO_DISTINCT_FAIR_CANDIDATE")
                        candidates.extend(rows)
                    extras=[r for r in candidates if r["Target_Frequency_Hz"]==target and r["Method"]=="T" and not r["FE_Selected"]]
                    for row in extras:
                        z=np.array([row[k] for k in GEOMETRY])/data["width"][:9]
                        if all(np.linalg.norm(z-v)>=.12 for v in selected_at_target):
                            row.update(FE_Selected=True,Trial_Class="EXPLORATION");break
                # Record distances to all frozen candidates (including cross-method).
                coords=np.array([[r[k] for k in GEOMETRY] for r in candidates])/data["width"][:9]
                for i,r in enumerate(candidates):
                    distances=np.linalg.norm(coords-coords[i],axis=1);distances[i]=np.inf
                    r["Nearest_Other_Frozen_Candidate_Distance"]=float(distances.min())
                save_json(frozen,candidates)
                state.update(Candidates=candidates,Frozen_UTC=datetime.now(timezone.utc).isoformat(),
                    Frozen_SHA256=hashlib.sha256(open(frozen,"rb").read()).hexdigest(),Policy=config.TARGET_DRIVEN_POLICY)
            state["Status"]="FROZEN_ZERO_FE" if not validate else "VALIDATING"
            write_csv(os.path.join(output,"target_driven_candidates.csv"),state["Candidates"])
            save_json(checkpoint,state);_target_report(state,output)
        if validate:
            if state.get("Resume_Blocked"):
                raise RuntimeError("RESOURCE_STOP_REQUIRES_EXPLICIT_RESOLUTION_NO_AUTOMATIC_RESUBMISSION")
            # Existing successful worker evidence plus compile check, without a probe job.
            worker=os.path.join(DATABASE_DIR,"v2_pilot_replay_worker.py")
            compile(open(worker,encoding="utf-8").read(),worker,"exec")
            if not shutil.which("abaqus"):raise RuntimeError("ABAQUS_EXECUTABLE_NOT_FOUND")
            runtime={"run_root":root,"output_dir":output,"output_name":"target_driven_abaqus_validation.csv","policy":config.TARGET_DRIVEN_POLICY}
            todo=[c for trial in ("FAIR_9","EXPLORATION") for c in state["Candidates"] if c["Trial_Class"]==trial]
            if len([c for c in todo if c["Trial_Class"]=="FAIR_9"])!=9:raise RuntimeError("FAIR_NINE_NOT_FROZEN")
            consecutive=0
            for c in todo:
                if c["Candidate_ID"] in state["Validation"]:continue
                _target_memory_guard()
                row=validate_one(c,state,state_dir,runtime)
                if row["Status"]=="INCOMPLETE_PREVIOUS_RUN_NOT_RESUBMITTED":raise RuntimeError(row["Status"])
                consecutive=consecutive+1 if row["Status"]!="SUCCESS" else 0
                save_json(checkpoint,state);_target_report(state,output)
                if consecutive>=3 or any(token in row.get("Failure_Reason","").upper() for token in ("TIE","SECTION","COMPLEX","NEAR_ZERO","INVALID_PRIMARY","MEMORY","DISK","RESOURCE_CHECK")):
                    raise RuntimeError("SYSTEMIC_OR_RESOURCE_FAILURE_STOP:"+row.get("Failure_Reason",""))
            state["Status"]="COMPLETED"
        assert all(hashlib.sha256(open(p,"rb").read()).hexdigest()==h for p,h in hashes.items())
    except Exception as exc:
        import traceback
        state.update(Status="STOPPED_WITH_EVIDENCE",Error=str(exc),Traceback=traceback.format_exc())
    finally:
        save_json(checkpoint,state);_target_report(state,output);lock.close()
    print(json.dumps({"Status":state["Status"],"FE_Calls":len(state["Jobs"]),"Error":state.get("Error"),"Output":output},ensure_ascii=False),flush=True)
    return state


def _target_memory_guard():
    """Read-only Windows resource guard; does not change solver memory settings."""
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[
            (name,ctypes.c_ulonglong) for name in ("total","available","page_total","page_available","virtual_total","virtual_available","extended")]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("MEMORY_STATUS_UNAVAILABLE")
    available=status.available/1024**3
    if available<config.TARGET_DRIVEN_POLICY["minimum_available_memory_gb"]:
        raise RuntimeError("INSUFFICIENT_AVAILABLE_MEMORY_GB:%.3f"%available)
    return available


def record_target_resource_stop(reason, evidence):
    """Record a confirmed stopped process tree without launching/resubmitting FE."""
    from paths import TARGET_DRIVEN_RESULTS_DIR, TARGET_DRIVEN_RUNS_DIR
    state_dir=os.path.join(TARGET_DRIVEN_RUNS_DIR,"_state")
    state=read_json(os.path.join(state_dir,"checkpoint.json"))
    candidates={c["Candidate_ID"]:c for c in state["Candidates"]}
    for sid,job in state["Jobs"].items():
        if sid in state["Validation"]:continue
        result=read_json(job["Request"]["Result_Path"])
        if result is None:
            result={"Status":"ABORTED_RESOURCE_LIMIT","Failure_Reason":reason,
                    "Resource_Evidence":evidence,"Abaqus_Frequency_Hz":None}
            save_json(job["Request"]["Result_Path"],result)
        state["Validation"][sid]=summarize_validation(candidates[sid],result)
        job["Status"]=result["Status"]
    state.update(Status="STOPPED_RESOURCE_INSUFFICIENT",Error=reason,
                 Resource_Evidence=evidence,Resume_Blocked=True)
    save_json(os.path.join(state_dir,"checkpoint.json"),state)
    _target_report(state,TARGET_DRIVEN_RESULTS_DIR)
    return state


def _powershell_json(query):
    """Use an explicit wire encoding, independent of Windows GBK locale."""
    prefix="[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); "
    p=subprocess.run(["powershell","-NoProfile","-Command",prefix+query],
                     capture_output=True,encoding="utf-8",errors="strict",timeout=30)
    if p.returncode:raise RuntimeError("POWERSHELL_TELEMETRY_FAILED:"+p.stderr)
    return json.loads(p.stdout.lstrip('\ufeff')) if p.stdout.strip() else []


def _owned_process_snapshot(root_pid):
    """Read-only telemetry; no process selection based merely on an exe name."""
    query="Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,WorkingSetSize,PageFileUsage,CommandLine | ConvertTo-Json -Compress"
    rows=_powershell_json(query); rows=rows if isinstance(rows,list) else [rows]
    owned={root_pid}
    for _ in range(len(rows)):
        children={r["ProcessId"] for r in rows if r["ParentProcessId"] in owned}
        if children<=owned:break
        owned.update(children)
    return [r for r in rows if r["ProcessId"] in owned]


def _windows_memory_snapshot():
    """Available physical and commit capacity; never changes system settings."""
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[
            (n,ctypes.c_ulonglong) for n in ("total","available","commit_limit","commit_free","vtotal","vfree","extended")]
    m=MemoryStatus();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise RuntimeError("MEMORY_SNAPSHOT_FAILED")
    return {"UTC":datetime.now(timezone.utc).isoformat(),"Available_RAM_GB":m.available/1024**3,
            "Commit_Headroom_GB":m.commit_free/1024**3}


def _resource_emergency_count(snapshot, policy, previous_count=0):
    """Return a sustained, genuine resource-emergency count.

    Low *available* RAM alone is normal for a successful Abaqus `memory=90%`
    job.  Therefore warnings never kill a job.  A stop is possible only when
    both physical RAM and commit headroom are critically low for several
    consecutive samples.
    """
    warning_ram=policy.get("warning_available_memory_gb",policy.get("abort_available_memory_gb",2.0))
    warning_commit=policy.get("warning_commit_headroom_gb",policy.get("abort_commit_headroom_gb",3.0))
    snapshot["Resource_Warning"]=(snapshot["Available_RAM_GB"]<warning_ram or
                                  snapshot["Commit_Headroom_GB"]<warning_commit)
    critical=(snapshot["Available_RAM_GB"]<policy.get("emergency_available_memory_gb",0.5) and
              snapshot["Commit_Headroom_GB"]<policy.get("emergency_commit_headroom_gb",1.0))
    return previous_count+1 if critical else 0


def _recovery_resource_check(resource_policy=None):
    """Pre-submit resource snapshot, with no changes to Windows configuration."""
    from paths import TARGET_DRIVEN_RUNS_DIR
    r=resource_policy or config.TARGET_RECOVERY_RESOURCES
    snapshot=_windows_memory_snapshot()
    snapshot["E_Free_GB"]=shutil.disk_usage(TARGET_DRIVEN_RUNS_DIR).free/1024**3
    query="Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(standard|explicit|pre|ABQcaeK|SMAPython)\\.exe$'} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    snapshot["Solver_Processes"]=_powershell_json(query)
    if snapshot["Solver_Processes"]:raise RuntimeError("CONFLICTING_ABAQUS_PROCESS_NO_SUBMISSION")
    if snapshot["Available_RAM_GB"]<r["minimum_available_memory_gb"]:raise RuntimeError("RAM_BELOW_POLICY_THRESHOLD:"+str(snapshot))
    if snapshot["Commit_Headroom_GB"]<r["minimum_commit_headroom_gb"]:raise RuntimeError("INSUFFICIENT_COMMIT_HEADROOM:"+str(snapshot))
    if snapshot["E_Free_GB"]<r["minimum_free_gb"]:raise RuntimeError("DISK_BELOW_10GB")
    return snapshot


def _recovery_report(state, output):
    """Attempt costs include failures; algorithm success uses valid spectra only."""
    from scipy.spatial.distance import pdist
    by_id={r["Candidate_ID"]:r for r in state["Candidates"]}
    rows=[]
    for a in state["Attempts"]:
        c=by_id[a["Candidate_ID"]];v=a.get("Result",{})
        rows.append({**c,**v,"Attempt":a["Number"],"Attempt_Status":a["Status"],
            "Algebraic_Geometry_Valid":c["Geometry_Valid"],
            "Geometry_Valid":v.get("Geometry_Valid") if v.get("Status")=="SUCCESS" else None,
            "Abaqus_Geometry_Valid":v.get("Geometry_Valid") if v.get("Status")=="SUCCESS" else None,
            "Run_Directory":a.get("Run_Directory"),"Result_Path":a.get("Result_Path"),
            "Valid_FE_Spectrum":v.get("Status")=="SUCCESS",
            "Algorithm_Frequency_Match":v.get("Frequency_Match_PASS") if v.get("Status")=="SUCCESS" else "NOT_EVALUABLE"})
    write_csv(os.path.join(output,"target_driven_abaqus_validation.csv"),rows)
    bounds=read_json(config.GENERAL_DESIGN_BOUNDS_FILE)["bounds"]
    widths=np.array([bounds[k]["upper"]-bounds[k]["lower"] for k in GEOMETRY])
    metrics=[]
    for trial in ("FAIR_9","EXPLORATION"):
        for target in list(config.TARGET_DRIVEN_POLICY["targets"])+["ALL"]:
            for method in ("R","A","T"):
                a=[r for r in rows if r["Trial_Class"]==trial and r["Method"]==method and (target=="ALL" or r["Target_Frequency_Hz"]==target)]
                valid=[r for r in a if r["Valid_FE_Spectrum"]]
                good=[r for r in valid if r.get("Frequency_Match_PASS")]
                unique={r["Candidate_ID"]:r for r in good}
                dist=pdist(np.array([[r[k] for k in GEOMETRY] for r in unique.values()])/widths) if len(unique)>1 else []
                metrics.append({"Trial_Class":trial,"Target_Frequency_Hz":target,"Method":method,
                    "FE_Submissions_Including_Failures":len(a),"Valid_FE_Validations":len(valid),
                    "Frequency_Matches":len(good),"Qualified_New_Geometries":len(unique),
                    "True_Match_Rate_Percent":100*len(good)/len(valid) if valid else None,
                    "Match_Rate_Status":"EVALUABLE_SMALL_SAMPLE" if valid else "NOT_EVALUABLE",
                    "FE_Cost_Per_Qualified_Geometry":len(a)/len(unique) if unique else None,
                    "Qualified_Min_Pair_Distance":float(min(dist)) if len(dist) else None,
                    "Known_Total_FE_Seconds":sum(r.get("FE_Elapsed_Seconds",0) for r in a),
                    "Timing_Complete":all("FE_Elapsed_Seconds" in r for r in a),
                    "Timing_Missing_Attempts":sum("FE_Elapsed_Seconds" not in r for r in a)})
    write_csv(os.path.join(output,"method_efficiency_comparison.csv"),metrics)
    lines=["# Target-driven inverse design — resource recovery", "", "STATUS = "+state["Status"],
        "原72候选/12个选定参数与冻结预测保持不变；没有重新训练或搜索。原结果保存在history/pre_resource_recovery。",
        "原2次提交加本轮最多10次，总预算12；失败/主动中止也计入成本。首批9点公平比较优先，仅余1次时按原顺序选择第一个冻结探索点。",
        "## 资源与根因", "Windows事件确认2026-10-11 18:30:20 SMAEqsDirSolverSymmetric.exe在ucrtbase.dll中以0xc0000409异常退出；随后才出现管道错误。",
        "直接事件是求解器子进程崩溃，底层触发原因仍未确认，不能仅凭管道错误断言OOM。未查询到该时间窗System 2004事件；这不排除资源压力。",
        "恢复策略：单CPU、2048 MB、getMemoryFromAnalysis=OFF、串行；不修改几何、材料、网格、BC或原FrequencyStep。",
        "本次attempt03在同一INP上明确报告2048 MB内存上限不足。该保守限额未能满足实际求解需求，恢复未成功；没有继续提交A/T或重复尝试。",
        "提交门槛：RAM>=4 GiB、提交余量>=8 GiB、E盘>=10 GiB、无旧求解进程。8 GiB为2 GiB求解上限另留CAE/装配余量的保守门槛，不是保证内存绝不会超限。",
        "## 真实FE结果", "|提交|目标Hz|方法|状态|最近频率Hz|阶次|误差%|±2%|", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("|%s|%s|%s|%s|%s|%s|%s|%s|"%(r["Attempt"],r["Target_Frequency_Hz"],r["Method"],r["Attempt_Status"],r.get("Abaqus_Frequency_Hz","NA"),r.get("Abaqus_Raw_Mode","NA"),r.get("Frequency_Error_Percent","NA"),r["Algorithm_Frequency_Match"]))
    lines += ["", "## 公平比较：资源失败不算算法频率失败", "|方法|有效频谱|匹配|真实匹配率%|含失败提交数|提交/合格几何|", "|---|---|---|---|---|---|"]
    for m in metrics:
        if m["Trial_Class"]=="FAIR_9" and m["Target_Frequency_Hz"]=="ALL":
            lines.append("|%s|%s|%s|%s|%s|%s|"%(m["Method"],m["Valid_FE_Validations"],m["Frequency_Matches"],m["True_Match_Rate_Percent"] if m["Valid_FE_Validations"] else "NOT_EVALUABLE",m["FE_Submissions_Including_Failures"],m["FE_Cost_Per_Qualified_Geometry"]))
    lines += ["", "## 各目标合格新构型"]
    for t in config.TARGET_DRIVEN_POLICY["targets"]:
        good={r["Candidate_ID"] for r in rows if r["Target_Frequency_Hz"]==t and r["Valid_FE_Spectrum"] and r.get("Frequency_Match_PASS")}
        lines.append("- %s Hz：%d组（探索与公平试验在CSV中分开）。"%(t,len(good)))
    lines += ["", "## 限制与下一步",
        "±2%为实际全部返回模态的最近目标匹配；指定排序误差单独列出；PHYSICAL_MODE_IDENTITY=UNCONFIRMED。",
        "只有全部9个公平候选均获得有效频谱后才能完整比较本次实验；每方法仅3点，仍不能推断全频段稳定性或普适优势。",
        "没有有效FE时，R/A/T胜负及每个合格构型成本均不能评价；先解决求解稳定性，而不是基于资源失败改变算法或补样。",
        "异常："+str(state.get("Error","NONE")),"累计提交数：%d/12。"%len(state["Attempts"]),
        "下一步需先释放更多RAM并重新评估更高的单作业内存上限，或在更大内存机器验证。仅当用户重新授权恢复且资源达标后，才能使用剩余预算；本轮不修改页面文件、不更新系统、不更换物理模型。",
        "旧79数据库/12点FE、PCA/T模型、4盲测及16点计划保护哈希未变。原4点远场补样未执行。",
        "Host耦合、PSD/RMS、fe-safe、新DOE：NOT_RUN。"]
    with open(os.path.join(output,"TARGET_DRIVEN_INVERSE_DESIGN_REPORT.md"),"w",encoding="utf-8") as h:h.write("\n".join(lines)+"\n")


def resume_target_driven_resources():
    """Explicit recovery only: frozen candidates, durable submission ledger, 12 total."""
    import msvcrt
    from paths import TARGET_DRIVEN_RESULTS_DIR as out, TARGET_DRIVEN_RUNS_DIR as root
    sd=os.path.join(root,"_state");cp=os.path.join(sd,"recovery_checkpoint.json")
    lock=open(os.path.join(sd,"controller.lock"),"a+b");lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state=read_json(cp)
    try:
        old=read_json(os.path.join(sd,"checkpoint.json"))
        frozen=os.path.join(sd,"frozen_candidates.json")
        if hashlib.sha256(open(frozen,"rb").read()).hexdigest()!=old["Frozen_SHA256"]:raise RuntimeError("FROZEN_HASH_CHANGED")
        if state is None:
            archive=ensure_directory(os.path.join(out,"history","pre_resource_recovery"))
            for name in ("TARGET_DRIVEN_INVERSE_DESIGN_REPORT.md","target_driven_abaqus_validation.csv","method_efficiency_comparison.csv"):
                dest=os.path.join(archive,name)
                if not os.path.exists(dest):shutil.copy2(os.path.join(out,name),dest)
            state={"Status":"RECOVERY_PREPARED","Candidates":read_json(frozen),"Source_Hashes":old["Source_Hashes"],"Attempts":[]}
            for sid,job in old["Jobs"].items():
                req=job["Request"]
                state["Attempts"].append({"Number":len(state["Attempts"])+1,"Candidate_ID":sid,"Status":job["Status"],
                    "Run_Directory":req["Run_Directory"],"Result_Path":req["Result_Path"],"Result":old["Validation"].get(sid,{})})
            if len(state["Attempts"])!=2:raise RuntimeError("PREVIOUS_SUBMISSION_COUNT_NOT_TWO")
        if state.get("Stopped_On_Failure"):raise RuntimeError("RECOVERY_ALREADY_FAILED_NO_AUTOMATIC_RETRY")
        assert all(hashlib.sha256(open(p,"rb").read()).hexdigest()==v for p,v in state["Source_Hashes"].items())
        todo=[c for c in state["Candidates"] if c["Trial_Class"]=="FAIR_9"]
        todo += [c for c in state["Candidates"] if c["Trial_Class"]=="EXPLORATION"][:1]
        for c in todo:
            previous=[a for a in state["Attempts"] if a["Candidate_ID"]==c["Candidate_ID"] and a["Number"]>2]
            if previous:
                if previous[-1]["Status"]=="SUCCESS":continue
                raise RuntimeError("EXISTING_RECOVERY_ATTEMPT_REQUIRES_RECONCILIATION_NO_RESUBMIT")
            if len(state["Attempts"])>=12:break
            snapshot=_recovery_resource_check()
            number=len(state["Attempts"])+1
            runroot=ensure_directory(os.path.join(root,"recovery_attempt_%02d"%number));attemptdir=ensure_directory(os.path.join(runroot,"_state"))
            attempt={"Number":number,"Candidate_ID":c["Candidate_ID"],"Status":"RESERVED","Resource_Snapshot":snapshot}
            state["Attempts"].append(attempt);state["Status"]="RUNNING_RECOVERY"
            save_json(cp,state)
            local={"Jobs":{},"Validation":{},"Freed_Bytes":0}
            runtime={"run_root":runroot,"output_dir":attemptdir,"output_name":"validation.csv",
                "policy":config.TARGET_DRIVEN_POLICY,"resource_policy":config.TARGET_RECOVERY_RESOURCES}
            result=validate_one(c,local,attemptdir,runtime)
            request=local["Jobs"][c["Candidate_ID"]]["Request"]
            attempt.update(Status=result["Status"],Result=result,Run_Directory=request["Run_Directory"],Result_Path=request["Result_Path"],
                           Worker_Exit_Code=local["Jobs"][c["Candidate_ID"]].get("Worker_Exit_Code"),Freed_Bytes=local["Freed_Bytes"])
            save_json(cp,state);_recovery_report(state,out)
            if result["Status"]!="SUCCESS":
                state["Stopped_On_Failure"]=True
                raise RuntimeError("RECOVERY_SOLVER_FAILED_STOP:"+result.get("Failure_Reason",""))
            if result.get("Frequency_Upper_Limit_Hz",0)<c["Target_Frequency_Hz"]:
                state["Stopped_On_Failure"]=True;raise RuntimeError("INSUFFICIENT_FREQUENCY_COVERAGE")
        state["Status"]="COMPLETED_RECOVERY_BUDGET"
    except Exception as exc:
        import traceback
        state=state or {"Candidates":[],"Attempts":[]}
        state.update(Status="STOPPED_WITH_EVIDENCE",Error=str(exc),Traceback=traceback.format_exc())
    finally:
        save_json(cp,state);_recovery_report(state,out);lock.close()
    print(json.dumps({"Status":state["Status"],"Total_Submissions":len(state["Attempts"]),"Error":state.get("Error")},ensure_ascii=False),flush=True)
    return state
