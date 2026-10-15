"""固定 15 mm 阻尼带的新样本库入口。

使用方法：
1. 只修改下方“用户设置”；
2. ``LIBRARY_ACTION = 'PREPARE'``：只生成并检查候选 CSV，不启动 Abaqus；
3. 人工确认 CSV 后改为 ``'RUN'``：串行计算未完成样本，可安全续跑。

旧 D02 数据库不会被修改；新记录会写入独立的固定物理带宽样本库。
"""
from __future__ import print_function

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np

# The file is deliberately runnable by double-click/``python path\to\file``;
# Abaqus noGUI does not always define ``__file__``, so also accept its argv.
_SCRIPT_PATH = globals().get("__file__") or (sys.argv[0] if sys.argv else os.getcwd())
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_SCRIPT_PATH)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import project_bootstrap
project_bootstrap.activate()

import config
from damping_policy import fixed_band_width_to_rx_ry
from general_geometry_validator import validate_general
from paths import FREQUENCY9D_RUNS_DIR, RESULTS_DIR, ensure_directory


# ============================== 用户设置 ==============================
# PREPARE：只生成并检查候选 CSV，不启动 Abaqus。
# RUN：串行计算既有计划中尚未完成的样本。
LIBRARY_ACTION = "PREPARE"
# SPACE_FILLING：覆盖整个 9D 参数空间；TARGET_LOCAL：围绕已有真实命中目标频率
# 的样本做小幅局部补样。后者用于提高某一目标频率的反向设计可靠性。
SAMPLING_STRATEGY = "TARGET_LOCAL"
TARGET_FREQUENCY_HZ = 129.30
EXISTING_MODAL_DATASET_CSV = r"C:\Users\ipc\Desktop\ABH_Frequency_Dataset_V2_2\output\dataset.csv"
TARGET_ANCHOR_TOLERANCE_PERCENT = 1.0
LOCAL_PERTURBATION_NORMALIZED_RADIUS = 0.025
N_NEW_GEOMETRIES = 6
RANDOM_SEED = 20261013

# 实体阻尼层从边缘单侧向内延伸的宽度；X、Y 独立设置，单位均为 mm。
FIXED_DAMPING_BAND_WIDTH_X_MM = 15.0
FIXED_DAMPING_BAND_WIDTH_Y_MM = 15.0

# 每次最多运行多少个样本。先小批量运行；再次运行会自动跳过已成功样本。
MAX_CASES_PER_RUN = 1
ABAQUS_COMMAND = os.environ.get("ABH_ABAQUS_COMMAND", "abaqus")

# 仅用于人工对照：用外部样本库中同一 Sample_ID 的 9D 参数，在当前正式 V2
# 链中创建一个 .cae。它不会提交 Frequency Job，也不会修改样本库数据。
COMPARISON_SOURCE_SAMPLE_ID = "S00186"
COMPARISON_MODEL_LABEL = "EXTERNAL_MATERIALS"
# =============================================================================

GENERAL_9D = ("Rou_2", "Rou_3", "Rou_4", "theta", "equal_t_length",
              "equal_t_width", "platform_height", "cutoff_thickness",
              "power_law")
def _plan_label():
    if str(SAMPLING_STRATEGY).strip().upper() == "TARGET_LOCAL":
        return "target_%sp%sHz" % (str(float(TARGET_FREQUENCY_HZ)).split(".")[0],
                                   str(float(TARGET_FREQUENCY_HZ)).split(".")[1])
    return "space_filling"


OUTPUT_DIR = os.path.join(RESULTS_DIR, "sample_library_fixed_band", _plan_label())
PLAN_PATH = os.path.join(OUTPUT_DIR, "sample_library_plan.json")
CANDIDATES_CSV = os.path.join(OUTPUT_DIR, "sample_library_candidates.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "sample_library_results.csv")
RUN_ROOT = os.path.join(FREQUENCY9D_RUNS_DIR, "fixed_band_library")
COMPARISON_OUTPUT_DIR = os.path.join(RESULTS_DIR, "comparison_models")


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_abaqus_command():
    """Resolve the local Abaqus launcher even when called from SMApy."""
    if os.path.isfile(ABAQUS_COMMAND):
        return ABAQUS_COMMAND
    drive = os.path.splitdrive(sys.executable)[0]
    fallback = os.path.join(drive + os.sep, "SIMULIA", "Commands", "abaqus.bat")
    if os.path.isfile(fallback):
        return fallback
    discovered = shutil.which(ABAQUS_COMMAND)
    if discovered and os.path.isfile(discovered):
        return discovered
    raise RuntimeError("ABAQUS_COMMAND_NOT_FOUND:set ABH_ABAQUS_COMMAND")


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _bounds():
    return dict((key, tuple(config.PARAMETER_BOUNDS[key])) for key in GENERAL_9D)


def _sample_geometry(rng):
    """Constraint-aware 9D sample; no clipping or repair is performed."""
    bounds = _bounds()
    eps = float(config.GENERAL_GEOMETRY_SAFETY_EPSILON_M)
    p = {}
    p["Rou_3"] = rng.uniform(*bounds["Rou_3"])
    p["theta"] = rng.uniform(*bounds["theta"])
    r2_low = max(bounds["Rou_2"][0], p["Rou_3"] * abs(math.cos(p["theta"])) + eps)
    r4_low = max(bounds["Rou_4"][0], p["Rou_3"] * abs(math.sin(p["theta"])) + eps)
    if r2_low >= bounds["Rou_2"][1] or r4_low >= bounds["Rou_4"][1]:
        return None
    p["Rou_2"] = rng.uniform(r2_low, bounds["Rou_2"][1])
    p["Rou_4"] = rng.uniform(r4_low, bounds["Rou_4"][1])
    p["platform_height"] = rng.uniform(*bounds["platform_height"])
    cutoff_upper = min(bounds["cutoff_thickness"][1], p["platform_height"] - eps)
    if cutoff_upper <= bounds["cutoff_thickness"][0]:
        return None
    p["cutoff_thickness"] = rng.uniform(bounds["cutoff_thickness"][0], cutoff_upper)
    for key in ("equal_t_length", "equal_t_width", "power_law"):
        p[key] = rng.uniform(*bounds[key])
    return p


def _target_anchors():
    """Return only real Abaqus samples whose complete spectrum reaches target."""
    if not os.path.isfile(EXISTING_MODAL_DATASET_CSV):
        raise RuntimeError("EXISTING_MODAL_DATASET_MISSING:%s" % EXISTING_MODAL_DATASET_CSV)
    anchors = []
    bounds = _bounds()
    band_x = float(FIXED_DAMPING_BAND_WIDTH_X_MM) * 1.0e-3
    band_y = float(FIXED_DAMPING_BAND_WIDTH_Y_MM) * 1.0e-3
    with open(EXISTING_MODAL_DATASET_CSV, newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("status") != "MODAL_OK":
                continue
            try:
                p = {key: float(raw[key]) for key in GENERAL_9D}
                spectrum = [float(raw["f_mode_%02d_hz" % i]) for i in range(1, 25)]
            except (KeyError, TypeError, ValueError):
                continue
            # Historical CSV values outside the current V2 legal domain cannot
            # act as a local continuation anchor: clipping them would create a
            # geometrically distant point while falsely labelling it "local".
            if any(p[key] < bounds[key][0] or p[key] > bounds[key][1] for key in GENERAL_9D):
                continue
            try:
                damping = fixed_band_width_to_rx_ry(p, band_x, band_y)
            except ValueError:
                continue
            valid, _, _ = validate_general(dict(p, rx=damping["rx"], ry=damping["ry"]))
            if not valid:
                continue
            mode = min(range(len(spectrum)), key=lambda i: abs(spectrum[i] - float(TARGET_FREQUENCY_HZ)))
            detuning = abs(spectrum[mode] - float(TARGET_FREQUENCY_HZ)) / float(TARGET_FREQUENCY_HZ) * 100.0
            if detuning <= float(TARGET_ANCHOR_TOLERANCE_PERCENT):
                anchors.append({"Parameters": p, "Anchor_Sample_ID": raw.get("sample_id", ""),
                                "Anchor_Mode_Index": mode + 1,
                                "Anchor_Mode_Frequency_Hz": spectrum[mode],
                                "Anchor_Detuning_Percent": detuning})
    if not anchors:
        raise RuntimeError("NO_REAL_TARGET_ANCHOR_WITHIN_TOLERANCE")
    return sorted(anchors, key=lambda item: item["Anchor_Detuning_Percent"])


def _comparison_source_geometry():
    """Load one known external-library geometry without using its FE response."""
    if not os.path.isfile(EXISTING_MODAL_DATASET_CSV):
        raise RuntimeError("EXISTING_MODAL_DATASET_MISSING:%s" % EXISTING_MODAL_DATASET_CSV)
    with open(EXISTING_MODAL_DATASET_CSV, newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("sample_id") != COMPARISON_SOURCE_SAMPLE_ID:
                continue
            try:
                return {key: float(raw[key]) for key in GENERAL_9D}
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("COMPARISON_SOURCE_PARAMETERS_INVALID")
    raise RuntimeError("COMPARISON_SOURCE_SAMPLE_NOT_FOUND:%s" % COMPARISON_SOURCE_SAMPLE_ID)


def _sample_target_local(rng, anchors):
    """New, constraint-valid 9D point close to an actually target-matching anchor.

    This creates an Abaqus experiment plan, not a claimed frequency prediction.
    """
    bounds = _bounds()
    lower = np.array([bounds[key][0] for key in GENERAL_9D])
    upper = np.array([bounds[key][1] for key in GENERAL_9D])
    width = upper - lower
    anchor = anchors[rng.randrange(len(anchors))]
    base = np.array([anchor["Parameters"][key] for key in GENERAL_9D])
    vector = np.clip(base + np.array([rng.uniform(-1.0, 1.0) for _ in GENERAL_9D]) *
                     float(LOCAL_PERTURBATION_NORMALIZED_RADIUS) * width, lower, upper)
    p = dict(zip(GENERAL_9D, vector))
    p["Anchor_Sample_ID"] = anchor["Anchor_Sample_ID"]
    p["Anchor_Mode_Index"] = anchor["Anchor_Mode_Index"]
    p["Anchor_Mode_Frequency_Hz"] = anchor["Anchor_Mode_Frequency_Hz"]
    p["Anchor_Detuning_Percent"] = anchor["Anchor_Detuning_Percent"]
    p["Anchor_Distance_Normalized"] = float(np.linalg.norm((vector - base) / width))
    return p


def _geometry_hash(p):
    text = "|".join("%s=%.14g" % (key, float(p[key])) for key in GENERAL_9D)
    return hashlib.sha1(text.encode("ascii")).hexdigest()[:12]


def prepare():
    """Create only new, prechecked fixed-band candidates; never run Abaqus."""
    ensure_directory(OUTPUT_DIR)
    if os.path.isfile(PLAN_PATH):
        raise RuntimeError("SAMPLE_LIBRARY_PLAN_ALREADY_EXISTS: inspect or run the existing plan; do not silently redraw it")
    if N_NEW_GEOMETRIES < 1:
        raise ValueError("N_NEW_GEOMETRIES_MUST_BE_POSITIVE")
    band_x = float(FIXED_DAMPING_BAND_WIDTH_X_MM) * 1.0e-3
    band_y = float(FIXED_DAMPING_BAND_WIDTH_Y_MM) * 1.0e-3
    rng, rows, tasks, used = random.Random(RANDOM_SEED), [], [], set()
    strategy = str(SAMPLING_STRATEGY).strip().upper()
    if strategy not in ("SPACE_FILLING", "TARGET_LOCAL"):
        raise ValueError("SAMPLING_STRATEGY_MUST_BE_SPACE_FILLING_OR_TARGET_LOCAL")
    anchors = _target_anchors() if strategy == "TARGET_LOCAL" else []
    attempts = 0
    while len(tasks) < int(N_NEW_GEOMETRIES) and attempts < 10000:
        attempts += 1
        metadata = {}
        if strategy == "TARGET_LOCAL":
            proposal = _sample_target_local(rng, anchors)
            geometry = {key: proposal[key] for key in GENERAL_9D}
            metadata = dict((key, proposal[key]) for key in proposal if key not in GENERAL_9D)
            if metadata["Anchor_Distance_Normalized"] > (
                    float(LOCAL_PERTURBATION_NORMALIZED_RADIUS) * math.sqrt(len(GENERAL_9D)) + 1.0e-12):
                raise RuntimeError("TARGET_LOCAL_ANCHOR_DISTANCE_INVALID")
        else:
            geometry = _sample_geometry(rng)
        if geometry is None:
            continue
        token = _geometry_hash(geometry)
        if token in used:
            continue
        try:
            damping = fixed_band_width_to_rx_ry(geometry, band_x, band_y)
        except ValueError:
            continue
        general = dict(geometry, rx=damping["rx"], ry=damping["ry"])
        valid, reason, derived = validate_general(general)
        if not valid:
            continue
        used.add(token)
        index = len(tasks) + 1
        sample_id = "FB15_LIBRARY_%03d" % index
        abh_id = "FB15_ABH_%s" % token
        source = damping["Damping_Configuration"]
        task = {"Sample_ID": sample_id, "ABH_ID": abh_id,
                "Source": "FIXED_PHYSICAL_BAND_LIBRARY",
                "Sampling_Strategy": strategy,
                "Damping_Configuration": source,
                "General_Input": general,
                "Fixed_Damping_Band_Width_X_m": band_x,
                "Fixed_Damping_Band_Width_Y_m": band_y,
                "Thickness_Field_Implementation": config.THICKNESS_FIELD_IMPLEMENTATION,
                "Geometry_Builder_Version": config.GEOMETRY_BUILDER_VERSION}
        tasks.append(task)
        row = {"Sample_ID": sample_id, "ABH_ID": abh_id,
               "Dataset_Source": task["Source"], "Sampling_Strategy": strategy,
               "Damping_Configuration": source,
               "Geometry_Valid_Precheck": True, "Failure_Reason": "",
               "Fixed_Damping_Band_Width_X_mm": FIXED_DAMPING_BAND_WIDTH_X_MM,
               "Fixed_Damping_Band_Width_Y_mm": FIXED_DAMPING_BAND_WIDTH_Y_MM}
        row.update(general); row.update(damping); row.update(derived); row.update(metadata)
        task.update(metadata)
        rows.append(row)
    if len(tasks) != int(N_NEW_GEOMETRIES):
        raise RuntimeError("SAMPLE_LIBRARY_CANDIDATE_GENERATION_INCOMPLETE")
    _write_csv(CANDIDATES_CSV, rows)
    with open(PLAN_PATH, "w") as handle:
        json.dump({"Status": "PREPARED", "Random_Seed": RANDOM_SEED,
                   "Sampling_Strategy": strategy,
                   "Target_Frequency_Hz": (float(TARGET_FREQUENCY_HZ) if strategy == "TARGET_LOCAL" else None),
                   "Target_Anchor_Tolerance_Percent": (float(TARGET_ANCHOR_TOLERANCE_PERCENT)
                                                       if strategy == "TARGET_LOCAL" else None),
                   "Local_Perturbation_Normalized_Radius": (float(LOCAL_PERTURBATION_NORMALIZED_RADIUS)
                                                              if strategy == "TARGET_LOCAL" else None),
                   "Candidate_Count": len(tasks), "Generation_Attempts": attempts,
                   "Damping_Configuration": tasks[0]["Damping_Configuration"],
                   "Fixed_Damping_Band_Width_X_m": band_x,
                   "Fixed_Damping_Band_Width_Y_m": band_y,
                   "Candidates": tasks}, handle, indent=2, sort_keys=True)
    return {"Status": "PREPARED_NO_ABAQUS", "Candidates": len(tasks),
            "Candidate_CSV": CANDIDATES_CSV, "Plan": PLAN_PATH}


def _worker(sample_id):
    """CAE-side execution of exactly one prepared candidate."""
    if not os.path.isfile(PLAN_PATH):
        raise RuntimeError("SAMPLE_LIBRARY_PLAN_MISSING")
    with open(PLAN_PATH, "r") as handle:
        plan = json.load(handle)
    task = [item for item in plan["Candidates"] if item["Sample_ID"] == sample_id]
    if len(task) != 1:
        raise RuntimeError("SAMPLE_LIBRARY_UNKNOWN_SAMPLE_ID")
    import eleven_d_worker as worker
    worker.RUNROOT = ensure_directory(RUN_ROOT)
    worker._run_one(task[0], RESULTS_CSV)


def run():
    """Submit only pending prepared candidates, one Abaqus process at a time."""
    if not os.path.isfile(PLAN_PATH):
        raise RuntimeError("SAMPLE_LIBRARY_PLAN_MISSING_RUN_PREPARE_FIRST")
    with open(PLAN_PATH, "r") as handle:
        plan = json.load(handle)
    completed = set(row.get("Sample_ID") for row in _read_csv(RESULTS_CSV)
                    if row.get("Job_Status") == "SUCCESS")
    pending = [item for item in plan["Candidates"] if item["Sample_ID"] not in completed]
    selected = pending[:int(MAX_CASES_PER_RUN)]
    own_path = os.path.abspath(__file__)
    logs = []
    for task in selected:
        log_path = os.path.join(OUTPUT_DIR, task["Sample_ID"] + "_controller.log")
        command = [_resolve_abaqus_command(), "cae", "noGUI=" + own_path, "--", "--worker", "--sample", task["Sample_ID"]]
        with open(log_path, "w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        logs.append({"Sample_ID": task["Sample_ID"], "Exit_Code": code, "Controller_Log": log_path})
    return {"Status": "RUN_FINISHED", "Submitted": len(selected), "Pending_Before_Run": len(pending),
            "Results_CSV": RESULTS_CSV, "Controller_Logs": logs}


def create_comparison_cae():
    """Ask Abaqus/CAE to build, mesh and save one current-V2 comparison model."""
    ensure_directory(COMPARISON_OUTPUT_DIR)
    cae_path = os.path.join(COMPARISON_OUTPUT_DIR,
                            "CURRENT_V2_%s_FIXED_X15_Y15MM_%s.cae" %
                            (COMPARISON_SOURCE_SAMPLE_ID, COMPARISON_MODEL_LABEL))
    if os.path.isfile(cae_path):
        return {"Status": "CAE_ALREADY_EXISTS_NO_JOB_SUBMITTED", "CAE_Path": cae_path}
    own_path = os.path.abspath(__file__)
    log_path = os.path.join(COMPARISON_OUTPUT_DIR,
                            "CURRENT_V2_%s_%s_BUILD_ONLY.log" %
                            (COMPARISON_SOURCE_SAMPLE_ID, COMPARISON_MODEL_LABEL))
    command = [_resolve_abaqus_command(), "cae", "noGUI=" + own_path, "--", "--create-comparison-cae"]
    with open(log_path, "w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    manifest = os.path.join(COMPARISON_OUTPUT_DIR,
                            "CURRENT_V2_%s_%s_BUILD_ONLY.json" %
                            (COMPARISON_SOURCE_SAMPLE_ID, COMPARISON_MODEL_LABEL))
    if code != 0 or not os.path.isfile(cae_path) or not os.path.isfile(manifest):
        raise RuntimeError("CURRENT_V2_COMPARISON_CAE_BUILD_FAILED: see %s" % log_path)
    return {"Status": "CAE_CREATED_NO_JOB_SUBMITTED", "CAE_Path": cae_path,
            "Manifest": manifest, "Build_Log": log_path}


def _create_comparison_cae_worker():
    """CAE-side half of CREATE_COMPARISON_CAE; it never calls job.submit()."""
    geometry = _comparison_source_geometry()
    band_x = float(FIXED_DAMPING_BAND_WIDTH_X_MM) * 1.0e-3
    band_y = float(FIXED_DAMPING_BAND_WIDTH_Y_MM) * 1.0e-3
    damping = fixed_band_width_to_rx_ry(geometry, band_x, band_y)
    general = dict(geometry, rx=damping["rx"], ry=damping["ry"])
    valid, reason, _ = validate_general(general)
    if not valid:
        raise RuntimeError("COMPARISON_SOURCE_NOT_VALID_IN_CURRENT_V2:%s" % reason)
    from frame_runner import run_frame
    parameters = dict(config.FRAME_FIXED_PARAMETERS)
    parameters.update(geometry)
    parameters["damp_inward_x"] = band_x
    parameters["damp_inward_y"] = band_y
    ensure_directory(COMPARISON_OUTPUT_DIR)
    run_directory = os.path.join(RUN_ROOT, "comparison_" + COMPARISON_SOURCE_SAMPLE_ID + "_" + COMPARISON_MODEL_LABEL)
    if os.path.isdir(run_directory) and os.listdir(run_directory):
        raise RuntimeError("COMPARISON_RUN_DIRECTORY_NOT_EMPTY:%s" % run_directory)
    cae_path = os.path.join(COMPARISON_OUTPUT_DIR,
                            "CURRENT_V2_%s_FIXED_X15_Y15MM_%s.cae" %
                            (COMPARISON_SOURCE_SAMPLE_ID, COMPARISON_MODEL_LABEL))
    result = run_frame(parameters, run_directory, "CURRENT_V2_" + COMPARISON_SOURCE_SAMPLE_ID,
                       split_xy=True, mesh_level="medium", mode_count=24,
                       field_implementation=config.THICKNESS_FIELD_IMPLEMENTATION,
                       dataset_source="CURRENT_V2_COMPARISON_ONLY", submit_job=False,
                       save_cae_path=cae_path)
    result.update({"Source_Sample_ID": COMPARISON_SOURCE_SAMPLE_ID,
                   "Fixed_Damping_Band_Width_X_mm": FIXED_DAMPING_BAND_WIDTH_X_MM,
                   "Fixed_Damping_Band_Width_Y_mm": FIXED_DAMPING_BAND_WIDTH_Y_MM,
                   "Abaqus_Job_Submitted": False})
    manifest = os.path.join(COMPARISON_OUTPUT_DIR,
                            "CURRENT_V2_%s_%s_BUILD_ONLY.json" %
                            (COMPARISON_SOURCE_SAMPLE_ID, COMPARISON_MODEL_LABEL))
    with open(manifest, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def main():
    action = str(LIBRARY_ACTION).strip().upper()
    if action == "PREPARE":
        result = prepare()
    elif action == "RUN":
        result = run()
    elif action == "CREATE_COMPARISON_CAE":
        result = create_comparison_cae()
    else:
        raise ValueError("LIBRARY_ACTION_MUST_BE_PREPARE_RUN_OR_CREATE_COMPARISON_CAE")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    if "--create-comparison-cae" in sys.argv:
        _create_comparison_cae_worker()
    elif "--worker" in sys.argv:
        index = sys.argv.index("--sample")
        _worker(sys.argv[index + 1])
    else:
        main()
