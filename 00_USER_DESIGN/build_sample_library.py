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
import subprocess
import sys

# The file is deliberately runnable by double-click/``python path\to\file``;
# make the project root importable before activating the organized modules.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
N_NEW_GEOMETRIES = 6
RANDOM_SEED = 20261013

# 实体阻尼层从边缘单侧向内延伸的宽度；X、Y 独立设置，单位均为 mm。
FIXED_DAMPING_BAND_WIDTH_X_MM = 15.0
FIXED_DAMPING_BAND_WIDTH_Y_MM = 15.0

# 每次最多运行多少个样本。先小批量运行；再次运行会自动跳过已成功样本。
MAX_CASES_PER_RUN = 1
ABAQUS_COMMAND = os.environ.get("ABH_ABAQUS_COMMAND", "abaqus")
# =============================================================================

GENERAL_9D = ("Rou_2", "Rou_3", "Rou_4", "theta", "equal_t_length",
              "equal_t_width", "platform_height", "cutoff_thickness",
              "power_law")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "sample_library_fixed_band")
PLAN_PATH = os.path.join(OUTPUT_DIR, "sample_library_plan.json")
CANDIDATES_CSV = os.path.join(OUTPUT_DIR, "sample_library_candidates.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "sample_library_results.csv")
RUN_ROOT = os.path.join(FREQUENCY9D_RUNS_DIR, "fixed_band_library")


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
    attempts = 0
    while len(tasks) < int(N_NEW_GEOMETRIES) and attempts < 10000:
        attempts += 1
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
                "Damping_Configuration": source,
                "General_Input": general,
                "Fixed_Damping_Band_Width_X_m": band_x,
                "Fixed_Damping_Band_Width_Y_m": band_y,
                "Thickness_Field_Implementation": config.THICKNESS_FIELD_IMPLEMENTATION,
                "Geometry_Builder_Version": config.GEOMETRY_BUILDER_VERSION}
        tasks.append(task)
        row = {"Sample_ID": sample_id, "ABH_ID": abh_id,
               "Dataset_Source": task["Source"], "Damping_Configuration": source,
               "Geometry_Valid_Precheck": True, "Failure_Reason": "",
               "Fixed_Damping_Band_Width_X_mm": FIXED_DAMPING_BAND_WIDTH_X_MM,
               "Fixed_Damping_Band_Width_Y_mm": FIXED_DAMPING_BAND_WIDTH_Y_MM}
        row.update(general); row.update(damping); row.update(derived)
        rows.append(row)
    if len(tasks) != int(N_NEW_GEOMETRIES):
        raise RuntimeError("SAMPLE_LIBRARY_CANDIDATE_GENERATION_INCOMPLETE")
    _write_csv(CANDIDATES_CSV, rows)
    with open(PLAN_PATH, "w") as handle:
        json.dump({"Status": "PREPARED", "Random_Seed": RANDOM_SEED,
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
        command = [ABAQUS_COMMAND, "cae", "noGUI=" + own_path, "--", "--worker", "--sample", task["Sample_ID"]]
        with open(log_path, "w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        logs.append({"Sample_ID": task["Sample_ID"], "Exit_Code": code, "Controller_Log": log_path})
    return {"Status": "RUN_FINISHED", "Submitted": len(selected), "Pending_Before_Run": len(pending),
            "Results_CSV": RESULTS_CSV, "Controller_Logs": logs}


def main():
    action = str(LIBRARY_ACTION).strip().upper()
    if action == "PREPARE":
        result = prepare()
    elif action == "RUN":
        result = run()
    else:
        raise ValueError("LIBRARY_ACTION_MUST_BE_PREPARE_OR_RUN")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    if "--worker" in sys.argv:
        index = sys.argv.index("--sample")
        _worker(sys.argv[index + 1])
    else:
        main()
