"""Database-only all-mode RABH target-frequency candidate retrieval."""
from __future__ import print_function

import csv
import json
import os
from types import SimpleNamespace

import config
from paths import RESULTS_DIR, RESULTS_INVERSE_DIR, ensure_directory
from damping_mapping import shell_outer_dimensions


MODAL_DATABASE = os.path.join(RESULTS_DIR, "v2_unified_all_modes_effective_mass_long.csv")
FORMAL_BATCH1_DATABASE = os.path.join(RESULTS_DIR, "formal_11d_batch1.csv")
REQUIRED_11D = (
    "Rou_2", "Rou_3", "Rou_4", "cutoff_thickness", "equal_t_length",
    "equal_t_width", "platform_height", "power_law", "rx", "ry", "theta",
)
REGRESSION_HOST_TARGETS_HZ = (34.41, 129.30, 213.63)
FIXED_BAND_LIBRARY = "FIXED_BAND_15MM_V2"
FIXED_BAND_9D = (
    "Rou_2", "Rou_3", "Rou_4", "cutoff_thickness", "equal_t_length",
    "equal_t_width", "platform_height", "power_law", "theta",
)


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError("DATABASE_NUMERIC_FIELD_INVALID:%s" % key)
    return value


def _target_folder_name(value):
    token = ("%.6f" % value).rstrip("0").rstrip(".").replace(".", "p")
    return "target_%sHz" % token


def _optional_number(row, key):
    """Return a numeric field when present; retain unavailable decomposition."""
    try:
        return _number(row, key)
    except ValueError:
        return None


def _fixed_band_shell_dimensions(parameters):
    """Calculate V2 ABH-shell outer size from the actual geometry equations."""
    full = dict(config.FIXED_BASELINE_PARAMETERS)
    full.update(parameters)
    return shell_outer_dimensions(full)


def _shell_size_bounds_m(request):
    """Read optional user-facing shell-envelope limits, expressed in mm."""
    if not bool(getattr(request, "USE_ABH_SHELL_SIZE_FILTER", False)):
        return None
    raw_x = getattr(request, "ABH_SHELL_OVERALL_LENGTH_X_MM_RANGE", None)
    raw_y = getattr(request, "ABH_SHELL_OVERALL_WIDTH_Y_MM_RANGE", None)
    try:
        x_min, x_max = map(float, raw_x)
        y_min, y_max = map(float, raw_y)
    except (TypeError, ValueError):
        raise ValueError("ABH_SHELL_SIZE_RANGE_MUST_HAVE_TWO_NUMBERS_IN_MM")
    # 0 mm is a supported lower bound: it means only an upper envelope limit.
    if not (0.0 <= x_min <= x_max and 0.0 <= y_min <= y_max):
        raise ValueError("ABH_SHELL_SIZE_RANGE_INVALID")
    return {
        "length_x_m": (x_min / 1000.0, x_max / 1000.0),
        "width_y_m": (y_min / 1000.0, y_max / 1000.0),
    }


def _formal_mass_index(path=FORMAL_BATCH1_DATABASE):
    """Read the only available per-instance mass decomposition.

    Batch1 recorded ABH_Shell and DAMP instance masses separately.  The DAMP
    instance contains both the damping and constrained layers, so the existing
    database does not support a trustworthy split between those two layers.
    """
    if not os.path.isfile(path):
        return {}
    index = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            try:
                shell = _number(raw, "ABH_Mass")
                damp_and_constrained = _number(raw, "DAMP_Mass")
                assembly = _number(raw, "Total_Attachment_Mass")
            except ValueError:
                continue
            index[raw.get("Sample_ID", "")] = {
                "ABH_Shell_Mass": shell,
                "Damping_And_Constrained_Layer_Mass": damp_and_constrained,
                "RABH_Assembly_Mass": assembly,
                "Mass_Component_Status": "BATCH1_INSTANCE_MASS_PROPERTIES",
            }
    return index


def _load_valid_modes(database_path=MODAL_DATABASE):
    if not os.path.isfile(database_path):
        raise RuntimeError("FORMAL_V2_MODAL_DATABASE_MISSING:%s" % database_path)
    modes, mass_index = [], _formal_mass_index()
    with open(database_path, "r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("Extraction_Status") != "SUCCESS":
                continue
            try:
                frequency = _number(raw, "Frequency_Hz")
                mode = int(float(raw["Mode_Index"]))
                if frequency <= 0.0 or mode <= 0:
                    continue
                record = {
                    "Sample_ID": raw["Sample_ID"],
                    "ABH_ID": raw["ABH_ID"],
                    "Damping_ID": raw.get("Damping_ID", ""),
                    "Source_Batch": raw.get("Source_Batch", ""),
                    "Mode_Index": mode,
                    "Mode_Frequency_Hz": frequency,
                    "Effective_Modal_Mass_Z": _number(raw, "Effective_Modal_Mass_Z"),
                    "Effective_Mass_Ratio_Z": _number(raw, "Effective_Mass_Ratio_Z"),
                    # In the modal long table this field was copied from the
                    # Batch1 Total_Attachment_Mass field, not from Abaqus's
                    # distinct DAT "TOTAL MASS OF MODEL" line.
                    "RABH_Assembly_Mass": _number(raw, "Total_Mass"),
                    "Abaqus_Total_Model_Mass_from_DAT": _optional_number(raw, "Total_Model_Mass"),
                    "Footprint": _number(raw, "Footprint"),
                }
                decomposition = mass_index.get(record["Sample_ID"])
                if decomposition:
                    record.update(decomposition)
                else:
                    record.update({
                        "ABH_Shell_Mass": None,
                        "Damping_And_Constrained_Layer_Mass": None,
                        "Mass_Component_Status": "NOT_AVAILABLE_IN_UNIFIED_LONG_DATABASE",
                    })
                record["Mass_Definition"] = (
                    "RABH_Assembly_Mass equals Batch1 Total_Attachment_Mass "
                    "(ABH_Shell_Mass + Damping_And_Constrained_Layer_Mass). "
                    "The DAMP instance combines damping and constrained layers."
                )
                for key in REQUIRED_11D:
                    record[key] = _number(raw, key)
                record["Damping_Configuration"] = (
                    "%s; rx=%.12g; ry=%.12g; damp_thickness_m=0.003; "
                    "constrained_layer_thickness_m=0.0001; damp_elastic_pa=1.45e8; "
                    "damp_loss_factor=0.7" %
                    (config.DAMP_GEOMETRY_MODE, record["rx"], record["ry"]))
                modes.append(record)
            except (KeyError, TypeError, ValueError):
                continue
    if not modes:
        raise RuntimeError("NO_VALID_V2_MODAL_DATABASE_RECORDS")
    return modes


def _load_fixed_band_15mm_modes(dataset_path):
    """读取固定 15 mm 实体阻尼带样本库，保留其物理长度语义。"""
    if not dataset_path or not os.path.isfile(dataset_path):
        raise RuntimeError("FIXED_BAND_15MM_DATASET_MISSING:%s" % dataset_path)
    modes = []
    with open(dataset_path, "r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("status") != "MODAL_OK":
                continue
            try:
                sample_id = str(raw["sample_id"]).strip()
                inward_x = _number(raw, "rx")
                inward_y = _number(raw, "ry")
                if not sample_id or inward_x <= 0.0 or inward_y <= 0.0:
                    continue
                parameters = {key: _number(raw, key) for key in FIXED_BAND_9D}
                for index in range(1, 25):
                    frequency = _number(raw, "f_mode_%02d_hz" % index)
                    if frequency <= 0.0:
                        continue
                    record = {
                        "Sample_ID": sample_id,
                        "ABH_ID": sample_id,
                        "Damping_ID": "FIXED_BAND_15MM",
                        "Source_Batch": FIXED_BAND_LIBRARY,
                        "Mode_Index": index,
                        "Mode_Frequency_Hz": frequency,
                        # 此库没有模态有效质量、质量或面积，检索时不得伪造这些量。
                        "Effective_Modal_Mass_Z": None,
                        "Effective_Mass_Ratio_Z": None,
                        "RABH_Assembly_Mass": None,
                        "Abaqus_Total_Model_Mass_from_DAT": None,
                        "ABH_Shell_Mass": None,
                        "Damping_And_Constrained_Layer_Mass": None,
                        "Footprint": None,
                        "Mass_Component_Status": "NOT_PROVIDED_BY_FIXED_BAND_LIBRARY",
                        "Mass_Definition": "NOT_AVAILABLE_IN_SOURCE_DATASET",
                        # 不把物理 15 mm 长度映射为旧 V2 的无量纲 rx/ry。
                        "rx": None,
                        "ry": None,
                        "Damping_Inward_X_m": inward_x,
                        "Damping_Inward_Y_m": inward_y,
                        "Damping_Configuration": (
                            "FIXED_BAND_WIDTH; inward_x_m=%.12g; inward_y_m=%.12g; "
                            "source_rx_ry_are_PHYSICAL_LENGTHS_NOT_RATIOS" % (inward_x, inward_y)
                        ),
                    }
                    record.update(parameters)
                    record.update(_fixed_band_shell_dimensions(parameters))
                    modes.append(record)
            except (KeyError, TypeError, ValueError):
                continue
    if not modes:
        raise RuntimeError("NO_VALID_FIXED_BAND_15MM_MODAL_RECORDS")
    return modes


def _modes_for_request(request, database_path):
    if str(getattr(request, "DATA_LIBRARY", "")).upper() == FIXED_BAND_LIBRARY:
        return _load_fixed_band_15mm_modes(
            getattr(request, "FIXED_BAND_DATASET_CSV", ""))
    return _load_valid_modes(database_path)


def retrieve(request, database_path=MODAL_DATABASE):
    """Return the raw tolerance-qualified all-mode pool in database order."""
    target = float(request.TARGET_FREQUENCY_HZ)
    tolerance = float(request.FREQUENCY_TOLERANCE_PERCENT)
    if target <= 0.0 or tolerance <= 0.0:
        raise ValueError("TARGET_AND_TOLERANCE_MUST_BE_POSITIVE")
    if str(request.TARGET_DIRECTION).upper() != "Z":
        raise ValueError("ONLY_Z_EFFECTIVE_MODAL_MASS_IS_AVAILABLE")
    max_mass = getattr(request, "MAX_MASS", None)
    mass_filter = bool(getattr(request, "USE_MASS_FILTER", False))
    shell_bounds = _shell_size_bounds_m(request)
    if mass_filter and max_mass is None:
        raise ValueError("MAX_MASS_REQUIRED_WHEN_MASS_FILTER_IS_TRUE")

    pool = []
    for row in _modes_for_request(request, database_path):
        detuning = abs(row["Mode_Frequency_Hz"] - target) / target * 100.0
        if detuning > tolerance:
            continue
        if shell_bounds is not None:
            length = row.get("ABH_Shell_Overall_Length_X_m")
            width = row.get("ABH_Shell_Overall_Width_Y_m")
            if length is None or width is None:
                raise ValueError("SHELL_SIZE_FILTER_UNAVAILABLE_FOR_SELECTED_DATA_LIBRARY")
            if not (shell_bounds["length_x_m"][0] <= length <= shell_bounds["length_x_m"][1] and
                    shell_bounds["width_y_m"][0] <= width <= shell_bounds["width_y_m"][1]):
                continue
        if mass_filter and row["RABH_Assembly_Mass"] is None:
            raise ValueError("MASS_FILTER_UNAVAILABLE_FOR_SELECTED_DATA_LIBRARY")
        if mass_filter and row["RABH_Assembly_Mass"] > float(max_mass):
            continue
        item = dict(row)
        item["Target_Frequency_Hz"] = target
        item["Detuning_Percent"] = detuning
        pool.append(item)
    return pool


def rank_candidates(request, pool):
    """Rank a copy of the raw pool according to the current preference."""
    ranked = [dict(item) for item in pool]
    has_meff = bool(ranked) and all(item["Effective_Modal_Mass_Z"] is not None for item in ranked)
    if bool(request.PREFER_HIGH_EFFECTIVE_MODAL_MASS) and has_meff:
        ranked.sort(key=lambda item: (-item["Effective_Modal_Mass_Z"], item["Detuning_Percent"],
                                      item["RABH_Assembly_Mass"], item["Sample_ID"], item["Mode_Index"]))
    else:
        # 固定 15 mm 库未提供质量/有效质量，因此只按频差和 ID 排序。
        ranked.sort(key=lambda item: (item["Detuning_Percent"], item["Sample_ID"], item["Mode_Index"]))
    for index, item in enumerate(ranked, 1):
        item["Rank"] = index
    return ranked


def _write_csv(path, rows, include_rank):
    fields = (
        "Target_Frequency_Hz", "Sample_ID", "ABH_ID", "Damping_ID", "Source_Batch",
        "Mode_Index", "Mode_Frequency_Hz", "Detuning_Percent", "Effective_Modal_Mass_Z",
        "Effective_Mass_Ratio_Z", "ABH_Shell_Mass", "Damping_And_Constrained_Layer_Mass",
        "RABH_Assembly_Mass", "Abaqus_Total_Model_Mass_from_DAT", "Mass_Component_Status",
        "Mass_Definition", "Footprint", "Damping_Inward_X_m", "Damping_Inward_Y_m",
        "ABH_Shell_Overall_Length_X_m", "ABH_Shell_Overall_Width_Y_m", "Damping_Configuration",
    ) + REQUIRED_11D
    if include_rank:
        fields = ("Rank",) + fields
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_design_output(request, pool, ranked):
    library = str(getattr(request, "DATA_LIBRARY", "V2_DATABASE"))
    base_dir = RESULTS_INVERSE_DIR if library != FIXED_BAND_LIBRARY else os.path.join(RESULTS_INVERSE_DIR, "fixed_band_15mm")
    target_dir = ensure_directory(os.path.join(base_dir, _target_folder_name(request.TARGET_FREQUENCY_HZ)))
    pool_path = os.path.join(target_dir, "candidate_pool.csv")
    ranking_path = os.path.join(target_dir, "candidate_rankings.csv")
    summary_path = os.path.join(target_dir, "design_summary.md")
    selected_path = os.path.join(target_dir, "selected_parameters.json")
    _write_csv(pool_path, pool, include_rank=False)
    _write_csv(ranking_path, ranked[:int(request.TOP_N)], include_rank=True)
    selected = ranked[0] if ranked else None
    with open(selected_path, "w", encoding="utf-8") as handle:
        json.dump(selected if selected is not None else {}, handle, indent=2, sort_keys=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Target-frequency database retrieval\n\n")
        handle.write("- Target frequency: %.12g Hz\n" % float(request.TARGET_FREQUENCY_HZ))
        handle.write("- Tolerance: %.12g %%\n" % float(request.FREQUENCY_TOLERANCE_PERCENT))
        handle.write("- Raw tolerance-qualified all-mode candidates: %d\n" % len(pool))
        handle.write("- Data library: %s\n" % library)
        if bool(getattr(request, "USE_ABH_SHELL_SIZE_FILTER", False)):
            x_bounds = tuple(getattr(request, "ABH_SHELL_OVERALL_LENGTH_X_MM_RANGE"))
            y_bounds = tuple(getattr(request, "ABH_SHELL_OVERALL_WIDTH_Y_MM_RANGE"))
            handle.write("- ABH shell outer-envelope constraint: X = %.12g--%.12g mm; Y = %.12g--%.12g mm. Damping and constrained layers are excluded.\n" %
                         (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]))
        if ranked and ranked[0]["Effective_Modal_Mass_Z"] is None:
            handle.write("- Ranking preference: lowest frequency detuning (source data has no effective modal mass or mass fields).\n")
        else:
            handle.write("- Ranking preference: %s\n" %
                         ("higher Z effective modal mass" if request.PREFER_HIGH_EFFECTIVE_MODAL_MASS else "lower detuning then lower mass"))
        handle.write("- Abaqus job submitted: NO\n")
        if library == FIXED_BAND_LIBRARY:
            handle.write("- Damping configuration: fixed physical inward bands, X = 15 mm and Y = 15 mm. Source `rx`/`ry` are physical lengths in metres, not legacy ratios.\n")
        else:
            handle.write("- Mass definition: `RABH_Assembly_Mass` is Batch1 `Total_Attachment_Mass`; it is the ABH shell plus the combined DAMP instance (damping and constrained layers). `Abaqus_Total_Model_Mass_from_DAT` is the separate DAT `TOTAL MASS OF MODEL` value and is not used for ranking.\n")
        if selected:
            handle.write("- Top-ranked database candidate under the current ranking preference: %s, mode %d, %.12g Hz, detuning %.12g %%\n" %
                         (selected["Sample_ID"], selected["Mode_Index"], selected["Mode_Frequency_Hz"],
                          selected["Detuning_Percent"]))
    return {
        "target_directory": target_dir,
        "candidate_pool": pool_path,
        "candidate_rankings": ranking_path,
        "summary": summary_path,
        "selected_parameters": selected_path,
    }


def _regression_candidate_row(item):
    """Map one normal retrieval record to the stable regression CSV schema."""
    row = {
        "Candidate_ID": item["Sample_ID"],
        "ABH_ID": item["ABH_ID"],
        "Mode": item["Mode_Index"],
        "Frequency_Hz": item["Mode_Frequency_Hz"],
        "Detuning_Percent": item["Detuning_Percent"],
        "Effective_Modal_Mass_Z": item["Effective_Modal_Mass_Z"],
        "Effective_Mass_Ratio_Z": item["Effective_Mass_Ratio_Z"],
        "ABH_Shell_Mass": item["ABH_Shell_Mass"],
        "Damping_And_Constrained_Layer_Mass": item["Damping_And_Constrained_Layer_Mass"],
        "RABH_Assembly_Mass": item["RABH_Assembly_Mass"],
        "Damping_Configuration": item["Damping_Configuration"],
    }
    for key in REQUIRED_11D:
        row[key] = item[key]
    return row


def _write_regression_pool(path, pool):
    fields = (
        "Candidate_ID", "ABH_ID", "Mode", "Frequency_Hz", "Detuning_Percent",
        "Effective_Modal_Mass_Z", "Effective_Mass_Ratio_Z", "ABH_Shell_Mass",
        "Damping_And_Constrained_Layer_Mass", "RABH_Assembly_Mass",
    ) + REQUIRED_11D + ("Damping_Configuration",)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in pool:
            writer.writerow(_regression_candidate_row(item))


def _representative_candidates(pool):
    """Return transparent, non-exclusive representatives from one raw pool."""
    if not pool:
        return {"closest": None, "high_meff": None, "lightweight": None}
    deterministic = lambda item: (item["Sample_ID"], item["Mode_Index"])
    return {
        "closest": min(pool, key=lambda item: (item["Detuning_Percent"],) + deterministic(item)),
        "high_meff": min(pool, key=lambda item: (-item["Effective_Modal_Mass_Z"],
                                                    item["Detuning_Percent"],
                                                    item["RABH_Assembly_Mass"]) + deterministic(item)),
        "lightweight": min(pool, key=lambda item: (item["RABH_Assembly_Mass"],
                                                      item["Detuning_Percent"]) + deterministic(item)),
    }


def _summary_value(item, key):
    return "" if item is None else item[key]


def run_host_mode_regression(database_path=MODAL_DATABASE, output_dir=None):
    """Regression-test all-mode retrieval at the three locked Fig.4.4 targets.

    This is intentionally database-only: it reuses ``retrieve`` and does not
    change the normal user-controlled ``design_request.py`` workflow.
    """
    regression_dir = output_dir or ensure_directory(
        os.path.join(RESULTS_INVERSE_DIR, "regression_F1_F2_F3"))
    ensure_directory(regression_dir)
    summary_rows = []
    target_results = []
    for target in REGRESSION_HOST_TARGETS_HZ:
        request = SimpleNamespace(
            TARGET_FREQUENCY_HZ=target,
            FREQUENCY_TOLERANCE_PERCENT=2.0,
            TARGET_DIRECTION="Z",
            PREFER_HIGH_EFFECTIVE_MODAL_MASS=True,
            USE_MASS_FILTER=False,
            MAX_MASS=None,
        )
        pool = retrieve(request, database_path=database_path)
        representatives = _representative_candidates(pool)
        name = "target_%sp%sHz_candidates.csv" % (str(int(target)),
                                                    ("%.2f" % target).split(".")[1])
        _write_regression_pool(os.path.join(regression_dir, name), pool)
        closest, high_meff, lightweight = (representatives["closest"],
                                           representatives["high_meff"],
                                           representatives["lightweight"])
        summary_rows.append({
            "Target_Frequency_Hz": target,
            "Candidate_Count": len(pool),
            "Closest_Candidate": _summary_value(closest, "Sample_ID"),
            "Closest_Mode": _summary_value(closest, "Mode_Index"),
            "Closest_Frequency_Hz": _summary_value(closest, "Mode_Frequency_Hz"),
            "Closest_Detuning_Percent": _summary_value(closest, "Detuning_Percent"),
            "High_Meff_Candidate": _summary_value(high_meff, "Sample_ID"),
            "High_Meff_Mode": _summary_value(high_meff, "Mode_Index"),
            "High_Meff_Frequency_Hz": _summary_value(high_meff, "Mode_Frequency_Hz"),
            "High_Meff_Z": _summary_value(high_meff, "Effective_Modal_Mass_Z"),
            "High_Meff_RABH_Assembly_Mass": _summary_value(high_meff, "RABH_Assembly_Mass"),
            "Lightweight_Candidate": _summary_value(lightweight, "Sample_ID"),
            "Lightweight_Mode": _summary_value(lightweight, "Mode_Index"),
            "Lightweight_Frequency_Hz": _summary_value(lightweight, "Mode_Frequency_Hz"),
            "Lightweight_Detuning_Percent": _summary_value(lightweight, "Detuning_Percent"),
            "Lightweight_RABH_Assembly_Mass": _summary_value(lightweight, "RABH_Assembly_Mass"),
        })
        target_results.append((target, pool, representatives))

    summary_path = os.path.join(regression_dir, "inverse_design_regression_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        fields = tuple(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    baseline_request = SimpleNamespace(
        TARGET_FREQUENCY_HZ=35.0,
        FREQUENCY_TOLERANCE_PERCENT=2.0,
        TARGET_DIRECTION="Z",
        PREFER_HIGH_EFFECTIVE_MODAL_MASS=True,
        USE_MASS_FILTER=False,
        MAX_MASS=None,
    )
    baseline_pool = retrieve(baseline_request, database_path=database_path)
    expected_35hz = ("FORMAL11D_B1_A009_D02", "FORMAL11D_B1_A014_D01")
    found_35hz = set(item["Sample_ID"] for item in baseline_pool)
    baseline_pass = all(sample_id in found_35hz for sample_id in expected_35hz)

    report_path = os.path.join(regression_dir, "INVERSE_DESIGN_V1_REGRESSION.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Inverse Design V1.0 Regression\n\n")
        handle.write("- Database-only all-mode retrieval. Abaqus jobs submitted: NO.\n")
        handle.write("- Frequency tolerance: +/-2 %.\n")
        for target, pool, representatives in target_results:
            handle.write("\n## %.2f Hz\n\n" % target)
            handle.write("- Candidate count: %d\n" % len(pool))
            for label, item in (("CLOSEST_FREQUENCY", representatives["closest"]),
                                ("HIGH_EFFECTIVE_MASS", representatives["high_meff"]),
                                ("LIGHTWEIGHT", representatives["lightweight"])):
                if item is None:
                    handle.write("- %s: none\n" % label)
                else:
                    handle.write("- %s: %s, mode %d, %.12g Hz, detuning %.12g %%, assembly mass %.12g kg\n" %
                                 (label, item["Sample_ID"], item["Mode_Index"],
                                  item["Mode_Frequency_Hz"], item["Detuning_Percent"],
                                  item["RABH_Assembly_Mass"]))
        handle.write("\n## Database evidence\n\n")
        if all(pool for _, pool, _ in target_results):
            handle.write("- The current database returns tolerance-qualified candidates for all three Fig.4.4 Host targets from 34.41 to 213.63 Hz. This is a database-coverage result only.\n")
        else:
            handle.write("- At least one Fig.4.4 Host target has no tolerance-qualified database candidate.\n")
        handle.write("- 35 Hz original retrieval regression: %s. Required candidates: %s.\n" %
                     ("PASS" if baseline_pass else "FAIL", ", ".join(expected_35hz)))

    return {
        "status": "PASS" if baseline_pass and all(pool for _, pool, _ in target_results) else "FAIL",
        "summary": summary_path,
        "report": report_path,
        "target_results": target_results,
        "baseline_35hz_pass": baseline_pass,
    }
