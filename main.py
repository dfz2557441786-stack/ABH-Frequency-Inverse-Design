"""用户入口：按目标频率检索或生成独立 ABH 构型。"""
from __future__ import print_function

import json
import argparse
import csv

import project_bootstrap

project_bootstrap.activate()

import design_request
from target_retrieval import rank_candidates, retrieve, write_design_output


def _shell_size_bounds_from_request():
    """Pass the user-edited shell-envelope limits to generation functions."""
    if not bool(getattr(design_request, "USE_ABH_SHELL_SIZE_FILTER", False)):
        return None
    return {
        "length_x_mm": tuple(getattr(design_request, "ABH_SHELL_OVERALL_LENGTH_X_MM_RANGE")),
        "width_y_mm": tuple(getattr(design_request, "ABH_SHELL_OVERALL_WIDTH_Y_MM_RANGE")),
    }


def main(design_mode=None):
    # 日常使用只需修改 ``00_USER_DESIGN/design_request.py``，然后运行本文件。
    # 默认 RETRIEVAL 仅查询数据库，不会提交 Abaqus 作业。
    # 模型检查始终为零 Abaqus 模式，优先于任何历史验证开关。
    design_mode = design_mode or getattr(design_request, "DESIGN_MODE", "RETRIEVAL")
    # 远距离探索始终只生成参数，任何验证开关都不能在此入口提交作业。
    if design_mode == "GENERATE_FAR":
        if design_request.DATA_LIBRARY != "FIXED_BAND_15MM_V2":
            raise ValueError("FAR_SEARCH_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import run_fixed_band_far_generation
        return run_fixed_band_far_generation(
            design_request.TARGET_FREQUENCY_HZ,
            design_request.GENERATIVE_ADDITIONAL_TARGETS_HZ,
            design_request.FIXED_BAND_DATASET_CSV,
            design_request.FAR_SEARCH_POLICY,
            _shell_size_bounds_from_request())
    # 直接全局多解反演：不要求数据库中已有频率锚点；只生成待验证参数，
    # 不提交 Abaqus。锚点局部生成 ``GENERATE`` 保留为高可信近邻路线。
    if design_mode == "GENERATE_DIRECT":
        if design_request.DATA_LIBRARY != "FIXED_BAND_15MM_V2":
            raise ValueError("DIRECT_GLOBAL_SEARCH_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import run_fixed_band_direct_global_generation
        return run_fixed_band_direct_global_generation(
            design_request.TARGET_FREQUENCY_HZ,
            design_request.GENERATIVE_ADDITIONAL_TARGETS_HZ,
            design_request.FIXED_BAND_DATASET_CSV,
            design_request.FAR_SEARCH_POLICY,
            _shell_size_bounds_from_request())
    if design_mode in ("MODEL_CHECK", "ANCHOR_CHECK"):
        if getattr(design_request, "DATA_LIBRARY", "") != "FIXED_BAND_15MM_V2":
            raise ValueError("MODEL_CHECK_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import run_fixed_band_model_check, run_fixed_band_anchor_check
        checker = run_fixed_band_anchor_check if design_mode == "ANCHOR_CHECK" else run_fixed_band_model_check
        return checker(
            design_request.TARGET_FREQUENCY_HZ,
            design_request.GENERATIVE_ADDITIONAL_TARGETS_HZ,
            design_request.FIXED_BAND_DATASET_CSV,
        )
    # 将已经完成的 FE 验证整理为反馈库；只读台账，不训练、不生成、不提交 Abaqus。
    if design_mode == "VALIDATION_FEEDBACK":
        if getattr(design_request, "DATA_LIBRARY", "") != "FIXED_BAND_15MM_V2":
            raise ValueError("VALIDATION_FEEDBACK_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import build_fixed_band_validation_feedback
        return build_fixed_band_validation_feedback()
    if design_mode == "VALIDATION_SPECTRUM_RECOVERY":
        if getattr(design_request, "DATA_LIBRARY", "") != "FIXED_BAND_15MM_V2":
            raise ValueError("VALIDATION_SPECTRUM_RECOVERY_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import recover_fixed_band_validation_spectra
        return recover_fixed_band_validation_spectra()
    if design_mode == "VALIDATION_BACKFILL_COMPLETE24":
        if getattr(design_request, "DATA_LIBRARY", "") != "FIXED_BAND_15MM_V2":
            raise ValueError("VALIDATION_BACKFILL_COMPLETE24_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import append_complete_validation_spectra_to_fixed_band_dataset
        return append_complete_validation_spectra_to_fixed_band_dataset(
            design_request.FIXED_BAND_DATASET_CSV)
    # 批量验证优先于单候选验证：候选 ID 由用户配置文件显式指定，按顺序串行提交。
    if getattr(design_request, "VALIDATE_BATCH_NEW_CANDIDATES", False):
        candidate_csv = getattr(design_request, "BATCH_VALIDATION_CANDIDATE_CSV", "").strip() or None
        if getattr(design_request, "VALIDATE_ALL_CANDIDATES_FROM_CSV", False):
            if not candidate_csv:
                raise ValueError("BATCH_VALIDATION_CANDIDATE_CSV_REQUIRED_FOR_VALIDATE_ALL")
            with open(candidate_csv, newline="", encoding="utf-8-sig") as stream:
                batch_ids = tuple(str(row.get("Candidate_ID", "")).strip()
                                  for row in csv.DictReader(stream)
                                  if str(row.get("Candidate_ID", "")).strip())
        else:
            batch_ids = tuple(getattr(design_request, "BATCH_CANDIDATE_IDS", ()))
        if not batch_ids:
            raise ValueError("BATCH_CANDIDATE_IDS_REQUIRED")
        if getattr(design_request, "VALIDATE_ONE_NEW_CANDIDATE", False):
            raise ValueError("CHOOSE_BATCH_OR_SINGLE_VALIDATION_NOT_BOTH")
        if getattr(design_request, "DATA_LIBRARY", "") != "FIXED_BAND_15MM_V2":
            raise ValueError("CURRENT_BATCH_VALIDATION_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import validate_fixed_band_saved_candidates_batch
        return validate_fixed_band_saved_candidates_batch(
            batch_ids,
            candidate_csv,
            bool(getattr(design_request, "BATCH_RETRY_PREVIOUSLY_ATTEMPTED", False)),
        )
    # 单候选验证只允许提交用户指定的一组新 ABH。
    if getattr(design_request, "VALIDATE_ONE_NEW_CANDIDATE", False):
        candidate_id = str(getattr(design_request, "CANDIDATE_ID_TO_VALIDATE", "")).strip()
        if not candidate_id:
            raise ValueError("CANDIDATE_ID_TO_VALIDATE_REQUIRED")
        if getattr(design_request, "DATA_LIBRARY", "") != "FIXED_BAND_15MM_V2":
            raise ValueError("CURRENT_VALIDATION_ENTRY_REQUIRES_FIXED_BAND_15MM_V2")
        from frequency_generation import validate_fixed_band_saved_candidate
        return validate_fixed_band_saved_candidate(
            candidate_id,
            getattr(design_request, "SINGLE_VALIDATION_CANDIDATE_CSV", "").strip() or None,
        )
    design_mode = design_mode or getattr(design_request, "DESIGN_MODE", "RETRIEVAL")
    if design_mode != "RETRIEVAL":
        from frequency_generation import run_frequency_design, run_fixed_band_frequency_generation
        if getattr(design_request, "DATA_LIBRARY", "") == "FIXED_BAND_15MM_V2":
            return run_fixed_band_frequency_generation(
                design_mode, design_request.TARGET_FREQUENCY_HZ,
                design_request.GENERATIVE_ADDITIONAL_TARGETS_HZ,
                getattr(design_request, "FIXED_BAND_DATASET_CSV", ""),
                _shell_size_bounds_from_request(),
            )
        return run_frequency_design(design_mode, design_request.TARGET_FREQUENCY_HZ,
                                    design_request.GENERATIVE_ADDITIONAL_TARGETS_HZ)
    pool = retrieve(design_request)
    candidates = rank_candidates(design_request, pool)
    outputs = write_design_output(design_request, pool, candidates)
    payload = {
        "Status": "SUCCESS",
        "Target_Frequency_Hz": design_request.TARGET_FREQUENCY_HZ,
        "Candidates_Found": len(pool),
        "Top_Candidates": [
            {"Sample_ID": item["Sample_ID"], "Mode_Index": item["Mode_Index"],
             "Mode_Frequency_Hz": item["Mode_Frequency_Hz"],
             "Detuning_Percent": item["Detuning_Percent"]}
            for item in candidates[:int(design_request.TOP_N)]
        ],
        "Validation_Requests_Recorded": {
            "RUN_COUPLED_VALIDATION": bool(design_request.RUN_COUPLED_VALIDATION),
            "RUN_DYNAMIC_VALIDATION": bool(design_request.RUN_DYNAMIC_VALIDATION),
        },
        "Abaqus_Job_Submitted": False,
        "Outputs": outputs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-mode", choices=("RETRIEVAL","GENERATE","GENERATE_FAR","GENERATE_DIRECT","MODEL_CHECK","ANCHOR_CHECK","VALIDATION_FEEDBACK","VALIDATION_SPECTRUM_RECOVERY","VALIDATION_BACKFILL_COMPLETE24","VALIDATE","GENERATE_AND_VALIDATE"))
    args = parser.parse_args()
    main(args.design_mode)
