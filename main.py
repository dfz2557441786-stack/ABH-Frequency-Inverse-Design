"""用户入口：按目标频率检索或生成独立 ABH 构型。"""
from __future__ import print_function

import json
import argparse

import project_bootstrap

project_bootstrap.activate()

import design_request
from target_retrieval import rank_candidates, retrieve, write_design_output


def main(design_mode=None):
    # 日常使用只需修改 ``00_USER_DESIGN/design_request.py``，然后运行本文件。
    # 默认 RETRIEVAL 仅查询数据库，不会提交 Abaqus 作业。
    # 单候选验证优先级最高：只允许提交用户指定的一组新 ABH。
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
    parser.add_argument("--design-mode", choices=("RETRIEVAL","GENERATE","VALIDATE","GENERATE_AND_VALIDATE"))
    args = parser.parse_args()
    main(args.design_mode)
