"""Configuration for the independent ABH damping-frequency design project.

Only the validation tolerance is active until a verified legacy-model adapter is
supplied.  Keeping the adapter location explicit prevents accidentally treating
an old ODB as a newly calculated result.
"""

from __future__ import print_function

import os

from paths import (DATABASE_DATA_DIR, DATABASE_MODEL_DIR, LOG_DIR, PROJECT_ROOT,
                   RESULTS_DIR, VERIFIED_SOURCE_ROOT, GENERAL_DESIGN_BOUNDS_FILE,
                   DOCS_DIR)

# Relative error allowed for each of the first six frequencies: 0.5 %.
FREQUENCY_RELATIVE_ERROR_TOLERANCE = 0.005

# The verified legacy project must supply these values before validation can
# submit Abaqus.  The adapter is expected to:
#   1. build the verified model from the supplied baseline parameters;
#   2. submit one Frequency job in ``run_directory``;
#   3. return the six reference frequencies and the newly written ODB path.
# No fallback to a pre-existing ODB is permitted.
LEGACY_PROJECT_ROOT = PROJECT_ROOT
LEGACY_VALIDATION_ADAPTER = "legacy_validation_adapter.run_fresh_validation"

# Fresh validation artifacts always live below this project directory.
VALIDATION_RESULTS_DIRECTORY = os.path.join(RESULTS_DIR, "validation")

# Immutable, read-only source of the fixed case audited in the user's
# material-identification project.  New code imports these functions but never
# writes to this directory.
VERIFIED_SHELL_CREATE = os.path.join(VERIFIED_SOURCE_ROOT, "abaqus_module", "Shell_create.py")
VERIFIED_BOUNDARY_SETUP = os.path.join(VERIFIED_SOURCE_ROOT, "abaqus_module", "Boundary_Set.py")
REFERENCE_FREQUENCY_HZ = (20.091, 30.565, 33.594, 59.931, 72.843, 83.958)
REFERENCE_FREQUENCY_SOURCE = (
    "fixed_frequency_audit.odb, audited fixed case; values are recorded here "
    "rather than read from an old ODB during validation"
)

# Fixed, audited baseline.  The material constants intentionally match the
# verified source and are not optimisation variables in this project.
FIXED_BASELINE_PARAMETERS = {
    "Rou_2": 157.28e-3,
    "Rou_3": 215.64e-3,
    "Rou_4": 168.79e-3,
    "theta": 50.0 / 180.0 * 3.141592653589793,
    "equal_t_length": 30.0e-3,
    "equal_t_width": 30.0e-3,
    "platform_height": 5.0e-3,
    "platform_length": 25.0e-3,
    "platform_width": 25.0e-3,
    "cutoff_thickness": 0.4e-3,
    "power_law": 3.0,
    "hole_radius": 4.0e-3,
    # Fixed damping material now follows the checked ABH_Frequency_Dataset_V2_2
    # reference model.  It is not a design variable.
    "damp_elastic": 2.5831e7,
    "damp_thickness": 3.0e-3,
    # The two dimensions are the dimensions of the central rectangular
    # opening in the damping/constraint ring.  They correspond exactly to
    # the audited legacy 40 mm equal inward offset at this baseline.
    "damp_length": 0.178286161624823,
    "damp_width": 0.244513888852735,
}

FIXED_ALUMINUM_PARAMETERS = {
    "density": 2.719e3,
    "elastic": 70630000000.0,
    "poisson": 0.32,
    "loss_factor": 0.001,
}
FIXED_MATERIAL_PARAMETERS = {
    "damp_density": 2139.201679,
    "damp_poisson": 0.4463,
    "damp_loss_factor": 0.7,
    "Constrained_Layer_thickness": 0.1e-3,
    "constrained_elastic": 68915063391.0,
    "constrained_density": 2.73e3,
    "constrained_poisson": 0.33,
    "constrained_loss_factor": 0.001,
}
LEGACY_MATERIAL_PARAMETERS = dict(FIXED_MATERIAL_PARAMETERS)
CALIBRATION_STATUS = "aligned_to_ABH_Frequency_Dataset_V2_2_reference"
MATERIAL_MODEL_STATUS = "fixed_reference_material_parameters"
MATERIAL_MODEL_NOTE = "Al, damping and constrained-layer values match the checked ABH_Frequency_Dataset_V2_2 model"

# Single source for the locked 35 Hz controlled-validation solver policy.
# The policy is enforced in the generated input before an Abaqus job can be
# submitted; it does not alter geometry, material, loading, or mesh settings.
CONTROLLED_VALIDATION_SOLVER_POLICY = {
    "frequency_eigensolver": "LANCZOS",
    "sim_selective_recovery": False,
    "normalization": "MASS",
}
# Coupled-modal validation uses one bounded modal request for every candidate.
# Eighty requested modes is intentionally an extraction policy (not a physical
# parameter) and provides a common spectrum beyond the 0--300 Hz diagnostic
# window without requesting every model degree of freedom.
CONTROLLED_COUPLED_MODAL_POLICY = {
    "upper_frequency_hz": 300.0,
    "requested_mode_count": 80,
}
# Single source for every current V2 Host--RABH assembly.  Both shell
# sections are OFFSET=SPOS, so the physical joint is ABH centre-platform SNEG
# to Plate SPOS.  The actual RABH Z translation is derived per candidate from
# platform_height; no fixed translation is physically valid across the 11D
# database.
V2_HOST_RABH_INTERFACE_RULE = "RABH_PLATFORM_SNEG_TO_PLATE_SPOS_H_CENTER_ALIGNED_V4"

# Single source for the existing Fig.4.4 Host plus current-V2-RABH damping
# definitions.  These values match the locked Host builder and V2 Shell_create
# source; they are used as pre-submit assertions, not as new material inputs.
CONTROLLED_VALIDATION_DAMPING_POLICY = {
    "HOST_STRUCTURAL_DAMPING": 0.001,
    "RABH_AL_DAMPING": 0.001,
    "CONSTRAINED_LAYER_DAMPING": 0.001,
    "DAMPING_LAYER_LOSS_FACTOR": 0.7,
}
GENERAL_MODEL_DIMENSION = 11
# Finite, opt-in validation budget for the complete database-led design task.
COMPLETE_DESIGN_POLICY = {
    "max_new_jobs": 30, "max_local_candidates": 12,
    "local_candidates_per_target": 2, "job_timeout_seconds": 7200,
    "minimum_free_gb": 15.0,
    "wide_band_hz": (10.0, 400.0), "dynamic_df_hz": 0.05,
    "target_band_relative_halfwidth": 0.10,
    "wideband_minimum_reduction_db": 0.0,
    "host_mac_diagnostic_threshold": 0.6,
}
GENERAL_11D_DOE_STATUS = "PILOT_AUTHORIZED_FORMAL_DOE_NOT_STARTED"
GP_UNCERTAINTY_STATUS = "UNRELIABLE_FOR_ACTIVE_LEARNING"
RX_RY_RANGE_STATUS = "numerical_pilot_range"
# Numerical separation from radical/thickness equality limits during the joint
# Pilot.  This is a safety margin, not a changed physical design bound.
GENERAL_GEOMETRY_SAFETY_EPSILON_M = 1.0e-4
TARGET_NEW_JOINT_PILOT = 20
N_NEW_ABH_GEOMETRIES = 15
N_DAMPING_DESIGNS_PER_ABH = 4
FORMAL_11D_BATCH1_SOURCE = "FORMAL_11D_DOE_BATCH_1"
FORMAL_11D_BATCH1_MIN_SUCCESS_RATE = 0.90
FORMAL_BATCH1_MAX_NEW_ABAQUS_RUNS = 60
FORMAL_11D_DOE_STATUS = "BATCH_1_AUTHORIZED"
THICKNESS_FIELD_IMPLEMENTATION = "DOMAIN_SAFE_CONTINUOUS_POWER_V2"
LEGACY_ANALYTICAL_FIELD_REFERENCE = "LEGACY_ANALYTICAL_FIELD_REFERENCE"
GEOMETRY_BUILDER_VERSION = "VERIFIED_SHELL_CREATE_TWO_AXIS_DOMAIN_SAFE_V2"
# Current V2 builder with the retained rectangular-frame topology.  The name
# is metadata only; it does not select a different physical geometry path.
DAMP_GEOMETRY_MODE = "V2_DOMAIN_SAFE_FRAME_XY"
RECT_PATCH_DISABLED = True
RANGE_DISCOVERY_ONLY = True
MATERIAL_TRANSFER_VALIDATION_STATUS = "PENDING_DATA"
HOST_STRUCTURE_VALIDATION_STATUS = "PENDING_HOST_STRUCTURE_VALIDATION"
FRAME_FIXED_PARAMETERS = dict(FIXED_BASELINE_PARAMETERS, damp_elastic=2.5831e7,
                              damp_thickness=3.0e-3, damp_inward_x=40.0e-3,
                              damp_inward_y=40.0e-3)
FRAME_OFAT_POINTS = (
    ("base", 40.0e-3, 40.0e-3), ("x_low", 30.0e-3, 40.0e-3),
    ("x_high", 50.0e-3, 40.0e-3), ("y_low", 40.0e-3, 30.0e-3),
    ("y_high", 40.0e-3, 50.0e-3),
)

# Local smoke perturbations only; they are not design-space bounds and must not
# be used to start LHS/DOE.  Physical Ld/Wd/td bounds remain explicitly unset.
SMOKE_FACTORS = (0.90, 1.00, 1.10)
MESH_LEVELS = {
    "coarse": {"shell_mesh_size": 0.0045, "damp_mesh_size": 0.0040},
    "medium": {"shell_mesh_size": 0.0035, "damp_mesh_size": 0.0030},
    "fine": {"shell_mesh_size": 0.0025, "damp_mesh_size": 0.0020},
}
MESH_CONVERGENCE_FREQUENCY_TOLERANCE = 0.01
MAX_NEW_ABAQUS_RUNS = 35
ALL_INPUT_PARAMETERS = tuple(sorted(set(FIXED_BASELINE_PARAMETERS) | set(FIXED_MATERIAL_PARAMETERS)))

FIXED_ABH_VARIABLES = ("damp_inward_x", "damp_inward_y")
GENERAL_ABH_VARIABLES = (
    "Rou_2", "Rou_3", "Rou_4", "theta", "equal_t_length",
    "equal_t_width", "platform_height", "cutoff_thickness", "power_law",
    "damp_inward_x", "damp_inward_y",
)
RESERVED_FIXED_INTERFACE_VARIABLES = ("platform_length", "platform_width", "hole_radius")

# All numerical limits and manufacturing constraints live here.  Bounds are
# for future sample generation only; they do not launch simulations.
PARAMETER_BOUNDS = {
    "Rou_2": (0.13, 0.19), "Rou_3": (0.18, 0.25), "Rou_4": (0.14, 0.20),
    "theta": (35.0 / 180.0 * 3.141592653589793, 55.0 / 180.0 * 3.141592653589793),
    "equal_t_length": (0.01, 0.05), "equal_t_width": (0.01, 0.05),
    "platform_height": (0.003, 0.008), "platform_length": (0.015, 0.04),
    "platform_width": (0.015, 0.04), "cutoff_thickness": (0.0002, 0.001),
    "power_law": (1.0, 5.0), "hole_radius": (0.001, 0.008),
    "damp_elastic": (2.5831e7, 2.5831e7), "damp_thickness": (0.003, 0.003),
    # Formal ranges remain intentionally unset; only the recorded OFAT points are valid now.
    "damp_inward_x": None, "damp_inward_y": None,
}
MIN_DAMP_RING_WIDTH = 0.0005
GEOMETRY_EQUALITY_TOLERANCE = 1.0e-10
DEFAULT_LHS_SAMPLE_COUNT = 50
DEFAULT_RANDOM_SEED = 20260919
DATASET_FILENAME = os.path.join(DATABASE_DATA_DIR, "abaqus_samples.csv")
MANIFEST_FILENAME = "run_manifest.json"
MODEL_DIRECTORY = DATABASE_MODEL_DIR
LOG_DIRECTORY = LOG_DIR
MODE_TRACKING_NOTE = (
    "Frequency_1_Hz through Frequency_12_Hz remain frequency-sorted eigenvalues. "
    "Diagnostic MAC uses only common ABH_SHELL nodes and U degrees of freedom; "
    "DAMP nodes are intentionally excluded because their mesh changes with dx/dy."
)

# Fixed-ABH numerical-pilot controls only.  The tolerance is a configurable
# engineering screen for the reachability report, not a claimed paper standard.
DESIGN_DOMAIN_STATUS = "numerical_pilot_only"
PILOT_DX_DY_LIMITS_M = (20.0e-3, 60.0e-3)
TARGET_FREQ_REL_TOLERANCE = 0.01
NEXT_STAGE_ABAQUS_JOB_TIMEOUT_SECONDS = 1200
NEXT_STAGE_LOCAL_POLL_SECONDS = 45
MODE_TRACKING_STATUS = "partial_with_ambiguous_regions"
DESIGN_SPACE_BOUNDARY_COVERAGE = "required"
MAX_NEW_ABAQUS_RUNS_FOR_THIS_TEST = 2
CLAMPED_INTERFACE_REFERENCE = True
COMPLETE_DESIGN_SUSPENDED = True  # User switched priority to isolated 9D inverse design.
FREQUENCY9D_POLICY = {
    "seed": 20261010, "reference_damping_id": "D02",
    "max_jobs": 12, "initial_probes": 3, "candidate_count": 8,
    "match_percent": 2.0, "probe_stop_percent": 5.0,
    "job_timeout_seconds": 2400, "minimum_free_gb": 10.0,
    "min_database_distance": 0.08, "min_pair_distance": 0.12,
    "max_support_distance": 0.35, "local_halfwidth": 0.15,
    "cv_folds": 5, "primary_mode_count": 8,
}
# Fixed 15 mm inverse generation is intentionally conservative.  Generation may only make a new
# geometry close to an *actual* target-matching database anchor.  A predicted
# exact frequency alone never qualifies a candidate for Abaqus validation.
FIXED_BAND_TARGET_GENERATION_POLICY = {
    "candidate_count": 8, #严格通过不确定度门槛的正式候选数量
    "match_percent": 2.0,
    "anchor_detuning_percent": 1.0,
    "local_trust_radius": 0.035,
    "minimum_new_distance": 0.004,
    "minimum_pair_distance": 0.060,
    "maximum_prediction_std_percent": 2.0,
    "upper_confidence_multiplier": 1.96,
    # 严格门槛未通过时，仍可单独导出少量“探索候选”，供人工选择后
    # 进行独立 Abaqus 验证。它们绝不写入严格候选表，也不表示频率已满足。
    "exploratory_candidate_count": 6, #严格门槛未通过时，仍导出供 Abaqus 验证的探索候选数量。
    "exploratory_max_prediction_std_percent": 12.0,
}
# 零有限元成本的模型检查：只比较实验模型，不自动替换生产模型。
# 训练、校准、测试按独立/近重复几何分组隔离，测试结果不可回填训练。
FIXED_BAND_MODEL_CHECK_POLICY = {
    "version": "GROUP_HOLDOUT_CALIBRATION_1",
    "seed": 20261015,
    "near_duplicate_radius": 0.02,
    "test_fraction": 0.20,
    "calibration_fraction_of_development": 0.25,
    "coverage": 0.95,
    "minimum_calibration_rows": 40,
    "target_training_band_fraction": 0.25,
    "minimum_mode_support": 20,
    "local_training_rows": 300,
    "optimizer_maxiter": 60,
}
TARGET_DRIVEN_POLICY = {
    "seed": 20261011, "targets": (34.41, 129.30, 213.63),
    "max_jobs": 12, "candidate_count": 8, "starts": 48,
    "min_geometry_distance": 0.08, "min_pair_distance": 0.12,
    "trust_radius": 0.35, "local_halfwidth": 0.15,
    "local_groups": 24, "minimum_local_groups": 12,
    "match_percent": 2.0, "job_timeout_seconds": 2400,
    "minimum_free_gb": 10.0,
    "minimum_available_memory_gb": 6.0,
}
TARGET_RECOVERY_RESOURCES = {
    "cpus": 1, "memory_mb": 2048,
    "minimum_available_memory_gb": 4.0,
    "minimum_commit_headroom_gb": 8.0,
    "minimum_free_gb": 10.0,
}

# Explicit 2026-10-12 experiment. Production defaults are unchanged.
MESH_MEMORY_EXPERIMENT_RESOURCES = {
    "experiment": "SINGLE_R_MEMORY6GB", "cpus": 1, "memory_mb": 6144,
    "minimum_available_memory_gb": 8.0,
    "minimum_commit_headroom_gb": 14.0,
    "minimum_free_gb": 10.0,
    "warning_available_memory_gb": 2.0,
    "warning_commit_headroom_gb": 3.0,
    "emergency_available_memory_gb": 0.5,
    "emergency_commit_headroom_gb": 1.0,
    "emergency_consecutive_samples": 3,
}

STAGED_ISOLATED_RESOURCES = {
    "cpus": 1, "memory": "90%", "mp_mode": "threads",
    "minimum_available_memory_gb": 8.0, "minimum_commit_headroom_gb": 14.0,
    "minimum_free_gb": 10.0,
    # Monitoring is informational at this level. A successful manual 90% job
    # legitimately crossed 2 GiB available RAM, so this must never kill a job.
    "warning_available_memory_gb": 2.0,
    "warning_commit_headroom_gb": 3.0,
    # Emergency stop requires *both* resources to be critically exhausted for
    # consecutive samples. This protects Windows without treating a transient
    # low available-RAM value as an Abaqus failure.
    "emergency_available_memory_gb": 0.5,
    "emergency_commit_headroom_gb": 1.0,
    "emergency_consecutive_samples": 3,
}
