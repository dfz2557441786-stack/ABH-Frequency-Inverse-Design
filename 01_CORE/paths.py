"""Single source of project-relative paths for active RABH code."""
from __future__ import print_function

import os


CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)
USER_DESIGN_DIR = os.path.join(PROJECT_ROOT, "00_USER_DESIGN")
ABAQUS_DIR = os.path.join(PROJECT_ROOT, "02_ABAQUS")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "03_DATABASE")
DATABASE_DATA_DIR = os.path.join(DATABASE_DIR, "data")
DATABASE_MODEL_DIR = os.path.join(DATABASE_DIR, "models")
DATABASE_SCALER_DIR = os.path.join(DATABASE_DIR, "scalers")
INVERSE_DESIGN_DIR = os.path.join(PROJECT_ROOT, "04_INVERSE_DESIGN")
POSTPROCESS_DIR = os.path.join(PROJECT_ROOT, "06_POSTPROCESS")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "07_RESULTS")
RESULTS_INVERSE_DIR = os.path.join(RESULTS_DIR, "inverse_design")
DOCS_DIR = os.path.join(PROJECT_ROOT, "08_DOCS")
HISTORICAL_DOCS_DIR = os.path.join(DOCS_DIR, "historical_summaries")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "09_ARCHIVE")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
GENERAL_DESIGN_BOUNDS_FILE = os.path.join(CORE_DIR, "general_design_bounds.json")

# External, user-machine locations are declared once.  Workers may override
# run directories through their documented environment variables, but never
# hard-code another drive path themselves.
VERIFIED_SOURCE_ROOT = os.environ.get(
    "ABH_VERIFIED_SOURCE_ROOT",
    r"C:\Users\ipc\Desktop\阻尼材料参数识别\Project_Material_constrained-三个参数5.8 - 副本",
)
HOST_REFERENCE_ROOT = os.environ.get(
    "ABH_HOST_REFERENCE_ROOT",
    r"C:\Users\ipc\Desktop\Project_optimizer_constrained",
)
ABAQUS_TEMP_DIR = os.environ.get(
    "ABH_ABAQUS_TEMP_DIR", r"E:\ABH_Damping_Frequency_Design_abaqus_temp",
)
ABAQUS_DYNAMIC_RUNS_DIR = os.environ.get(
    "ABH_DYNAMIC_RUNS_DIR", r"E:\ABH_Damping_Frequency_Design_dynamic_runs",
)
ABAQUS_DYNAMIC_15_55_RUNS_DIR = os.environ.get(
    "ABH_DYNAMIC_15_55_RUNS_DIR", r"E:\ABH_Damping_Frequency_Design_dynamic_15_55_runs",
)
ABAQUS_ENERGY_PROBE_RUNS_DIR = os.environ.get(
    "ABH_ENERGY_PROBE_RUNS_DIR", r"E:\ABH_Damping_Frequency_Design_energy_probe_runs",
)
ABAQUS_COUPLED_V12_RUNS_DIR = os.environ.get(
    "ABH_COUPLED_V12_RUNS_DIR", r"E:\ABH_Damping_Frequency_Design_coupled_v12_runs",
)
ABAQUS_COMPLETE_DESIGN_RUNS_DIR = os.environ.get(
    "ABH_COMPLETE_DESIGN_RUNS_DIR", r"E:\ABH_Damping_Frequency_Design_complete_runs")
COMPLETE_DESIGN_RESULTS_DIR = os.path.join(RESULTS_INVERSE_DIR, "complete_F1_F2_F3")
FREQUENCY9D_RESULTS_DIR = os.path.join(RESULTS_INVERSE_DIR, "frequency_9d")
FREQUENCY9D_RUNS_DIR = os.environ.get("ABH_FREQUENCY9D_RUNS_DIR", r"E:\ABH_Damping_Frequency_Design_frequency9d_runs")
TARGET_DRIVEN_RESULTS_DIR = os.path.join(RESULTS_INVERSE_DIR, "target_driven")
TARGET_DRIVEN_RUNS_DIR = os.path.join(FREQUENCY9D_RUNS_DIR, "target_driven")


def ensure_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
