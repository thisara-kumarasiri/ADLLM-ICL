from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# FREESURFER PROCESSING
# ============================================================

MRI_INPUT_DIR = (PROJECT_ROOT / "ADNI" / "raw_images")
FREESURFER_SUBJECTS_DIR = (PROJECT_ROOT / "FREESURFER_Extractions")
FREESURFER_MAX_SUBJECTS = 3
FREESURFER_THREADS_PER_SUBJECT = 5
FREESURFER_RUN_BRAINSTEM = True

# ============================================================
# FEATURE VECTOR CREATION
# ============================================================

SUBJECT_INFO_FILE = (PROJECT_ROOT / "ADNI" / "subject_info.csv")
ALL_FEATURES_CSV = (PROJECT_ROOT / "all_features.csv")

# ============================================================
# DATA SPLIT
# ============================================================

DATA_SPLIT_DIR = (PROJECT_ROOT / "Data")

# ============================================================
# 04 - MACHINE LEARNING MODELS
# ============================================================

ML_MODEL_OUTPUT_ROOT = (PROJECT_ROOT / "ML")

# ============================================================
# EXAMPLE SELECTION
# ============================================================

# Delta KNN parameters
DELTA_KNN_K = 13
REPLACE_DELTA = False

N_SHOT = 0
LLM_MODEL = "grok-4.20-reasoning" # gpt5.4 / grok-4.20-reasoning
SELECTION_METHOD = ("lr_weighted_xgboost_path_gain")
EXAMPLE_SELECTION_OUTPUT_ROOT = (PROJECT_ROOT / "Examples_Selected")

# ============================================================
# LLM INFERENCE
# ============================================================

NEURO = "Alzheimer's disease"
INFORMATION = ("brain region volumes and cortical thickness values")

RETRY_COUNT = 5
RETRY_WAIT_SECONDS = 5
LLM_PARALLEL_CALLS = 4

LLM_RESULTS_ROOT = (PROJECT_ROOT / "LLM_results")
LLM_N_REPEATS = 3

# ============================================================
# REASONING 
# ============================================================

REASONING_ROOT = (PROJECT_ROOT / "Reasoning")