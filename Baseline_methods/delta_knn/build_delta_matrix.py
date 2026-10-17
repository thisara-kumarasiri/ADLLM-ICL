from deltaknn import extract_example
from calculate_delta_matrix import store_delta_matrix_and_index

import Settings as cfg

# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = (
    cfg.DATA_SPLIT_DIR / "train.csv"
)

DATASET_TRAIN = "adni-train"
MODEL = cfg.LLM_MODEL
N_REPEATS = 3

print("Loading training subjects...")

train_docs = extract_example(TRAIN_CSV)

print(
    f"Number of training subjects: "
    f"{len(train_docs)}"
)

_, delta_matrix, doc_index = (
    store_delta_matrix_and_index(
        dataset=DATASET_TRAIN,
        model=MODEL,
        train_docs=train_docs,
        n_repeats=N_REPEATS,
        replace=False
    )
)

print("\nDelta Matrix completed.")

print(
    f"Shape: {delta_matrix.shape}"
)

print(
    f"Number of subjects: "
    f"{len(doc_index)}"
)

print(
    "\nFirst five subject IDs:"
)

print(
    doc_index[:5]
)