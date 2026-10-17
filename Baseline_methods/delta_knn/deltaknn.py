"""
MRI-adapted implementation of Delta-KNN.

Adapted from the original Delta-KNN implementation provided by the
method authors.

Modifications for this study include:
- MRI-derived tabular features instead of text documents
- standardized feature-space cosine similarity for KNN selection
- AD/CN diagnostic labels
- MRI-specific LLM prompting for Delta-matrix construction
- balanced AD/CN demonstration selection

The underlying Delta-KNN ranking procedure is preserved.
"""

import numpy as np
import pandas as pd

from .calculate_delta_matrix import (store_delta_matrix_and_index)
from .calculate_knn import (get_dk_nearest_examples)

def extract_example(csv_path):
    """
    Load ADNI subjects from a CSV file.

    Expected columns
    ----------------
    Subject : unique subject ID
    Group   : AD or CN
    Sex     : optional demographic column
    Remaining columns : brain-region features

    Returns
    -------
    all_docs : list of dictionaries

        Example:
        {
            "name": "002_S_0295",
            "gold": "AD",
            "sex": "M",
            "features": np.array([...]),
            "feature_dict": {...}
        }
    """

    df = pd.read_csv(csv_path)
    required_cols = ["Subject", "Group"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {csv_path}"
            )

    metadata_cols = ["Subject", "Group"]

    if "Sex" in df.columns:
        metadata_cols.append("Sex")

    if "Age" in df.columns:
        metadata_cols.append("Age")

    feature_cols = [
        col
        for col in df.columns
        if col not in metadata_cols
    ]

    if len(feature_cols) == 0:
        raise ValueError(
            f"No brain-region feature columns found in {csv_path}"
        )

    for col in feature_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    all_docs = []

    for _, row in df.iterrows():
        subject_id = str(row["Subject"]).strip()
        label = str(row["Group"]).strip().upper()

        if label not in ["AD", "CN"]:
            raise ValueError(
                f"Unexpected label '{label}' for subject "
                f"{subject_id}. Expected AD or CN."
            )

        features = (
            row[feature_cols]
            .astype(float)
            .to_numpy()
        )

        subject = {
            "name": subject_id,
            "gold": label,
            "features": features,
            "feature_dict": {
                feature: float(row[feature])
                for feature in feature_cols
            }
        }

        if "Sex" in df.columns:
            subject["sex"] = str(row["Sex"]).strip()

        if "Age" in df.columns:
            subject["age"] = float(row["Age"])

        all_docs.append(subject)

    return all_docs

def prepare_target_examples(
    selection,
    pretrained_model,
    dataset_train,
    dataset_test,
    train_docs,
    test_docs,
    n_shot=4,
    dk=13,
    replace_delta=False
):
    """
    Select demonstrations for each test subject using Delta-KNN.

    Delta matrix orientation:
        rows    = candidate demonstrations
        columns = training subjects used as targets

    For each test subject:
        1. Find dk nearest training subjects.
        2. Use these subjects as representative Delta columns.
        3. Calculate mean Delta across those columns for
           every candidate demonstration.
        4. Rank all training demonstrations.
        5. Select highest-ranked AD and CN demonstrations.

    Parameters
    ----------
    selection : [str]  Must be "knndelta".
    pretrained_model : [str] LLM used when constructing the Delta matrix.
    dataset_train : [str] Training dataset identifier.
    dataset_test : [str] Test dataset identifier.
    train_docs : [list] Training subjects.
    test_docs : [list] Test subjects.
    n_shot : [int] Number of demonstrations. For your experiment this is 4.
    dk : [int] Number of KNN representative training subjects.
    replace_delta : [bool]
        True  -> rebuild Delta matrix
        False -> use stored Delta matrix if available.

    Returns
    -------
    batches : list of dict
    """

    if selection != "knndelta":
        raise ValueError(
            "This implementation currently supports only "
            "selection='knndelta'."
        )

    if n_shot % 2 != 0:
        raise ValueError(
            "n_shot must be even because demonstrations are "
            "balanced between AD and CN."
        )

    if dk <= 0:
        raise ValueError(
            "dk must be greater than 0."
        )

    train_names = [str(doc["name"])for doc in train_docs]
    test_names = [str(doc["name"])for doc in test_docs]

    test_golds = [doc["gold"] for doc in test_docs]

    if len(train_names) != len(set(train_names)):
        raise ValueError(
            "Duplicate subject IDs found in training data."
        )

    if len(test_names) != len(set(test_names)):
        raise ValueError(
            "Duplicate subject IDs found in test data."
        )


    train_label_map = {
        str(doc["name"]): doc["gold"]
        for doc in train_docs
    }

    for subject_name, label in train_label_map.items():

        if label not in ["AD", "CN"]:

            raise ValueError(
                f"Unexpected label '{label}' for training "
                f"subject {subject_name}."
            )

    print("\nLoading Delta matrix...")

    _, delta_mat, doc_index = store_delta_matrix_and_index(
        dataset=dataset_train,
        model=pretrained_model,
        train_docs=train_docs,
        n_repeats=3,
        replace=replace_delta
    )

    doc_index = [str(x) for x in doc_index]
    delta_mat = np.asarray(delta_mat, dtype=float)

    print(
        f"Delta matrix shape: {delta_mat.shape}"
    )

    if delta_mat.ndim != 2:
        raise ValueError(
            "Delta matrix must be two-dimensional."
        )

    if delta_mat.shape[0] != len(doc_index):
        raise ValueError(
            "Number of Delta matrix rows does not match "
            "the number of subjects in doc_index."
        )

    if delta_mat.shape[1] != len(doc_index):
        raise ValueError(
            "Number of Delta matrix columns does not match "
            "the number of subjects in doc_index."
        )

    delta_index_map = {
        subject_name: i
        for i, subject_name in enumerate(doc_index)
    }

    missing_training_subjects = [
        subject_name
        for subject_name in train_names
        if subject_name not in delta_index_map
    ]

    if missing_training_subjects:
        raise ValueError(
            "The following training subjects are missing "
            "from the Delta matrix:\n"
            f"{missing_training_subjects}"
        )

    train_row_ids = np.array([
        delta_index_map[subject_name]
        for subject_name in train_names
    ])


    delta_train = delta_mat[train_row_ids,:]

    print(
        f"Eligible demonstration matrix shape: "
        f"{delta_train.shape}"
    )

    batches = []

    number_per_class = n_shot // 2

    for test_number, (gold, test_name) in enumerate(
        zip(test_golds, test_names),
        start=1
    ):

        print(
            f"\nProcessing test subject "
            f"{test_number}/{len(test_names)}: "
            f"{test_name}"
        )

        # ----------------------------------------------------
        # Step 1: KNN representative selection
        # ----------------------------------------------------

        current_test_doc = next(
            doc
            for doc in test_docs
            if doc["name"] == test_name
        )

        knn_selected, knn_scores = (
            get_dk_nearest_examples(
                train_docs=train_docs,
                test_doc=current_test_doc,
                dk=dk
            )
        )

        knn_selected = [str(x) for x in knn_selected]

        if len(knn_selected) != dk:
            raise ValueError(
                f"Expected {dk} KNN representatives for "
                f"{test_name}, but received "
                f"{len(knn_selected)}."
            )

        missing_knn = [
            subject_name
            for subject_name in knn_selected
            if subject_name not in delta_index_map
        ]

        if missing_knn:
            raise ValueError(
                f"KNN subjects missing from Delta matrix "
                f"for test subject {test_name}:\n"
                f"{missing_knn}"
            )

        knn_column_ids = np.array([
            delta_index_map[subject_name]
            for subject_name in knn_selected
        ])

        delta_knn = delta_train[:,knn_column_ids]

        with np.errstate(invalid="ignore", divide="ignore"):
            row_mean = np.nanmean(delta_knn, axis=1)
            
        row_mean = np.where(np.isnan(row_mean), -np.inf, row_mean)
        sorted_row_indices = np.argsort(row_mean)[::-1]
        sorted_doc_names = [train_names[i] for i in sorted_row_indices]
        sorted_delta_scores = [float(row_mean[i]) for i in sorted_row_indices]

        ranked_ad = [subject_name for subject_name in sorted_doc_names if train_label_map[subject_name] == "AD"]
        ranked_cn = [subject_name for subject_name in sorted_doc_names if train_label_map[subject_name] == "CN"]

        if len(ranked_ad) < number_per_class:
            raise ValueError(
                f"Not enough AD demonstrations available "
                f"for {test_name}."
            )

        if len(ranked_cn) < number_per_class:
            raise ValueError(
                f"Not enough CN demonstrations available "
                f"for {test_name}."
            )

        selected_ad = ranked_ad[:number_per_class]
        selected_cn = ranked_cn[:number_per_class]

        target_docs = (selected_ad + selected_cn)

        print(f"Selected AD: {selected_ad}")
        print(f"Selected CN: {selected_cn}")

        batches.append({
            "name": test_name,
            "gold": gold,
            "knn_representatives": knn_selected,
            "knn_scores": knn_scores,
            "selected_ad": selected_ad,
            "selected_cn": selected_cn,
            "target_docs": target_docs,
            "ranked_subjects": sorted_doc_names,
            "ranked_delta_scores": sorted_delta_scores
        })

    return batches