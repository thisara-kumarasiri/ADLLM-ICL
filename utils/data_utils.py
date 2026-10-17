import numpy as np
import pandas as pd


def extract_example(csv_path):
    """
    Load MRI subjects from CSV.

    Expected columns
    ----------------
    Subject
    Group
    Sex (optional)
    Age (optional)

    Remaining columns are treated as MRI features.
    """

    df = pd.read_csv(csv_path)
    required_cols = ["Subject", "Group"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column "
                f"'{col}' in {csv_path}")

    metadata_cols = ["Subject","Group"]
    if "Sex" in df.columns:
        metadata_cols.append("Sex")

    if "Age" in df.columns:
        metadata_cols.append("Age")

    feature_cols = [col for col in df.columns if col not in metadata_cols]

    if not feature_cols:
        raise ValueError(
            f"No MRI feature columns "
            f"found in {csv_path}"
        )

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    all_docs = []

    for _, row in df.iterrows():
        subject_id = str(row["Subject"]).strip()
        label = str(row["Group"]).strip().upper()

        if label not in ["AD", "CN"]:
            raise ValueError(
                f"Unexpected label "
                f"'{label}' for "
                f"{subject_id}."
            )

        features = (row[feature_cols].astype(float).to_numpy())

        subject = {
            "name" : subject_id,
            "gold" : label,
            "features" : features,
            "feature_dict": {feature:float(row[feature]) for feature in feature_cols}
        }

        if "Sex" in df.columns:
            subject["sex"] = str(row["Sex"]).strip()

        if "Age" in df.columns:
            subject["age"] = float(row["Age"])

        all_docs.append(subject)

    return all_docs