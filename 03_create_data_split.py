"""
Create a reproducible stratified train/validation/test split.

The split is generated once and saved to disk. All subsequent
experiments use these saved files so that every example-selection
method and LLM is evaluated using exactly the same subjects.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import Settings as cfg


# ============================================================
# CREATE SPLIT
# ============================================================

def create_split(
    df,
    label_column="Group",
    train_size=0.7,
    validation_size=0.1,
    test_size=0.2,
    random_seed=42,
):

    if round(
        train_size + validation_size + test_size,
        10,
    ) != 1.0:
        raise ValueError(
            "train_size + validation_size + test_size "
            "must equal 1."
        )

    # --------------------------------------------------------
    # Train vs validation + test
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=validation_size + test_size,
        stratify=df[label_column],
        random_state=random_seed,
    )

    # --------------------------------------------------------
    # Validation vs test
    # --------------------------------------------------------

    relative_test_size = (
        test_size
        / (validation_size + test_size)
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[label_column],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# SUMMARY
# ============================================================

def print_split_summary(
    name,
    df,
    label_column,
):

    print(f"\n{name}")
    print("-" * 40)
    print(f"Subjects: {len(df)}")

    print(
        df[label_column]
        .value_counts()
        .sort_index()
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create and save a reproducible stratified "
            "train/validation/test split."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=cfg.ALL_FEATURES_CSV,
        help=(
            "Feature-vector CSV. Defaults to "
            "Settings.ALL_FEATURES_CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.DATA_SPLIT_DIR,
        help=(
            "Directory for saved dataset splits. Defaults to "
            "Settings.DATA_SPLIT_DIR."
        ),
    )

    parser.add_argument(
        "--subject-column",
        default="Subject",
        help="Name of the subject ID column.",
    )

    parser.add_argument(
        "--label-column",
        default="Group",
        help="Name of the diagnostic label column.",
    )

    parser.add_argument(
        "--train-size",
        type=float,
        default=cfg.SPLIT_TRAIN_SIZE,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=cfg.SPLIT_VALIDATION_SIZE,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=cfg.SPLIT_TEST_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.RANDOM_SEED,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Feature-vector file does not exist: "
            f"{args.input}"
        )

    df = pd.read_csv(
        args.input
    )

    required_columns = [
        args.subject_column,
        args.label_column,
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column not found: {column}"
            )

    # --------------------------------------------------------
    # Ensure one row per subject
    # --------------------------------------------------------

    if (
        df[args.subject_column]
        .duplicated()
        .any()
    ):

        duplicates = (
            df.loc[
                df[
                    args.subject_column
                ].duplicated(),
                args.subject_column,
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate subject IDs found: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    train_df, validation_df, test_df = (
        create_split(
            df=df,
            label_column=args.label_column,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_seed=args.seed,
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_split_summary(
        "TRAIN",
        train_df,
        args.label_column,
    )

    print_split_summary(
        "VALIDATION",
        validation_df,
        args.label_column,
    )

    print_split_summary(
        "TEST",
        test_df,
        args.label_column,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        args.output_dir
        / "train.csv"
    )

    validation_path = (
        args.output_dir
        / "val.csv"
    )

    test_path = (
        args.output_dir
        / "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save exact subject assignments
    # --------------------------------------------------------

    train_df[
        [args.subject_column]
    ].to_csv(
        args.output_dir
        / "train_subjects.csv",
        index=False,
    )

    validation_df[
        [args.subject_column]
    ].to_csv(
        args.output_dir
        / "val_subjects.csv",
        index=False,
    )

    test_df[
        [args.subject_column]
    ].to_csv(
        args.output_dir
        / "test_subjects.csv",
        index=False,
    )

    print(
        "\nDataset splits saved to:"
    )

    print(
        f"  Train:      {train_path}"
    )

    print(
        f"  Validation: {validation_path}"
    )

    print(
        f"  Test:       {test_path}"
    )


if __name__ == "__main__":
    main()