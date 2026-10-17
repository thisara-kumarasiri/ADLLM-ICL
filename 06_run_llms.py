"""
Build prompts in memory from the selected in-context examples and
run the configured LLM.

The example-selection method and model are controlled through
Settings.py.

Pipeline
--------
1. Load train.csv and test.csv
2. Load the selected examples for cfg.SELECTION_METHOD
3. Build one prompt per test subject in memory
4. Run the configured LLM for cfg.LLM_N_REPEATS
5. Save raw responses
6. Resume incomplete runs automatically if output files already exist

"""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from numbers import Number
from pathlib import Path

import hashlib
import time

import pandas as pd
from tqdm import tqdm

import Settings as cfg

from utils.client import (
    create_client,
    response_to_text,
)

from utils.prompts import (
    INST_SYS_CTX,
    INST_USR_REASONING,
    INST_USR_REASONING_ZERO_SHOT,
)

from utils.extraction_functions import process_file

# ============================================================
# CONSTANTS
# ============================================================

SUBJECT_COLUMN = "Subject"
GROUP_COLUMN = "Group"
AGE_COLUMN = "Age"
SEX_COLUMN = "Sex"

# ============================================================
# VALUE FORMATTING
# ============================================================

def format_value(
    value,
    decimal_places: int = 4,
):
    """
    Format values for inclusion in the prompt.
    """

    if pd.isna(value):
        return "Missing"

    if isinstance(value, Number):
        return (
            f"{float(value):.{decimal_places}f}"
            .rstrip("0")
            .rstrip(".")
        )

    return str(value)

# ============================================================
# DEMOGRAPHIC BLOCK
# ============================================================

def build_demographic_block(
    row: pd.Series,
):
    """
    Build age and sex information for a subject.
    """

    age = format_value(
        row[AGE_COLUMN],
        decimal_places=1,
    )

    sex = str(
        row[SEX_COLUMN]
    ).strip().upper()

    if sex == "M":
        gender = "Male"

    elif sex == "F":
        gender = "Female"

    else:
        gender = str(
            row[SEX_COLUMN]
        ).strip()

    return (
        f"Age: {age}, "
        f"Gender: {gender}"
    )

# ============================================================
# MRI FEATURE COLUMNS
# ============================================================

def get_metric_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Identify MRI feature columns common to train and test data.
    """

    excluded_columns = {
        SUBJECT_COLUMN,
        GROUP_COLUMN,
        AGE_COLUMN,
        "Sex",
        "Gender",
    }

    metric_columns = [
        column
        for column in train_df.columns
        if (
            column in test_df.columns
            and column not in excluded_columns
        )
    ]

    if not metric_columns:
        raise ValueError(
            "No MRI measurement columns were found."
        )

    return metric_columns

# ============================================================
# MRI METRIC BLOCK
# ============================================================

def build_metric_block(
    row: pd.Series,
    metric_columns: list[str],
):
    """
    Format MRI metrics for inclusion in the prompt.

    Volume and cortical-thickness features belonging to the same
    region are presented together.
    """

    volume_suffix = " Volume"
    thickness_suffix = " Cortical Thickness"

    volume_columns = {}
    thickness_columns = {}
    standalone_columns = []

    # --------------------------------------------------------
    # Group regional measurements
    # --------------------------------------------------------

    for column in metric_columns:

        if column.endswith(
            volume_suffix
        ):

            region_name = (
                column.removesuffix(
                    volume_suffix
                )
            )

            volume_columns[
                region_name
            ] = column

        elif column.endswith(
            thickness_suffix
        ):

            region_name = (
                column.removesuffix(
                    thickness_suffix
                )
            )

            thickness_columns[
                region_name
            ] = column

        else:
            standalone_columns.append(
                column
            )

    lines = []

    # --------------------------------------------------------
    # Standalone MRI measurements
    # --------------------------------------------------------

    for column in standalone_columns:

        value = format_value(
            row[column]
        )

        lines.append(
            f"{column}: {value}"
        )

    # --------------------------------------------------------
    # Regional volume/thickness measurements
    # --------------------------------------------------------

    processed_regions = set()

    for column in metric_columns:

        if column.endswith(
            volume_suffix
        ):

            region_name = (
                column.removesuffix(
                    volume_suffix
                )
            )

        elif column.endswith(
            thickness_suffix
        ):

            region_name = (
                column.removesuffix(
                    thickness_suffix
                )
            )

        else:
            continue

        if region_name in processed_regions:
            continue

        processed_regions.add(
            region_name
        )

        values = []

        volume_column = (
            volume_columns.get(
                region_name
            )
        )

        if volume_column is not None:

            volume_value = (
                format_value(
                    row[
                        volume_column
                    ]
                )
            )

            values.append(
                f"{volume_value} mm³"
            )

        thickness_column = (
            thickness_columns.get(
                region_name
            )
        )

        if thickness_column is not None:

            thickness_value = (
                format_value(
                    row[
                        thickness_column
                    ]
                )
            )

            values.append(
                f"{thickness_value} mm"
            )

        lines.append(
            f"{region_name}: "
            f"{', '.join(values)}"
        )

    return "\n".join(
        lines
    )

# ============================================================
# SUBJECT LOOKUP
# ============================================================

def get_subject_row(
    dataframe: pd.DataFrame,
    subject_id,
    dataset_name: str,
):
    """
    Retrieve exactly one subject row.
    """

    matches = dataframe[
        dataframe[
            SUBJECT_COLUMN
        ].astype(str)
        == str(subject_id)
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Subject '{subject_id}' "
            f"was not found in "
            f"{dataset_name}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Subject '{subject_id}' "
            f"appears multiple times in "
            f"{dataset_name}."
        )

    return matches.iloc[0]

# ============================================================
# DIAGNOSIS TEXT
# ============================================================

def diagnosis_text(
    label,
):
    """
    Convert dataset labels into the text shown to the LLM.
    """

    label = (
        str(label)
        .strip()
        .upper()
    )

    if label == "AD":
        return "Alzheimer's disease"

    if label == "CN":
        return "Non-Alzheimer's"

    raise ValueError(
        f"Unexpected diagnosis label: "
        f"{label}"
    )

# ============================================================
# BUILD ONE EXAMPLE
# ============================================================

def build_example_block(
    row: pd.Series,
    expected_label: str,
    example_number: int,
    metric_columns: list[str],
):
    """
    Build one in-context learning demonstration.
    """

    actual_label = (
        str(
            row[GROUP_COLUMN]
        )
        .strip()
        .upper()
    )

    if (
        actual_label
        != expected_label
    ):
        raise ValueError(
            f"Selected example "
            f"{row[SUBJECT_COLUMN]} "
            f"was expected to be "
            f"{expected_label}, "
            f"but train.csv contains "
            f"{actual_label}."
        )

    demographic_block = (
        build_demographic_block(
            row
        )
    )

    metric_block = (
        build_metric_block(
            row=row,
            metric_columns=metric_columns,
        )
    )

    return "\n".join([
        f"Example {example_number}:",
        demographic_block,
        f"{cfg.INFORMATION}:",
        metric_block,
        (
            "Diagnosis: "
            f"{diagnosis_text(actual_label)}"
        ),
    ])

# ============================================================
# SHOT COLUMNS
# ============================================================

def get_shot_columns(
    selections_df: pd.DataFrame,
):
    """
    Identify AD1, AD2, ... and CN1, CN2, ... columns.
    """

    ad_columns = sorted(
        [
            column
            for column in selections_df.columns
            if (
                column.startswith("AD")
                and column[2:].isdigit()
            )
        ],
        key=lambda column: int(
            column[2:]
        ),
    )

    cn_columns = sorted(
        [
            column
            for column in selections_df.columns
            if (
                column.startswith("CN")
                and column[2:].isdigit()
            )
        ],
        key=lambda column: int(
            column[2:]
        ),
    )

    if not ad_columns:
        raise ValueError(
            "No AD example columns were found."
        )

    if not cn_columns:
        raise ValueError(
            "No CN example columns were found."
        )

    if (
        len(ad_columns)
        != len(cn_columns)
    ):
        raise ValueError(
            "The number of AD and CN "
            "example columns must match."
        )

    expected_total = (
        len(ad_columns)
        + len(cn_columns)
    )

    if expected_total != cfg.N_SHOT:
        raise ValueError(
            f"Selection CSV contains "
            f"{expected_total} demonstrations, "
            f"but Settings.N_SHOT is "
            f"{cfg.N_SHOT}."
        )

    return (
        ad_columns,
        cn_columns,
    )

# ============================================================
# BUILD ONE FULL PROMPT
# ============================================================

def build_prompt(
    selection_row: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metric_columns: list[str],
    ad_columns: list[str],
    cn_columns: list[str],
):
    """
    Build the full LLM prompt for one test subject.
    """

    test_subject = str(
        selection_row[
            "Test_Subject"
        ]
    )

    test_row = get_subject_row(
        dataframe=test_df,
        subject_id=test_subject,
        dataset_name="test.csv",
    )

    example_blocks = []

    example_number = 1

    # --------------------------------------------------------
    # Alternate AD and CN demonstrations
    # --------------------------------------------------------

    for (
        ad_column,
        cn_column,
    ) in zip(
        ad_columns,
        cn_columns,
    ):

        # --------------------------------------------------------
        # AD demonstration
        # --------------------------------------------------------

        ad_subject = (
            selection_row[
                ad_column
            ]
        )

        ad_row = get_subject_row(
            dataframe=train_df,
            subject_id=ad_subject,
            dataset_name="train.csv",
        )

        example_blocks.append(
            build_example_block(
                row=ad_row,
                expected_label="AD",
                example_number=(
                    example_number
                ),
                metric_columns=(
                    metric_columns
                ),
            )
        )

        example_number += 1

        # --------------------------------------------------------
        # CN demonstration
        # --------------------------------------------------------

        cn_subject = (
            selection_row[
                cn_column
            ]
        )

        cn_row = get_subject_row(
            dataframe=train_df,
            subject_id=cn_subject,
            dataset_name="train.csv",
        )

        example_blocks.append(
            build_example_block(
                row=cn_row,
                expected_label="CN",
                example_number=(
                    example_number
                ),
                metric_columns=(
                    metric_columns
                ),
            )
        )

        example_number += 1

    examples_block = (
        "\n\n".join(
            example_blocks
        )
    )

    demographic_block = (
        build_demographic_block(
            test_row
        )
    )

    metric_block = (
        build_metric_block(
            row=test_row,
            metric_columns=metric_columns,
        )
    )

    # --------------------------------------------------------
    # Prompt templates
    # --------------------------------------------------------

    system_content = (
        INST_SYS_CTX.format(
            neuro=cfg.NEURO
        ).strip()
    )

    user_content = (
        INST_USR_REASONING.format(
            neuro=cfg.NEURO,
            information=cfg.INFORMATION,
            examples_block=(
                examples_block
            ),
            demographic_block=(
                demographic_block
            ),
            metric_block=(
                metric_block
            ),
        ).strip()
    )

    return (
        system_content
        + "\n\n"
        + user_content
    )

# ============================================================
# BUILD ONE ZERO-SHOT PROMPT
# ============================================================

def build_zero_shot_prompt(
    test_row: pd.Series,
    metric_columns: list[str],
):
    """
    Build the final zero-shot LLM prompt for one test subject.

    No in-context demonstrations are included.
    """

    demographic_block = (
        build_demographic_block(
            test_row
        )
    )

    metric_block = (
        build_metric_block(
            row=test_row,
            metric_columns=metric_columns,
        )
    )

    system_content = (
        INST_SYS_CTX.format(
            neuro=cfg.NEURO
        ).strip()
    )

    user_content = (
        INST_USR_REASONING_ZERO_SHOT.format(
            neuro=cfg.NEURO,
            information=cfg.INFORMATION,
            demographic_block=(
                demographic_block
            ),
            metric_block=(
                metric_block
            ),
        ).strip()
    )

    return (
        system_content
        + "\n\n"
        + user_content
    )

# ============================================================
# PROMPT HASH
# ============================================================

def get_prompt_hash(
    prompt: str,
):
    """
    Create a SHA-256 hash of the exact prompt.

    The prompt itself is not saved, but the hash provides a
    reproducibility check that the same prompt was used.
    """

    return hashlib.sha256(
        prompt.encode(
            "utf-8"
        )
    ).hexdigest()

# ============================================================
# BUILD PROMPTS IN MEMORY
# ============================================================

def build_prompt_data(
    selections_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metric_columns: list[str],
):
    """
    Build all prompts in memory without writing an intermediate
    prompt CSV.
    """

    if (
        "Test_Subject"
        not in selections_df.columns
    ):
        raise ValueError(
            "Example-selection file does "
            "not contain 'Test_Subject'."
        )

    (
        ad_columns,
        cn_columns,
    ) = get_shot_columns(
        selections_df
    )

    prompt_rows = []

    for _, selection_row in (
        selections_df.iterrows()
    ):

        subject = str(
            selection_row[
                "Test_Subject"
            ]
        )

        prompt = build_prompt(
            selection_row=(
                selection_row
            ),
            train_df=train_df,
            test_df=test_df,
            metric_columns=(
                metric_columns
            ),
            ad_columns=(
                ad_columns
            ),
            cn_columns=(
                cn_columns
            ),
        )

        prompt_rows.append({
            "Subject":
                subject,

            "Prompt_Text":
                prompt,

            "Prompt_Hash":
                get_prompt_hash(
                    prompt
                ),
        })

    return pd.DataFrame(
        prompt_rows
    )

# ============================================================
# BUILD ZERO-SHOT PROMPTS IN MEMORY
# ============================================================

def build_zero_shot_prompt_data(
    test_df: pd.DataFrame,
    metric_columns: list[str],
):
    """
    Build one zero-shot prompt for every test subject.

    No example-selection file is required.
    """

    prompt_rows = []

    for _, test_row in test_df.iterrows():

        subject = str(
            test_row[
                SUBJECT_COLUMN
            ]
        )

        prompt = (
            build_zero_shot_prompt(
                test_row=test_row,
                metric_columns=(
                    metric_columns
                ),
            )
        )

        prompt_rows.append({
            "Subject":
                subject,

            "Prompt_Text":
                prompt,

            "Prompt_Hash":
                get_prompt_hash(
                    prompt
                ),
        })

    return pd.DataFrame(
        prompt_rows
    )

# ============================================================
# CALL LLM
# ============================================================

def call_llm(
    llm,
    prompt: str,
):
    """
    Run one LLM request with retry handling.
    """

    last_error = None

    for attempt in range(
        1,
        cfg.RETRY_COUNT + 1,
    ):

        try:

            response = llm.invoke(
                prompt
            )

            answer = response_to_text(
                response
            )

            if not answer:
                raise ValueError(
                    "LLM returned an empty answer."
                )

            return answer

        except Exception as exc:

            last_error = exc

            print(
                f"\nLLM attempt "
                f"{attempt}/"
                f"{cfg.RETRY_COUNT} failed"
                f"\nError type: "
                f"{type(exc).__name__}"
                f"\nError: {exc}",
                flush=True,
            )

            if (
                attempt
                == cfg.RETRY_COUNT
            ):
                break

            time.sleep(
                cfg.RETRY_WAIT_SECONDS
            )

    raise RuntimeError(
        "LLM request failed after "
        f"{cfg.RETRY_COUNT} attempts."
    ) from last_error

# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results(
    output_file: Path,
):
    """
    Load an incomplete/complete run so execution can resume.
    """

    expected_columns = [
        "Subject",
        "Prompt_Hash",
        "Answer",
    ]

    if not output_file.exists():

        return pd.DataFrame(
            columns=expected_columns
        )

    existing = pd.read_csv(
        output_file
    )

    if (
        list(existing.columns)
        != expected_columns
    ):

        raise ValueError(
            "Existing result file has "
            "unexpected columns:\n"
            f"{output_file}\n"
            f"Expected: {expected_columns}\n"
            f"Found: "
            f"{existing.columns.tolist()}"
        )

    return existing

# ============================================================
# RUN ONE SUBJECT
# ============================================================

def run_subject(
    llm,
    subject,
    prompt,
    prompt_hash,
):
    """
    Run one test subject.
    """

    try:

        answer = call_llm(
            llm=llm,
            prompt=prompt,
        )

    except Exception as exc:

        answer = (
            f"ERROR: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return {
        "Subject":
            subject,

        "Prompt_Hash":
            prompt_hash,

        "Answer":
            answer,
    }

# ============================================================
# RUN ONE REPEAT
# ============================================================

def run_method_once(
    llm,
    prompt_data: pd.DataFrame,
    method_name: str,
    run_number: int,
    output_file: Path,
):
    """
    Run all test subjects for one experimental repeat.
    """

    existing_results = (
        load_existing_results(
            output_file
        )
    )

    completed_subjects = set(
        existing_results[
            "Subject"
        ].astype(str)
    )

    # --------------------------------------------------------
    # Validate hashes when resuming
    # --------------------------------------------------------

    if not existing_results.empty:

        current_hash_lookup = dict(
            zip(
                prompt_data[
                    "Subject"
                ].astype(str),

                prompt_data[
                    "Prompt_Hash"
                ].astype(str),
            )
        )

        for _, row in (
            existing_results.iterrows()
        ):

            subject = str(
                row["Subject"]
            )

            if (
                subject
                not in current_hash_lookup
            ):
                raise ValueError(
                    f"Existing output contains "
                    f"unexpected subject: "
                    f"{subject}"
                )

            if (
                str(
                    row[
                        "Prompt_Hash"
                    ]
                )
                != current_hash_lookup[
                    subject
                ]
            ):
                raise ValueError(
                    f"Prompt changed for "
                    f"subject {subject}. "
                    "Do not resume this run "
                    "with a different prompt."
                )

    result_rows = (
        existing_results
        .to_dict(
            orient="records"
        )
    )

    remaining_rows = []

    for _, row in (
        prompt_data.iterrows()
    ):

        subject = str(
            row["Subject"]
        )

        if (
            subject
            in completed_subjects
        ):
            continue

        remaining_rows.append({
            "Subject":
                subject,

            "Prompt_Text":
                str(
                    row[
                        "Prompt_Text"
                    ]
                ),

            "Prompt_Hash":
                str(
                    row[
                        "Prompt_Hash"
                    ]
                ),
        })

    if not remaining_rows:

        print(
            f"{method_name} "
            f"run {run_number} "
            "already complete."
        )

        return

    # --------------------------------------------------------
    # Parallel LLM execution
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=(
            cfg.LLM_PARALLEL_CALLS
        )
    ) as executor:

        future_to_subject = {}

        for row in remaining_rows:

            future = executor.submit(
                run_subject,
                llm,
                row["Subject"],
                row["Prompt_Text"],
                row["Prompt_Hash"],
            )

            future_to_subject[
                future
            ] = row["Subject"]

        with tqdm(
            total=len(
                remaining_rows
            ),
            desc=(
                f"{method_name} "
                f"run {run_number}"
            ),
        ) as progress_bar:

            for future in (
                as_completed(
                    future_to_subject
                )
            ):

                subject = (
                    future_to_subject[
                        future
                    ]
                )

                try:

                    result = (
                        future.result()
                    )

                except Exception as exc:

                    result = {
                        "Subject":
                            subject,

                        "Prompt_Hash":
                            "",

                        "Answer":
                            (
                                f"ERROR: "
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                    }

                result_rows.append(
                    result
                )

                # --------------------------------------------------------
                # Checkpoint after every subject
                # --------------------------------------------------------

                checkpoint_df = (
                    pd.DataFrame(
                        result_rows,
                        columns=[
                            "Subject",
                            "Prompt_Hash",
                            "Answer",
                        ],
                    )
                )

                checkpoint_df.to_csv(
                    output_file,
                    index=False,
                )

                progress_bar.update(
                    1
                )

    # --------------------------------------------------------
    # Restore original subject order
    # --------------------------------------------------------

    final_df = pd.DataFrame(
        result_rows,
        columns=[
            "Subject",
            "Prompt_Hash",
            "Answer",
        ],
    )

    subject_order = {
        str(subject): index
        for index, subject
        in enumerate(
            prompt_data[
                "Subject"
            ].astype(str)
        )
    }

    final_df[
        "_sort_order"
    ] = (
        final_df[
            "Subject"
        ]
        .astype(str)
        .map(
            subject_order
        )
    )

    final_df = (
        final_df
        .sort_values(
            "_sort_order"
        )
        .drop(
            columns=[
                "_sort_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    final_df.to_csv(
        output_file,
        index=False,
    )

# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "LLM CLASSIFICATION"
    )

    if cfg.N_SHOT == 0:

        print(
            "Method: Zero-shot"
        )

    else:

        print(
            f"Method: "
            f"{cfg.SELECTION_METHOD}"
        )

    print(
        f"Model: {cfg.LLM_MODEL}"
    )

    print(
        f"N-shot: {cfg.N_SHOT}"
    )

    print(
        f"Repeated runs: "
        f"{cfg.LLM_N_REPEATS}"
    )

    # --------------------------------------------------------
    # Load train/test datasets
    # --------------------------------------------------------

    train_df = pd.read_csv(
        cfg.DATA_SPLIT_DIR / "train.csv"
    )

    test_df = pd.read_csv(
        cfg.DATA_SPLIT_DIR / "test.csv"
    )

    # --------------------------------------------------------
    # MRI feature columns
    # --------------------------------------------------------

    metric_columns = (
        get_metric_columns(
            train_df=train_df,
            test_df=test_df,
        )
    )

    print(
        f"MRI features used in prompts: "
        f"{len(metric_columns)}"
    )



    # --------------------------------------------------------
    # Build prompts in memory
    # --------------------------------------------------------

    if cfg.N_SHOT == 0:

        print(
            "Prompting mode: zero-shot"
        )

        prompt_data = (
            build_zero_shot_prompt_data(
                test_df=test_df,
                metric_columns=(
                    metric_columns
                ),
            )
        )

        experiment_name = (
            "zero_shot"
        )

    else:

        print(
            f"Prompting mode: "
            f"{cfg.N_SHOT}-shot"
        )

        # ----------------------------------------------------
        # Load selected examples
        # ----------------------------------------------------

        selection_file = (
            cfg.EXAMPLE_SELECTION_OUTPUT_ROOT
            / f"{cfg.SELECTION_METHOD}.csv"
        )

        if not selection_file.exists():

            raise FileNotFoundError(
                "Example-selection file "
                "does not exist:\n"
                f"{selection_file}\n\n"
                "Run 05_select_examples.py "
                "for this method first."
            )

        selections_df = pd.read_csv(
            selection_file
        )

        # ----------------------------------------------------
        # Build few-shot prompts
        # ----------------------------------------------------

        prompt_data = build_prompt_data(
            selections_df=(
                selections_df
            ),
            train_df=train_df,
            test_df=test_df,
            metric_columns=(
                metric_columns
            ),
        )

        experiment_name = (
            cfg.SELECTION_METHOD
        )


    print(
        f"Prompts created in memory: "
        f"{len(prompt_data)}"
    )

    # --------------------------------------------------------
    # Create LLM client
    # --------------------------------------------------------

    llm = create_client(
        cfg.LLM_MODEL
    )

    # --------------------------------------------------------
    # Output structure
    # results/<model>/<method>/run_1.csv
    # --------------------------------------------------------

    method_output_directory = (
        cfg.LLM_RESULTS_ROOT
        / f"{cfg.N_SHOT}-shot"
        / cfg.LLM_MODEL
        / experiment_name
    )

    method_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Repeated runs
    # --------------------------------------------------------

    for run_number in range(
        1,
        cfg.LLM_N_REPEATS + 1,
    ):

        output_file = (
            method_output_directory
            / f"run_{run_number}.csv"
        )

        print(
            f"\nStarting "
            f"{experiment_name} "
            f"run {run_number}/"
            f"{cfg.LLM_N_REPEATS}"
        )

        try:

            run_method_once(
                llm=llm,
                prompt_data=(
                    prompt_data
                ),
                method_name=(
                    experiment_name
                ),
                run_number=(
                    run_number
                ),
                output_file=(
                    output_file
                ),
            )

            # --------------------------------------------------------
            # Extract prediction and reasoning
            # --------------------------------------------------------
            
            process_file(
                str(output_file)
            )

        except Exception as exc:

            print(
                f"\nERROR running "
                f"{experiment_name} "
                f"run {run_number}:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "Continuing to next run..."
            )

    print(
        "ALL LLM RUNS COMPLETE"
    )


if __name__ == "__main__":
    main()