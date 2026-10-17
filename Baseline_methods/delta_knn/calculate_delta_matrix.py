import os
import re
import json
import time
import pickle
import numpy as np
import pathlib as Path

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.client import send_chat_request
from utils.prompts import INST_SYS_CTX, INST_USR_PROB_SHORT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import Settings as cfg

MODEL_NAME = cfg.LLM_MODEL

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.01
TOP_P = 0.9
SEED = 0

OPENAI_API_KEY = ""
OPENAI_API_BASE = ""

NEURO = cfg.NEURO

INFORMATION = (cfg.INFORMATION)

OUTPUT_FORMAT = """
Prediction: <AD or CN>
Probability: <probability assigned to the predicted class between 0 and 1>
""".strip()

def build_demographic_block(doc):
    """
    Build the demographic block for one subject.

    Example:
        Sex: F
    """

    lines = []

    if "sex" in doc:
        sex = str(doc["sex"]).strip()

        if (sex and sex.lower() != "nan"):
            lines.append(f"Sex: {sex}")

    if "age" in doc:
        age = doc["age"]

        if not np.isnan(age):
            lines.append(f"Age: {age}")

    return "\n".join(lines)


def build_metric_block(doc):
    """
    Convert all brain-region features for one subject
    into text.

    Example:
        Left Hippocampus: 3124.0
        Right Hippocampus: 3290.0
        ...
    """

    if "feature_dict" not in doc:
        raise ValueError(
            f"Subject {doc.get('name')} does not contain "
            "'feature_dict'."
        )

    lines = []

    for feature_name, value in (
        doc["feature_dict"].items()
    ):
        lines.append(
            f"{feature_name}: {value}"
        )

    return "\n".join(lines)


# ============================================================
# DEMONSTRATION EXAMPLE
# ============================================================

def build_example_block(doc):
    """
    Build one labelled training example.

    Example:

        Example:

        Sex: F

        brain region volumes and cortical thickness values:
        ...
        Diagnosis: AD
    """

    demographic_block = (
        build_demographic_block(doc)
    )

    metric_block = (
        build_metric_block(doc)
    )

    label = str(
        doc["gold"]
    ).strip().upper()

    if label not in ["AD", "CN"]:
        raise ValueError(
            f"Unexpected label '{label}' for "
            f"subject {doc.get('name')}."
        )

    parts = [
        "Example:"
    ]

    if demographic_block:
        parts.append(
            demographic_block
        )

    parts.append(
        f"{INFORMATION}:\n{metric_block}"
    )

    parts.append(
        f"Diagnosis: {label}"
    )

    return "\n\n".join(parts)


# ============================================================
# ZERO-SHOT PROMPT
# ============================================================

def build_zero_shot_messages(target_doc):
    """
    Build a zero-shot prompt.

    There is NO demonstration example.
    """

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_content = (
        INST_SYS_CTX.format(
            neuro=NEURO
        )
        .strip()
    )

    # --------------------------------------------------------
    # Target subject
    # --------------------------------------------------------

    demographic_block = (
        build_demographic_block(
            target_doc
        )
    )

    metric_block = (
        build_metric_block(
            target_doc
        )
    )

    user_content = (
        INST_USR_PROB_SHORT.format(
            neuro=NEURO,
            information=INFORMATION,
            demographic_block=
                demographic_block,
            metric_block=
                metric_block
        )
        .strip()
    )

    # Add explicit format after "Output:"
    user_content = (
        user_content
        + "\n"
        + OUTPUT_FORMAT
    )

    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    return messages


# ============================================================
# ONE-SHOT PROMPT
# ============================================================

def build_one_shot_messages(
    demo_doc,
    target_doc
):
    """
    Build a one-shot prompt.

    The demonstration is inserted after the system
    instruction and before the target subject.
    """

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_content = (
        INST_SYS_CTX.format(
            neuro=NEURO
        )
        .strip()
    )

    # --------------------------------------------------------
    # Add one labelled demonstration
    # --------------------------------------------------------

    example_block = (
        build_example_block(
            demo_doc
        )
    )

    system_content = (
        system_content
        + "\n\n"
        + example_block
    )

    # --------------------------------------------------------
    # Target subject
    # --------------------------------------------------------

    demographic_block = (
        build_demographic_block(
            target_doc
        )
    )

    metric_block = (
        build_metric_block(
            target_doc
        )
    )

    user_content = (
        INST_USR_PROB_SHORT.format(
            neuro=NEURO,
            information=INFORMATION,
            demographic_block=
                demographic_block,
            metric_block=
                metric_block
        )
        .strip()
    )

    user_content = (
        user_content
        + "\n"
        + OUTPUT_FORMAT
    )

    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    return messages


# ============================================================
# GPT-5.4 CALL
# ============================================================

def run_llm(
    messages,
    max_retries=5
):
    """
    Run GPT-5.4 through client.py.

    Basic retry handling is included because Delta
    construction requires many LLM calls.
    """

    last_error = None

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            response = send_chat_request(
                OPENAI_API_KEY,
                OPENAI_API_BASE,
                MODEL_NAME,
                messages,
                MAX_NEW_TOKENS,
                TEMPERATURE,
                TOP_P,
                SEED
            )

            return response

        except Exception as exc:

            last_error = exc

            if attempt == max_retries:
                break

            wait_seconds = min(
                2 ** attempt,
                30
            )

            print(
                f"\nGPT request failed "
                f"(attempt {attempt}/{max_retries})."
            )

            print(
                f"Error: {exc}"
            )

            print(
                f"Retrying in "
                f"{wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "GPT-5.4 request failed after "
        f"{max_retries} attempts."
    ) from last_error


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_prediction_probability(
    answer
):
    """
    Extract:

        Prediction: AD
        Probability: 0.82

    Also accepts:

        Prediction: Alzheimer's disease (AD)
        Probability: 82%

    Probability refers to the class predicted by the LLM.
    """

    if answer is None:
        raise ValueError(
            "LLM returned an empty response."
        )

    answer = str(
        answer
    ).strip()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_match = re.search(
        r"Prediction\s*:"
        r"[^\n]*?"
        r"\b(AD|CN)\b",
        answer,
        flags=re.IGNORECASE
    )

    if prediction_match is None:

        raise ValueError(
            "Could not extract Prediction "
            "from LLM output:\n\n"
            f"{answer}"
        )

    prediction = (
        prediction_match
        .group(1)
        .upper()
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_match = re.search(
        r"Probability\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*(%)?",
        answer,
        flags=re.IGNORECASE
    )

    if probability_match is None:

        raise ValueError(
            "Could not extract Probability "
            "from LLM output:\n\n"
            f"{answer}"
        )

    probability = float(
        probability_match.group(1)
    )

    percentage_symbol = (
        probability_match.group(2)
    )

    # Example:
    # 82% -> 0.82

    if percentage_symbol == "%":
        probability /= 100.0

    # Also tolerate:
    # Probability: 82

    elif probability > 1:
        probability /= 100.0

    if not (
        0.0
        <= probability
        <= 1.0
    ):

        raise ValueError(
            f"Invalid probability "
            f"{probability}.\n\n"
            f"Original response:\n{answer}"
        )

    return (
        prediction,
        probability
    )


# ============================================================
# TRUE-CLASS PROBABILITY
# ============================================================

def get_correct_class_probability(
    true_label,
    prediction,
    probability
):
    """
    Delta-KNN requires the probability assigned to the
    TRUE class.

    However, our prompt requests the probability assigned
    to the PREDICTED class.

    Since AD vs CN is binary:

        if prediction == true label:
            P(true class) = probability

        otherwise:
            P(true class) = 1 - probability

    Example:

        true label      = AD
        prediction      = CN
        probability(CN) = 0.80

    Therefore:

        probability(AD) = 0.20
    """

    true_label = (
        str(true_label)
        .strip()
        .upper()
    )

    prediction = (
        str(prediction)
        .strip()
        .upper()
    )

    if true_label not in [
        "AD",
        "CN"
    ]:
        raise ValueError(
            f"Unexpected true label: "
            f"{true_label}"
        )

    if prediction not in [
        "AD",
        "CN"
    ]:
        raise ValueError(
            f"Unexpected prediction: "
            f"{prediction}"
        )

    if prediction == true_label:
        return float(
            probability
        )

    return float(
        1.0 - probability
    )


# ============================================================
# VALIDATE TRAINING SUBJECTS
# ============================================================

def validate_train_docs(
    train_docs
):
    """
    Basic validation before performing expensive GPT calls.
    """

    if not train_docs:
        raise ValueError(
            "train_docs is empty."
        )

    names = []

    for doc in train_docs:

        for required_key in [
            "name",
            "gold",
            "feature_dict"
        ]:

            if required_key not in doc:
                raise ValueError(
                    f"Training subject is missing "
                    f"'{required_key}':\n"
                    f"{doc}"
                )

        name = str(
            doc["name"]
        ).strip()

        label = str(
            doc["gold"]
        ).strip().upper()

        if label not in [
            "AD",
            "CN"
        ]:
            raise ValueError(
                f"Unexpected label '{label}' "
                f"for subject {name}."
            )

        names.append(
            name
        )

    if len(names) != len(
        set(names)
    ):

        raise ValueError(
            "Duplicate subject IDs found "
            "in train_docs."
        )


# ============================================================
# SAVE JSON HELPER
# ============================================================

def save_json(
    path,
    data
):
    """
    Safely save JSON.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

def process_one_shot_pair(
    demo_idx,
    demo_doc,
    target_idx,
    target_doc,
    zero_score,
    max_output_retries=5
):
    """
    Process one demo-target pair.

    If GPT returns an empty or unparseable response,
    retry the same pair.

    If all retries fail, raise the error so the main
    thread can log it and leave this Delta cell as NaN.
    """

    demo_name = str(
        demo_doc["name"]
    ).strip()

    target_name = str(
        target_doc["name"]
    ).strip()

    true_label = str(
        target_doc["gold"]
    ).strip().upper()

    messages = build_one_shot_messages(
        demo_doc=demo_doc,
        target_doc=target_doc
    )

    last_error = None

    for attempt in range(
        1,
        max_output_retries + 1
    ):

        try:

            # --------------------------------------------
            # Call GPT
            # --------------------------------------------

            answer = run_llm(
                messages
            )

            # --------------------------------------------
            # Reject empty responses
            # --------------------------------------------

            if (
                answer is None
                or not str(answer).strip()
            ):
                raise ValueError(
                    "LLM returned an empty response."
                )

            # --------------------------------------------
            # Parse prediction + probability
            # --------------------------------------------

            prediction, probability = (
                parse_prediction_probability(
                    answer
                )
            )

            # --------------------------------------------
            # Convert to true-class probability
            # --------------------------------------------

            one_score = (
                get_correct_class_probability(
                    true_label=true_label,
                    prediction=prediction,
                    probability=probability
                )
            )

            # --------------------------------------------
            # Delta
            # --------------------------------------------

            delta_score = (
                one_score
                -
                zero_score
            )

            return {
                "demo_idx": demo_idx,
                "target_idx": target_idx,
                "demo": demo_name,
                "target": target_name,
                "true_label": true_label,
                "prediction": prediction,
                "probability": probability,
                "zero_score": zero_score,
                "one_score": one_score,
                "delta": delta_score,
                "answer": answer
            }

        except ValueError as exc:

            last_error = exc

            print(
                f"\nInvalid LLM output "
                f"for Demo={demo_name}, "
                f"Target={target_name} "
                f"(attempt "
                f"{attempt}/{max_output_retries})"
            )

            print(
                f"Error: {exc}"
            )

            if attempt < max_output_retries:

                wait_seconds = min(
                    2 ** attempt,
                    20
                )

                print(
                    f"Retrying same pair in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        f"Could not obtain a valid response "
        f"after {max_output_retries} attempts "
        f"for Demo={demo_name}, "
        f"Target={target_name}"
    ) from last_error

# ============================================================
# BUILD ONE COMPLETE DELTA MATRIX
# ============================================================

def build_delta_matrix_once(
    train_docs,
    run_number,
    run_dir,
    checkpoint_every=100,
    max_workers=4
):
    """
    Construct one complete Delta Matrix.

    Matrix orientation:

        ROW    = candidate demonstration
        COLUMN = target training subject

    Therefore:

        delta_matrix[i, j]

    represents the effect of using training subject i
    as a demonstration when classifying training subject j.

    Delta:

        delta(i,j)
            =
        P(true label_j | demo_i, target_j)
            -
        P(true label_j | target_j)

    Zero-shot calls are processed sequentially.

    Exhaustive one-shot calls are processed in parallel
    using ThreadPoolExecutor.
    """

    # ========================================================
    # VALIDATE TRAINING DATA
    # ========================================================

    validate_train_docs(
        train_docs
    )

    n_subjects = len(
        train_docs
    )

    doc_index = [
        str(doc["name"]).strip()
        for doc in train_docs
    ]

    print(
        f"\nTraining subjects: "
        f"{n_subjects}"
    )

    print(
        f"Parallel workers: "
        f"{max_workers}"
    )

    # ========================================================
    # FILE PATHS FOR THIS RUN
    # ========================================================

    zero_file = os.path.join(
        run_dir,
        f"zero_scores_run_{run_number}.json"
    )

    delta_checkpoint_file = os.path.join(
        run_dir,
        f"delta_checkpoint_run_{run_number}.npy"
    )

    raw_output_file = os.path.join(
        run_dir,
        f"raw_outputs_run_{run_number}.jsonl"
    )

    # ========================================================
    # ZERO-SHOT SCORES
    # ========================================================

    if os.path.exists(
        zero_file
    ):

        print(
            f"\nLoading existing zero-shot "
            f"scores for run {run_number}"
        )

        with open(
            zero_file,
            "r",
            encoding="utf-8"
        ) as f:

            zero_scores = (
                json.load(f)
            )

    else:

        zero_scores = {}

    # ========================================================
    # FIND ZERO-SHOT SUBJECTS NOT YET COMPLETED
    # ========================================================

    remaining_zero = [
        doc
        for doc in train_docs
        if str(doc["name"]).strip()
        not in zero_scores
    ]

    # ========================================================
    # ZERO-SHOT CALLS
    # ========================================================

    if remaining_zero:

        print(
            f"\nRunning zero-shot predictions "
            f"for run {run_number}..."
        )

        for target_doc in tqdm(
            remaining_zero,
            desc=(
                f"Zero-shot "
                f"run {run_number}"
            )
        ):

            target_name = str(
                target_doc["name"]
            ).strip()

            true_label = str(
                target_doc["gold"]
            ).strip().upper()

            # ------------------------------------------------
            # Build prompt
            # ------------------------------------------------

            messages = (
                build_zero_shot_messages(
                    target_doc
                )
            )

            # ------------------------------------------------
            # GPT call
            # ------------------------------------------------

            answer = run_llm(
                messages
            )

            # ------------------------------------------------
            # Parse result
            # ------------------------------------------------

            (
                prediction,
                probability
            ) = (
                parse_prediction_probability(
                    answer
                )
            )

            # ------------------------------------------------
            # Probability assigned to true class
            # ------------------------------------------------

            correct_probability = (
                get_correct_class_probability(
                    true_label=
                        true_label,
                    prediction=
                        prediction,
                    probability=
                        probability
                )
            )

            # ------------------------------------------------
            # Store zero-shot score
            # ------------------------------------------------

            zero_scores[
                target_name
            ] = float(
                correct_probability
            )

            # ------------------------------------------------
            # Save immediately for resume support
            # ------------------------------------------------

            save_json(
                zero_file,
                zero_scores
            )

            # ------------------------------------------------
            # Save raw result
            # ------------------------------------------------

            with open(
                raw_output_file,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps({
                        "type":
                            "zero_shot",
                        "target":
                            target_name,
                        "demo":
                            None,
                        "true_label":
                            true_label,
                        "prediction":
                            prediction,
                        "probability":
                            probability,
                        "correct_class_probability":
                            correct_probability,
                        "answer":
                            answer
                    })
                    + "\n"
                )

    else:

        print(
            f"\nAll zero-shot predictions "
            f"already available for "
            f"run {run_number}."
        )

    # ========================================================
    # CHECK THAT ALL ZERO-SHOT SCORES EXIST
    # ========================================================

    missing_zero = [
        name
        for name in doc_index
        if name not in zero_scores
    ]

    if missing_zero:

        raise ValueError(
            "Missing zero-shot scores for:\n"
            f"{missing_zero}"
        )

    # ========================================================
    # INITIALISE / RESUME DELTA MATRIX
    # ========================================================

    if os.path.exists(
        delta_checkpoint_file
    ):

        print(
            f"\nLoading Delta checkpoint "
            f"for run {run_number}"
        )

        delta_matrix = np.load(
            delta_checkpoint_file
        )

        expected_shape = (
            n_subjects,
            n_subjects
        )

        if (
            delta_matrix.shape
            != expected_shape
        ):

            raise ValueError(
                "Stored Delta checkpoint "
                "has the wrong shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found: {delta_matrix.shape}"
            )

    else:

        print(
            f"\nCreating new "
            f"{n_subjects} x {n_subjects} "
            f"Delta Matrix"
        )

        delta_matrix = np.full(
            (
                n_subjects,
                n_subjects
            ),
            np.nan,
            dtype=float
        )

    # ========================================================
    # ONE-SHOT PROGRESS
    # ========================================================

    total_pairs = (
        n_subjects
        *
        (n_subjects - 1)
    )

    completed_pairs = int(
        np.sum(
            ~np.isnan(
                delta_matrix
            )
        )
    )

    remaining_pairs = (
        total_pairs
        -
        completed_pairs
    )

    print(
        f"\nExhaustive one-shot "
        f"run {run_number}"
    )

    print(
        f"Total required pairs: "
        f"{total_pairs}"
    )

    print(
        f"Already completed: "
        f"{completed_pairs}"
    )

    print(
        f"Remaining pairs: "
        f"{remaining_pairs}"
    )

    print(
        f"Using {max_workers} "
        f"parallel GPT workers"
    )

    progress = tqdm(
        total=total_pairs,
        initial=completed_pairs,
        desc=(
            f"One-shot "
            f"run {run_number}"
        )
    )

    calls_since_checkpoint = 0

    # ========================================================
    # PARALLEL EXHAUSTIVE ONE-SHOT
    # ========================================================

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        # ====================================================
        # ROW = DEMONSTRATION
        # ====================================================

        for demo_idx, demo_doc in enumerate(
            train_docs
        ):

            demo_name = str(
                demo_doc["name"]
            ).strip()

            futures = []

            # =================================================
            # COLUMN = TARGET
            # =================================================

            for target_idx, target_doc in enumerate(
                train_docs
            ):

                # ---------------------------------------------
                # Do not use subject as its own demonstration
                # ---------------------------------------------

                if demo_idx == target_idx:
                    continue

                # ---------------------------------------------
                # Resume support:
                # skip already completed Delta cells
                # ---------------------------------------------

                if not np.isnan(
                    delta_matrix[
                        demo_idx,
                        target_idx
                    ]
                ):
                    continue

                target_name = str(
                    target_doc["name"]
                ).strip()

                zero_score = float(
                    zero_scores[
                        target_name
                    ]
                )

                # ---------------------------------------------
                # Submit one GPT request
                # ---------------------------------------------

                future = executor.submit(
                    process_one_shot_pair,
                    demo_idx,
                    demo_doc,
                    target_idx,
                    target_doc,
                    zero_score
                )

                futures.append(
                    future
                )

            # =================================================
            # IF THIS DEMONSTRATION IS ALREADY COMPLETE
            # =================================================

            if len(futures) == 0:

                continue

            # =================================================
            # PROCESS COMPLETED GPT CALLS
            # =================================================

            for future in as_completed(
                futures
            ):
                try:
                    result = (
                        future.result()
                    )

                except Exception as exc:
                    print("\nWARNING: One-shot GPT call failed.", exc)

                    continue

                result_demo_idx = (
                    result["demo_idx"]
                )

                result_target_idx = (
                    result["target_idx"]
                )

                # ---------------------------------------------
                # Update Delta Matrix
                #
                # MAIN THREAD ONLY
                # ---------------------------------------------

                delta_matrix[
                    result_demo_idx,
                    result_target_idx
                ] = float(
                    result["delta"]
                )

                # ---------------------------------------------
                # Save raw output
                #
                # MAIN THREAD ONLY
                # ---------------------------------------------

                with open(
                    raw_output_file,
                    "a",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        json.dumps({
                            "type":
                                "one_shot",
                            "demo":
                                result["demo"],
                            "target":
                                result["target"],
                            "true_label":
                                result[
                                    "true_label"
                                ],
                            "prediction":
                                result[
                                    "prediction"
                                ],
                            "probability":
                                result[
                                    "probability"
                                ],
                            "zero_score":
                                result[
                                    "zero_score"
                                ],
                            "one_score":
                                result[
                                    "one_score"
                                ],
                            "delta":
                                result[
                                    "delta"
                                ],
                            "answer":
                                result[
                                    "answer"
                                ]
                        })
                        + "\n"
                    )

                # ---------------------------------------------
                # Update progress
                # ---------------------------------------------

                progress.update(
                    1
                )

                calls_since_checkpoint += 1

                # ---------------------------------------------
                # Periodic checkpoint
                # ---------------------------------------------

                if (
                    calls_since_checkpoint
                    >= checkpoint_every
                ):

                    np.save(
                        delta_checkpoint_file,
                        delta_matrix
                    )

                    calls_since_checkpoint = 0

            # =================================================
            # SAVE AFTER EACH DEMONSTRATION ROW
            # =================================================

            np.save(
                delta_checkpoint_file,
                delta_matrix
            )

    # ========================================================
    # CLOSE PROGRESS BAR
    # ========================================================

    progress.close()

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    np.save(
        delta_checkpoint_file,
        delta_matrix
    )

    # ========================================================
    # VALIDATE MATRIX COMPLETION
    # ========================================================

    final_completed = int(
        np.sum(
            ~np.isnan(
                delta_matrix
            )
        )
    )

    if (
        final_completed
        != total_pairs
    ):

        raise ValueError(
            "Delta Matrix is incomplete.\n"
            f"Expected pairs: "
            f"{total_pairs}\n"
            f"Completed pairs: "
            f"{final_completed}"
        )

    # ========================================================
    # COMPLETED
    # ========================================================

    print(
        f"\nDelta run "
        f"{run_number} completed."
    )

    print(
        f"Completed one-shot pairs: "
        f"{final_completed}"
    )

    print(
        f"Checkpoint saved to:\n"
        f"{delta_checkpoint_file}"
    )

    return (
        delta_matrix,
        doc_index
    )

# ============================================================
# BUILD / LOAD FINAL DELTA MATRIX
# ============================================================

def store_delta_matrix_and_index(
    dataset,
    model,
    train_docs,
    n_repeats=3,
    replace=False
):
    """
    Build or load the final Delta Matrix.

    Parameters
    ----------
    dataset : str
        Dataset identifier, e.g. "adni-train".

    model : str
        Model name used for output-directory naming.

    train_docs : list
        Training subjects created by extract_example().

    n_repeats : int
        Number of complete Delta Matrix runs.

    replace : bool
        False:
            Load the final matrix if it already exists.
            If it does not exist, build/resume it.

        True:
            Delete previous Delta Matrix files and
            rebuild everything from the beginning.

    Returns
    -------
    None, delta_matrix, doc_index
    """

    validate_train_docs(
        train_docs
    )

    if n_repeats <= 0:
        raise ValueError(
            "n_repeats must be greater than 0."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    safe_model_name = (
        str(model)
        .replace("/", "_")
        .replace("\\", "_")
    )

    delta_root = (
        Path(__file__).resolve().parent
        / "data"
        / "doc_delta"
    )

    delta_dir = (
        delta_root
        / dataset
        / safe_model_name
        / "delta-1"
    )

    delta_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.makedirs(
        delta_dir,
        exist_ok=True
    )

    final_delta_file = os.path.join(
        delta_dir,
        "delta_mri.npy"
    )

    final_index_file = os.path.join(
        delta_dir,
        "doc_index_mri.pkl"
    )

    # ========================================================
    # REBUILD FROM SCRATCH IF REQUESTED
    # ========================================================

    if replace:

        print(
            "\nreplace=True:"
            "\nRemoving previous Delta files "
            "and starting from scratch."
        )

        files_to_remove = [
            final_delta_file,
            final_index_file
        ]

        for run_number in range(
            1,
            n_repeats + 1
        ):

            files_to_remove.extend([
                os.path.join(
                    delta_dir,
                    f"zero_scores_run_"
                    f"{run_number}.json"
                ),
                os.path.join(
                    delta_dir,
                    f"delta_checkpoint_run_"
                    f"{run_number}.npy"
                ),
                os.path.join(
                    delta_dir,
                    f"raw_outputs_run_"
                    f"{run_number}.jsonl"
                )
            ])

        for path in files_to_remove:

            if os.path.exists(
                path
            ):
                os.remove(
                    path
                )

    # ========================================================
    # LOAD EXISTING FINAL MATRIX
    # ========================================================

    if (
        not replace
        and os.path.exists(
            final_delta_file
        )
        and os.path.exists(
            final_index_file
        )
    ):

        print(
            "\nLoading existing final "
            "Delta Matrix:"
        )

        print(
            final_delta_file
        )

        delta_mat = np.load(
            final_delta_file
        )

        with open(
            final_index_file,
            "rb"
        ) as f:

            doc_index = pickle.load(
                f
            )

        return (
            None,
            delta_mat,
            doc_index
        )

    # ========================================================
    # BUILD / RESUME EACH DELTA RUN
    # ========================================================

    delta_matrices = []
    doc_indices = []

    for run_number in range(
        1,
        n_repeats + 1
    ):

        print(
            "\n"
            "================================================"
        )

        print(
            f"DELTA MATRIX RUN "
            f"{run_number}/{n_repeats}"
        )

        print(
            "================================================"
        )

        (
            delta_mat,
            doc_index
        ) = (
            build_delta_matrix_once(
                train_docs=
                    train_docs,
                run_number=
                    run_number,
                run_dir=
                    delta_dir,
                checkpoint_every=
                    100
            )
        )

        delta_matrices.append(
            delta_mat
        )

        doc_indices.append(
            doc_index
        )

    # ========================================================
    # VERIFY SUBJECT ORDER
    # ========================================================

    first_index = (
        doc_indices[0]
    )

    for current_index in (
        doc_indices[1:]
    ):

        if (
            current_index
            != first_index
        ):

            raise ValueError(
                "Subject ordering differs "
                "between Delta Matrix runs."
            )

    # ========================================================
    # AVERAGE DELTA MATRICES
    # ========================================================

    stacked = np.stack(
        delta_matrices,
        axis=0
    )

    # Avoid warnings from the all-NaN diagonal.

    valid_counts = np.sum(
        ~np.isnan(stacked),
        axis=0
    )

    summed = np.nansum(
        stacked,
        axis=0
    )

    averaged_delta = np.full(
        summed.shape,
        np.nan,
        dtype=float
    )

    np.divide(
        summed,
        valid_counts,
        out=averaged_delta,
        where=valid_counts > 0
    )

    # A subject is never used as its own demonstration.
    np.fill_diagonal(
        averaged_delta,
        np.nan
    )

    # ========================================================
    # SAVE FINAL MATRIX
    # ========================================================

    np.save(
        final_delta_file,
        averaged_delta
    )

    with open(
        final_index_file,
        "wb"
    ) as f:

        pickle.dump(
            first_index,
            f
        )

    print(
        "\nFinal averaged Delta Matrix saved:"
    )

    print(
        final_delta_file
    )

    print(
        f"Shape: "
        f"{averaged_delta.shape}"
    )

    return (
        None,
        averaged_delta,
        first_index
    )