from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

import Settings as cfg


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY_GPT4")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_GPT4")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4")


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
Extract the reasoning into short atomic claims.

For each claim, return:

1. Claim
   - A short sub-sentence containing one main reasoning statement.

2. Claim Type
   - Assign exactly one of the following:

     Numerical:
     Reports a specific numerical measurement.

     Descriptive:
     Describes the state of a brain region or measurement,
     such as reduced volume, preserved thickness, enlargement,
     or symmetry.

     Comparative:
     Explicitly compares the test subject with another subject,
     AD/CN examples, or a stated reference.

     Interpretive:
     Draws a diagnostic, etiological, or overall pattern-level
     conclusion from the evidence.

   - If a sentence contains more than one claim type, split it
     into separate atomic claims whenever possible.
   - Do not infer comparisons that are not explicitly stated.
   - If a claim compares diagnostic interpretations
     (e.g. "more consistent with AD than non-AD"), classify it
     as Interpretive rather than Comparative.

3. Region
   - Brain region, or demographic variable mentioned in the claim.
   - If none is explicitly mentioned, return null.

4. Value
   - Any numerical value explicitly mentioned in the claim.
   - If none is mentioned, return null.

5. Direction
   - Extract the descriptive wording used in the claim, such as:
     "enlarged", "reduced", "preserved", "relatively preserved",
     "symmetric", "thinning", "closer to AD", "closer to non-AD",
     "AD-like", or "age-related/global atrophy".
   - Preserve the wording from the original reasoning as closely
     as possible.
   - Do not judge whether the direction is correct.

Rules:
- Split numerical measurements and descriptive interpretations
  into separate claims.
- Split left and right numerical values.
- Split volume and cortical thickness measurements.
- Keep joint descriptions and explicit comparisons together.
- If a statement describes a region and draws a diagnostic
  conclusion, separate those into Descriptive and Interpretive claims.
- Assign exactly one Claim Type to each claim.
- Do not invent, evaluate, or correct claims.
- Do not use outside knowledge.

Example:

Input:
"The left entorhinal cortex is markedly reduced
(722 mm³, 1.772 mm)."

Output:

| Claim | Claim Type | Region | Value | Direction |
|---|---|---|---|---|
| Left entorhinal cortex volume is 722 mm³. | Numerical | left entorhinal cortex | 722 mm³ | null |
| Left entorhinal cortex thickness is 1.772 mm. | Numerical | left entorhinal cortex | 1.772 mm | null |
| Left entorhinal cortex is markedly reduced. | Descriptive | left entorhinal cortex | null | markedly reduced |

Return a table with columns:

Claim | Claim Type | Region | Value | Direction

Reasoning:

{TEXT}
"""


# ============================================================
# CLIENT
# ============================================================

def create_client() -> AzureOpenAI:
    if (
        not AZURE_OPENAI_KEY
        or not AZURE_OPENAI_ENDPOINT
        or not AZURE_OPENAI_DEPLOYMENT
    ):
        raise ValueError(
            "One or more Azure OpenAI environment variables "
            "for GPT-4.1 are missing."
        )

    return AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2025-04-01-preview",
    )


# ============================================================
# SINGLE EXTRACTION CALL
# ============================================================

def extract_claims_from_reason(
    reason_text: str,
    client: AzureOpenAI,
) -> str:

    prompt = PROMPT_TEMPLATE.replace(
        "{TEXT}",
        reason_text,
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if content is None or not content.strip():
        raise ValueError(
            "Claim extraction returned an empty response."
        )

    return content.strip()


# ============================================================
# RETRY
# ============================================================

def extract_with_retry(
    reason_text: str,
    client: AzureOpenAI,
) -> str:

    last_error = None

    for attempt in range(
        1,
        cfg.RETRY_COUNT + 1,
    ):
        try:
            return extract_claims_from_reason(
                reason_text=reason_text,
                client=client,
            )

        except Exception as error:
            last_error = error

            if attempt == cfg.RETRY_COUNT:
                break

            time.sleep(
                cfg.RETRY_WAIT_SECONDS
            )

    raise RuntimeError(
        "Claim extraction failed "
        f"after {cfg.RETRY_COUNT} attempts."
    ) from last_error


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    row_index: int,
    subject: str,
    reason: str,
    client: AzureOpenAI,
) -> dict:

    try:
        extraction = extract_with_retry(
            reason_text=reason,
            client=client,
        )

    except Exception as error:
        extraction = (
            f"ERROR: {type(error).__name__}: {error}"
        )

    return {
        "row_index": row_index,
        "Subject": subject,
        "Brain_Region_Extraction": extraction,
    }


# ============================================================
# RESTORE PREVIOUS PROGRESS
# ============================================================

def restore_existing_progress(
    input_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:

    df = input_df.copy()

    if "Brain_Region_Extraction" not in df.columns:
        df["Brain_Region_Extraction"] = ""

    if not output_path.exists():
        return df

    previous_df = pd.read_csv(
        output_path
    )

    required_previous_columns = {
        "Subject",
        "Brain_Region_Extraction",
    }

    if not required_previous_columns.issubset(
        previous_df.columns
    ):
        print(
            "Existing output file has an unexpected structure. "
            "Starting extraction again:\n"
            f"{output_path}"
        )

        return df

    previous_lookup = {}

    for _, row in previous_df.iterrows():

        subject = str(
            row["Subject"]
        ).strip()

        extraction = row[
            "Brain_Region_Extraction"
        ]

        if pd.isna(extraction):
            continue

        extraction = str(
            extraction
        ).strip()

        if not extraction:
            continue

        if extraction.startswith(
            "ERROR:"
        ):
            continue

        previous_lookup[
            subject
        ] = extraction

    for row_index in range(
        len(df)
    ):
        subject = str(
            df.iloc[row_index]["Subject"]
        ).strip()

        if subject in previous_lookup:
            df.at[
                row_index,
                "Brain_Region_Extraction",
            ] = previous_lookup[subject]

    return df


# ============================================================
# PROCESS ONE RUN CSV
# ============================================================

def process_csv(
    csv_path: Path,
    output_dir: Path,
    client: AzureOpenAI,
) -> None:

    csv_path = Path(
        csv_path
    )

    output_path = (
        output_dir
        / csv_path.name
    )

    print(
        f"\nProcessing: {csv_path.name}"
    )

    input_df = pd.read_csv(
        csv_path
    )

    required_columns = {
        "Subject",
        "Reason",
    }

    missing_columns = (
        required_columns
        - set(input_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    input_df["Subject"] = (
        input_df["Subject"]
        .astype(str)
        .str.strip()
    )

    if (
        input_df["Subject"]
        .duplicated()
        .any()
    ):
        duplicates = (
            input_df.loc[
                input_df[
                    "Subject"
                ].duplicated(),
                "Subject",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate Subject values found in "
            f"{csv_path}:\n{duplicates}"
        )

    df = restore_existing_progress(
        input_df=input_df,
        output_path=output_path,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    remaining_rows = []

    for row_index in range(
        len(df)
    ):
        subject = str(
            df.iloc[row_index]["Subject"]
        ).strip()

        reason = df.iloc[
            row_index
        ]["Reason"]

        extraction = df.iloc[
            row_index
        ][
            "Brain_Region_Extraction"
        ]

        if (
            pd.notna(extraction)
            and str(extraction).strip()
            and not str(extraction).startswith(
                "ERROR:"
            )
        ):
            continue

        if (
            pd.isna(reason)
            or not str(reason).strip()
        ):
            continue

        remaining_rows.append(
            {
                "row_index": row_index,
                "Subject": subject,
                "Reason": str(
                    reason
                ).strip(),
            }
        )

    completed_count = (
        len(df)
        - len(remaining_rows)
    )

    print(
        f"Already completed: {completed_count}"
    )

    print(
        f"Remaining: {len(remaining_rows)}"
    )

    if not remaining_rows:
        print(
            "No extraction calls required."
        )
        return

    with ThreadPoolExecutor(
        max_workers=cfg.LLM_PARALLEL_CALLS
    ) as executor:

        future_to_row = {}

        for row in remaining_rows:

            future = executor.submit(
                process_subject,
                row["row_index"],
                row["Subject"],
                row["Reason"],
                client,
            )

            future_to_row[
                future
            ] = row

        with tqdm(
            total=len(
                remaining_rows
            ),
            desc=csv_path.stem,
        ) as progress_bar:

            for future in as_completed(
                future_to_row
            ):

                row_info = (
                    future_to_row[
                        future
                    ]
                )

                try:
                    result = future.result()

                    row_index = result[
                        "row_index"
                    ]

                    extraction = result[
                        "Brain_Region_Extraction"
                    ]

                except Exception as error:

                    row_index = row_info[
                        "row_index"
                    ]

                    extraction = (
                        f"ERROR: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                df.at[
                    row_index,
                    "Brain_Region_Extraction",
                ] = extraction

                # Checkpoint after every subject.
                df.to_csv(
                    output_path,
                    index=False,
                )

                progress_bar.update(
                    1
                )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_claim_extraction(
        model: str = cfg.LLM_MODEL,
        method: str = cfg.SELECTION_METHOD,
        shot: str = f"{cfg.N_SHOT}-shot",
) -> None:

    client = create_client()

    input_dir = (
        cfg.LLM_RESULTS_ROOT
        / shot
        / model
        / method
    )

    output_dir = (
        cfg.REASONING_ROOT / "Atomic_claims"
        / model
        / method
    )

    if not input_dir.exists():
        raise FileNotFoundError(
            "LLM result directory does not exist:\n"
            f"{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_files = sorted(
        input_dir.glob(
            "run_*.csv"
        )
    )

    if not csv_files:
        raise FileNotFoundError(
            "No run_*.csv files found in:\n"
            f"{input_dir}"
        )

    print(
        "\n========================================"
    )
    print(
        "CLAIM EXTRACTION"
    )
    print(
        "========================================"
    )
    print(
        f"Model : {model}"
    )
    print(
        f"Method: {method}"
    )
    print(
        f"Runs  : {len(csv_files)}"
    )
    print(
        f"Input : {input_dir}"
    )
    print(
        f"Output: {output_dir}"
    )

    for csv_path in csv_files:

        process_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            client=client,
        )

    print(
        "\nClaim extraction complete."
    )


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    run_claim_extraction()