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
You are evaluating the semantic consistency of three reasoning
outputs generated independently for the same Alzheimer's disease
classification case.

Your task is ONLY to evaluate consistency across the three reasoning
outputs.

Do NOT evaluate whether the reasoning is medically correct.
Do NOT use outside medical knowledge.
Do NOT correct the reasoning.
Do NOT judge the final diagnosis itself.

Evaluate whether the three reasoning outputs are semantically
consistent with one another.

Consider:

1. Brain-region evidence
   - Determine whether claims about the same brain regions are
     compatible across the three runs.
   - Different runs do NOT need to mention exactly the same regions.

2. Direction of findings
   - Check whether descriptions such as reduced, preserved, enlarged,
     thinning, atrophic, high, low, or asymmetric are compatible when
     they refer to the same evidence.

3. Differences in emphasis
   - Different wording, different levels of detail, or mentioning
     additional supporting evidence should NOT be treated as a
     contradiction.

Classify the relationship across ALL THREE reasoning outputs as
exactly one of:

CONSISTENT

All three reasoning outputs are semantically compatible.
They may use different wording, emphasize different regions, or
contain different levels of detail, but there are no substantive
contradictions.

PARTIALLY CONSISTENT

The reasoning outputs contain some compatible interpretations but
also meaningful differences in the evidence or interpretation.
These differences are more substantial than simple wording or
omission, but they do not amount to a clear direct contradiction.

CONTRADICTORY

At least one reasoning output makes a substantive claim that
directly conflicts with another reasoning output about the same
evidence or its interpretation.

Important:

A region appearing in one run but not another is NOT by itself
a contradiction.

Different numerical detail is NOT automatically a contradiction
unless the values or their interpretations are clearly incompatible.

Return exactly this format:

Classification: CONSISTENT / PARTIALLY CONSISTENT / CONTRADICTORY
Explanation: <brief explanation of why>

REASONING RUN 1

{REASON_1}

REASONING RUN 2

{REASON_2}

REASONING RUN 3

{REASON_3}
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
# CONSISTENCY CALL
# ============================================================

def evaluate_consistency(
    reason_1: str,
    reason_2: str,
    reason_3: str,
    client: AzureOpenAI,
) -> str:

    prompt = (
        PROMPT_TEMPLATE
        .replace("{REASON_1}", reason_1)
        .replace("{REASON_2}", reason_2)
        .replace("{REASON_3}", reason_3)
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
            "Consistency evaluation returned an empty response."
        )

    return content.strip()


# ============================================================
# RETRY
# ============================================================

def evaluate_with_retry(
    reason_1: str,
    reason_2: str,
    reason_3: str,
    client: AzureOpenAI,
) -> str:

    last_error = None

    for attempt in range(
        1,
        cfg.RETRY_COUNT + 1,
    ):
        try:
            return evaluate_consistency(
                reason_1=reason_1,
                reason_2=reason_2,
                reason_3=reason_3,
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
        "Consistency evaluation failed "
        f"after {cfg.RETRY_COUNT} attempts."
    ) from last_error


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    subject: str,
    reason_1: str,
    reason_2: str,
    reason_3: str,
    client: AzureOpenAI,
) -> dict:

    try:
        answer = evaluate_with_retry(
            reason_1=reason_1,
            reason_2=reason_2,
            reason_3=reason_3,
            client=client,
        )

    except Exception as error:
        answer = (
            f"ERROR: {type(error).__name__}: {error}"
        )

    return {
        "Subject": subject,
        "Answer": answer,
    }


# ============================================================
# LOAD EXACTLY THREE RUNS
# ============================================================

def load_three_runs(
    method_dir: Path,
) -> pd.DataFrame:

    run_files = []

    for run_number in [1, 2, 3]:

        run_path = (
            method_dir
            / f"run_{run_number}.csv"
        )

        if not run_path.exists():
            raise FileNotFoundError(
                f"Missing run file:\n{run_path}"
            )

        run_files.append(
            run_path
        )

    dfs = []

    for run_number, run_path in enumerate(
        run_files,
        start=1,
    ):

        df = pd.read_csv(
            run_path
        )

        required_columns = {
            "Subject",
            "Reason",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{run_path} is missing columns: "
                f"{sorted(missing)}"
            )

        df = df[
            [
                "Subject",
                "Reason",
            ]
        ].copy()

        df["Subject"] = (
            df["Subject"]
            .astype(str)
            .str.strip()
        )

        df = df.rename(
            columns={
                "Reason": f"Reason_{run_number}"
            }
        )

        if (
            df["Subject"]
            .duplicated()
            .any()
        ):
            duplicates = (
                df.loc[
                    df["Subject"].duplicated(),
                    "Subject",
                ]
                .tolist()
            )

            raise ValueError(
                f"Duplicate subjects found in "
                f"{run_path}:\n{duplicates}"
            )

        dfs.append(
            df
        )

    merged_df = (
        dfs[0]
        .merge(
            dfs[1],
            on="Subject",
            how="inner",
        )
        .merge(
            dfs[2],
            on="Subject",
            how="inner",
        )
    )

    for column in [
        "Reason_1",
        "Reason_2",
        "Reason_3",
    ]:

        merged_df = merged_df[
            merged_df[column].notna()
            & merged_df[column]
            .astype(str)
            .str.strip()
            .ne("")
        ]

    return (
        merged_df
        .reset_index(drop=True)
    )


# ============================================================
# RESTORE PREVIOUS PROGRESS
# ============================================================

def load_existing_results(
    output_path: Path,
) -> dict:

    if not output_path.exists():
        return {}

    try:
        previous_df = pd.read_csv(
            output_path
        )

    except Exception:
        return {}

    required_columns = {
        "Subject",
        "Answer",
    }

    if not required_columns.issubset(
        previous_df.columns
    ):
        return {}

    existing_results = {}

    for _, row in previous_df.iterrows():

        subject = str(
            row["Subject"]
        ).strip()

        answer = row[
            "Answer"
        ]

        if pd.isna(answer):
            continue

        answer = str(
            answer
        ).strip()

        if not answer:
            continue

        if answer.startswith(
            "ERROR:"
        ):
            continue

        existing_results[
            subject
        ] = answer

    return existing_results


# ============================================================
# PROCESS SELECTED MODEL / METHOD
# ============================================================

def process_method(
    method_dir: Path,
    output_path: Path,
    client: AzureOpenAI,
) -> None:

    # print(
    #     f"\nProcessing consistency: "
    #     f"{cfg.LLM_MODEL} | {cfg.METHOD}"
    # )

    merged_df = load_three_runs(
        method_dir
    )

    print(
        f"Subjects present in all 3 runs: "
        f"{len(merged_df)}"
    )

    existing_results = load_existing_results(
        output_path
    )

    result_lookup = dict(
        existing_results
    )

    remaining_rows = []

    for _, row in merged_df.iterrows():

        subject = str(
            row["Subject"]
        ).strip()

        if subject in existing_results:
            continue

        remaining_rows.append(
            {
                "Subject": subject,
                "Reason_1": str(
                    row["Reason_1"]
                ).strip(),
                "Reason_2": str(
                    row["Reason_2"]
                ).strip(),
                "Reason_3": str(
                    row["Reason_3"]
                ).strip(),
            }
        )

    print(
        f"Already completed: "
        f"{len(existing_results)}"
    )

    print(
        f"Remaining: "
        f"{len(remaining_rows)}"
    )

    if not remaining_rows:

        final_df = pd.DataFrame(
            [
                {
                    "Subject": subject,
                    "Answer": result_lookup.get(
                        subject
                    ),
                }
                for subject in (
                    merged_df["Subject"]
                    .astype(str)
                    .str.strip()
                )
            ]
        )

        final_df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Completed: {output_path}"
        )

        return

    with ThreadPoolExecutor(
        max_workers=cfg.LLM_PARALLEL_CALLS
    ) as executor:

        future_to_subject = {}

        for row in remaining_rows:

            future = executor.submit(
                process_subject,
                row["Subject"],
                row["Reason_1"],
                row["Reason_2"],
                row["Reason_3"],
                client,
            )

            future_to_subject[
                future
            ] = row["Subject"]

        with tqdm(
            total=len(remaining_rows),
            desc=(
                f"{cfg.LLM_MODEL} | "
                f"{cfg.SELECTION_METHOD}"
            ),
        ) as progress_bar:

            for future in as_completed(
                future_to_subject
            ):

                subject = (
                    future_to_subject[
                        future
                    ]
                )

                try:
                    result = future.result()

                    answer = result[
                        "Answer"
                    ]

                except Exception as error:
                    answer = (
                        f"ERROR: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                result_lookup[
                    subject
                ] = answer

                current_df = pd.DataFrame(
                    [
                        {
                            "Subject": s,
                            "Answer": result_lookup.get(
                                s
                            ),
                        }
                        for s in (
                            merged_df["Subject"]
                            .astype(str)
                            .str.strip()
                        )
                        if s in result_lookup
                    ]
                )

                current_df.to_csv(
                    output_path,
                    index=False,
                )

                progress_bar.update(
                    1
                )

    final_df = pd.DataFrame(
        [
            {
                "Subject": subject,
                "Answer": result_lookup.get(
                    subject
                ),
            }
            for subject in (
                merged_df["Subject"]
                .astype(str)
                .str.strip()
            )
        ]
    )

    final_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_consistency_analysis(
        model: str = cfg.LLM_MODEL,
        method: str = cfg.SELECTION_METHOD,
        shot: str = f"{cfg.N_SHOT}-shot"
) -> None:

    if cfg.LLM_N_REPEATS != 3:
        raise ValueError(
            "Reasoning consistency analysis requires "
            "exactly 3 LLM runs."
        )

    client = create_client()

    method_dir = (
        cfg.LLM_RESULTS_ROOT / shot
        / model
        / method
    )

    if not method_dir.exists():
        raise FileNotFoundError(
            "LLM results directory does not exist:\n"
            f"{method_dir}"
        )

    output_dir = (
        cfg.REASONING_ROOT / "Consistency" 
        / model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{method}.csv"
    )

    print(
        "\n========================================"
    )
    print(
        "REASONING CONSISTENCY"
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
        f"Input : {method_dir}"
    )
    print(
        f"Output: {output_path}"
    )

    process_method(
        method_dir=method_dir,
        output_path=output_path,
        client=client,
    )

    print(
        "\nConsistency analysis complete."
    )


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    run_consistency_analysis()