"""
Extraction utilities for Alzheimer's disease LLM outputs.

The functions in this module extract:

    Prediction
    Reason

from raw LLM responses.

Expected prediction labels
--------------------------
AD : Alzheimer's disease
CN : Cognitively normal / non-Alzheimer's

The extraction logic supports several common LLM response styles,
including:

    Prediction: AD
    Diagnosis: Alzheimer's disease
    Final Diagnosis: CN
    Alzheimer's disease: Yes
    Decision: Non-Alzheimer's
    narrative classifications

"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd


# CONSTANTS

POSITIVE_LABEL = "AD"
NEGATIVE_LABEL = "CN"

ALZHEIMER_RE = (
    r"Alzheimer"
    r"(?:['\u2018\u2019]s)?"
    r"(?:\s+disease)?"
)

NON_ALZHEIMER_RE = (
    rf"non\s*[-\u2010-\u2015\u2212]?\s*"
    rf"{ALZHEIMER_RE}"
)

NEGATED_ALZHEIMER_RE = (
    rf"(?:"
    rf"not\s+{ALZHEIMER_RE}|"
    rf"not\s+(?:likely|consistent\s+with|suggestive\s+of|"
    rf"supportive\s+of|typical\s+of)\s+{ALZHEIMER_RE}|"
    rf"unlikely\s+{ALZHEIMER_RE}|"
    rf"no\s+(?:clear\s+)?(?:MRI\s+)?"
    rf"(?:evidence|sign|pattern)\s+of\s+{ALZHEIMER_RE}|"
    rf"does\s+not\s+(?:support|suggest|show)\s+{ALZHEIMER_RE}|"
    rf"not\s+strongly\s+(?:supportive|suggestive)\s+of\s+"
    rf"{ALZHEIMER_RE}"
    rf")"
)

STRUCTURED_PREFIX = (
    r"(?:Analysis|Decision|Diagnosis|Assessment|Impression|"
    r"Conclusion|Output|Prediction|Result)"
)

# RAW RESPONSE CLEANING

def extract_content_payload(
    text: str,
) -> str:
    """
    Extract only the model-generated content when the stored
    answer contains a serialized response object.

    If no wrapper is detected, the original text is returned.
    """

    match = re.search(
        r"content=(['\"])(.*?)\1"
        r"(?=\s+(?:additional_kwargs|"
        r"response_metadata|id|tool_calls|"
        r"invalid_tool_calls|usage_metadata)=|$)",
        text,
        flags=re.DOTALL,
    )

    if match:
        return match.group(2)

    return text


def strip_bracketed_text(
    text: str,
) -> str:
    """
    Remove text enclosed in (), [] and {}.

    This helps avoid extracting labels from metadata or formatting
    comments rather than from the actual model decision.
    """

    cleaned = re.sub(
        r"\([^)]*\)",
        "",
        text,
    )

    cleaned = re.sub(
        r"\[[^\]]*\]",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\{[^}]*\}",
        "",
        cleaned,
    )

    return cleaned


def normalise_answer_text(
    answer_value,
) -> str:
    """
    Convert a raw response into normalised plain text.
    """

    if pd.isna(answer_value):
        return ""

    text = str(
        answer_value
    )

    text = extract_content_payload(
        text
    )

    text = (
        text
        .replace("\\\n", " ")
        .replace("\n", " ")
        .replace("\\'", "'")
        .replace("\\’", "’")
        .replace("**", "")
    )

    return " ".join(
        text.split()
    ).strip()


# PREDICTION CLEANING

def clean_prediction_text(
    text: str,
) -> str:
    """
    Convert a prediction-like text span into AD, CN, or NA.
    """

    prediction = " ".join(
        str(text).split()
    )

    prediction = (
        prediction
        .strip("- '")
        .strip('"')
        .strip()
    )

    # Binary labels
    first_label = re.match(
        r"^\s*([01])\b",
        prediction,
    )

    if first_label:

        return (
            NEGATIVE_LABEL
            if first_label.group(1) == "0"
            else POSITIVE_LABEL
        )

    # Explicit cognitively normal / control terminology
    if re.search(
        r"\b("
        r"normal\s+aging|"
        r"cognitively\s+normal|"
        r"cognitive(?:ly)?\s+normal|"
        r"CN|"
        r"healthy\s+control|"
        r"control"
        r")\b",
        prediction,
        flags=re.IGNORECASE,
    ):
        return NEGATIVE_LABEL

    # --------------------------------------------------------
    # Explicit negative Alzheimer wording
    # Must be checked BEFORE positive Alzheimer's wording.
    # --------------------------------------------------------

    if re.search(
        NEGATED_ALZHEIMER_RE,
        prediction,
        flags=re.IGNORECASE,
    ):
        return NEGATIVE_LABEL


    # --------------------------------------------------------
    # Explicit non-Alzheimer's wording
    # --------------------------------------------------------

    if re.search(
        NON_ALZHEIMER_RE,
        prediction,
        flags=re.IGNORECASE,
    ):
        return NEGATIVE_LABEL


    # --------------------------------------------------------
    # Positive Alzheimer's wording
    # --------------------------------------------------------

    if re.search(
        ALZHEIMER_RE,
        prediction,
        flags=re.IGNORECASE,
    ):
        return POSITIVE_LABEL

    # Standalone class labels
    if re.fullmatch(
        r"AD",
        prediction,
        flags=re.IGNORECASE,
    ):
        return POSITIVE_LABEL

    if re.fullmatch(
        r"CN",
        prediction,
        flags=re.IGNORECASE,
    ):
        return NEGATIVE_LABEL

    # Numeric/confidence-only outputs are not predictions
    if re.fullmatch(
        r"\d+(?:\.\d+)?%?",
        prediction,
    ):
        return "NA"

    return prediction


# REASON CLEANING

def clean_reason_text(
    text: str,
) -> str:
    """
    Clean extracted reasoning text.
    """

    reason = " ".join(
        str(text).split()
    )

    reason = (
        reason
        .strip("- '")
        .strip('"')
        .strip()
    )

    return reason


# PREDICTION EXTRACTION

def extract_prediction(
    answer_value,
) -> str:
    """
    Extract an Alzheimer's disease classification from an LLM
    response.

    Returns
    -------
    AD
        Alzheimer's disease.

    CN
        Cognitively normal / non-Alzheimer's.

    NA
        A reliable prediction could not be extracted.
    """

    if pd.isna(answer_value):
        return ""

    text = str(
        answer_value
    )

    text = extract_content_payload(
        text
    )

    text = strip_bracketed_text(
        text
    )

    text = (
        text
        .replace("\\\n", " ")
        .replace("\n", " ")
        .replace("\\'", "'")
        .replace("\\’", "’")
        .replace("**", "")
    )

    text = " ".join(
        text.split()
    )

    # ========================================================
    # DIRECT NEGATIVE CLASSIFICATION PHRASES
    # ========================================================

    negative_prediction_patterns = [

        # --------------------------------------------------
        # Direct negative forms
        # --------------------------------------------------

        rf"^\s*no\s+{ALZHEIMER_RE}\b",
        rf"^\s*no\s*,?\s*does\s+not\s+meet\s+criteria\s+for\s+{ALZHEIMER_RE}\b",
        rf"^\s*does\s+not\s+meet\s+criteria\s+for\s+{ALZHEIMER_RE}\b",
        rf"^\s*no\s*,?\s*(?:the\s+subject\s+)?does\s+not\s+have\s+{ALZHEIMER_RE}\b",
        rf"^\s*(?:the\s+subject\s+)?does\s+not\s+have\s+{ALZHEIMER_RE}\b",
        rf"^\s*no\s*,?\s*does\s+not\s+have\s+{ALZHEIMER_RE}\b",

        # Not AD
        rf"^\s*not\s+AD\b",

        # Not Alzheimer's disease
        rf"^\s*not\s+{ALZHEIMER_RE}\b",

        # Not likely Alzheimer's disease
        rf"^\s*not\s+likely\s+{ALZHEIMER_RE}\b",

        # More likely not Alzheimer's disease
        rf"^\s*more\s+likely\s+not\s+{ALZHEIMER_RE}\b",

        # Likely not Alzheimer's disease
        rf"^\s*likely\s+not\s+{ALZHEIMER_RE}\b",

        # Likely no Alzheimer's disease
        rf"^\s*likely\s+no\s+{ALZHEIMER_RE}"
        rf"(?:\s+(?:pattern|evidence|changes?))?\b",

        # Likely does not have Alzheimer's disease
        rf"^\s*likely\s+does\s+not\s+have\s+{ALZHEIMER_RE}\b",

        # Does not have Alzheimer's disease
        rf"^\s*does\s+not\s+have\s+{ALZHEIMER_RE}\b",

        # Not consistent with Alzheimer's disease
        rf"^\s*not\s+(?:clearly\s+)?consistent\s+with\s+"
        rf"{ALZHEIMER_RE}\b",

        # Unlikely Alzheimer's disease
        rf"^\s*(?:most\s+)?unlikely\s+{ALZHEIMER_RE}\b",

        # Unlikely to have Alzheimer's disease
        rf"^\s*(?:most\s+)?unlikely\s+to\s+have\s+"
        rf"{ALZHEIMER_RE}\b",

        # No clear evidence/sign/pattern of Alzheimer's disease
        rf"^\s*no\s+(?:clear\s+)?(?:MRI\s+)?"
        rf"(?:evidence|sign|pattern)\s+of\s+"
        rf"{ALZHEIMER_RE}\b",

        # No — ...
        rf"^\s*No\s*[-—–:]",


        # --------------------------------------------------
        # Structured forms
        # e.g. Decision:, Diagnosis:, Conclusion:
        # --------------------------------------------------

        # Decision: Not AD
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"not\s+AD\b",

        # Decision: Not Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"not\s+{ALZHEIMER_RE}\b",

        # Decision: Not likely Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"not\s+likely\s+{ALZHEIMER_RE}\b",

        # Decision: More likely not Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"more\s+likely\s+not\s+{ALZHEIMER_RE}\b",

        # Decision: Likely not Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"likely\s+not\s+{ALZHEIMER_RE}\b",

        # Decision: Likely no Alzheimer's disease pattern
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"likely\s+no\s+{ALZHEIMER_RE}"
        rf"(?:\s+(?:pattern|evidence|changes?))?\b",

        # Decision: Likely does not have Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"likely\s+does\s+not\s+have\s+{ALZHEIMER_RE}\b",

        # Decision: Does not have Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"does\s+not\s+have\s+{ALZHEIMER_RE}\b",

        # Decision: Unlikely Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"(?:most\s+)?unlikely\s+{ALZHEIMER_RE}\b",

        # Decision: Unlikely to have Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"(?:most\s+)?unlikely\s+to\s+have\s+"
        rf"{ALZHEIMER_RE}\b",

        # Decision: No clear MRI evidence/sign/pattern of Alzheimer's
        rf"^\s*{STRUCTURED_PREFIX}\s*:\s*"
        rf"no\s+(?:clear\s+)?(?:MRI\s+)?"
        rf"(?:evidence|sign|pattern)\s+of\s+"
        rf"{ALZHEIMER_RE}\b",

        # Decision: Not consistent with Alzheimer's disease
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"not\s+(?:clearly\s+)?consistent\s+with\s+"
        rf"{ALZHEIMER_RE}\b",

        # Decision: No ...
        rf"^\s*(?:Decision|Diagnosis|Assessment|Impression|"
        rf"Conclusion|Output|Prediction|Result)\s*:\s*"
        rf"No\b",

        # Assessment: Does not meet criteria for Alzheimer's disease
        rf"^\s*{STRUCTURED_PREFIX}\s*:\s*"
        rf"does\s+not\s+meet\s+criteria\s+for\s+"
        rf"{ALZHEIMER_RE}\b",
    ]


    for pattern in negative_prediction_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return NEGATIVE_LABEL

    # Markers indicating start of reasoning rather than label
    stop_marker = (
        r"(?:(?:Clinical\s+)?Reason(?:ing)?\s*:|"
        r"Reason\s*:|"
        r"Reason\s+for\s+Decision\s*:|"
        r"Reason\s+behind\s+this\s+decision"
        r"\s*(?::|is)\s*|"
        r"Rationale\s*:|"
        r"Why\s*:|"
        r"Explanation\s*:|"
        r"Evidence\s*:|"
        r"Short\s+[Rr]easoning\s*:|"
        r"Confidence(?:\s+(?:Score|Level))?"
        r"\s*(?::|is|of)|"
        r"$)"
    )

    # Common structured answer formats
    patterns = [

        # AD / CN directly at beginning
        (
            rf"^\s*("
            rf"{NON_ALZHEIMER_RE}|"
            rf"{ALZHEIMER_RE}|"
            rf"AD|CN"
            rf")\b"
        ),

        # Likely Alzheimer's disease
        (
            rf"^\s*(?:Likely|Probably|Most\s+likely|"
            rf"Consistent\s+with)\s+"
            rf"(?:is\s+)?("
            rf"{NON_ALZHEIMER_RE}|"
            rf"{ALZHEIMER_RE}|"
            rf"AD|CN"
            rf")\b"
        ),

        # Prediction: AD
        (
            rf"(?:Final\s+)?"
            rf"(?:Output|Prediction|Conclusion|Result|"
            rf"Summary|Diagnosis|Decision|Assessment)"
            rf"\s*:\s*("
            rf"{NON_ALZHEIMER_RE}|"
            rf"{ALZHEIMER_RE}|"
            rf"AD|CN"
            rf")\b"
        ),

        # Final Diagnosis: AD
        (
            rf"Final\s+Diagnosis\s*:\s*("
            rf"{NON_ALZHEIMER_RE}|"
            rf"{ALZHEIMER_RE}|"
            rf"AD|CN"
            rf")\b"
        ),

        # Alzheimer's disease: Yes / No
        (
            rf"{ALZHEIMER_RE}"
            rf"\s*:\s*"
            rf"(?:Final\s+decision\s*:\s*)?"
            rf"(Yes|No)\b"
        ),

        # Non-Alzheimer's: Yes / No
        (
            rf"{NON_ALZHEIMER_RE}"
            rf"\s*:\s*"
            rf"(?:Final\s+decision\s*:\s*)?"
            rf"(Yes|No)\b"
        ),

        # Generic decision span
        (
            rf"(?:Binary\s+)?"
            rf"(?:Decision|Diagnosis|Assessment|Status|"
            rf"Output|Prediction)"
            rf"\s*:\s*(.*?)\s*"
            rf"{stop_marker}"
        ),

        # "my decision is AD"
        (
            rf"(?:my\s+)?decision\s+"
            rf"(?:is|would\s+be|leans\s+towards|"
            rf"would\s+lean\s+to)\s*("
            rf"{NON_ALZHEIMER_RE}|"
            rf"{ALZHEIMER_RE}|"
            rf"AD|CN"
            rf")\b"
        ),
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if not match:
            continue

        matched_text = (
            match.group(0)
        )

        captured = (
            clean_prediction_text(
                match.group(1)
            )
        )

        # Alzheimer's disease: Yes / No
        if re.search(
            rf"{ALZHEIMER_RE}"
            rf"\s*:\s*"
            rf"(?:Final\s+decision\s*:\s*)?"
            rf"Yes\b",
            matched_text,
            flags=re.IGNORECASE,
        ):
            return POSITIVE_LABEL

        if re.search(
            rf"{ALZHEIMER_RE}"
            rf"\s*:\s*"
            rf"(?:Final\s+decision\s*:\s*)?"
            rf"No\b",
            matched_text,
            flags=re.IGNORECASE,
        ):
            return NEGATIVE_LABEL

        # Non-Alzheimer's: Yes / No
        if re.search(
            rf"{NON_ALZHEIMER_RE}"
            rf"\s*:\s*"
            rf"(?:Final\s+decision\s*:\s*)?"
            rf"Yes\b",
            matched_text,
            flags=re.IGNORECASE,
        ):
            return NEGATIVE_LABEL

        if re.search(
            rf"{NON_ALZHEIMER_RE}"
            rf"\s*:\s*"
            rf"(?:Final\s+decision\s*:\s*)?"
            rf"No\b",
            matched_text,
            flags=re.IGNORECASE,
        ):
            return POSITIVE_LABEL

        # Prevent rationale paragraphs being mistaken for labels
        if captured not in {
            POSITIVE_LABEL,
            NEGATIVE_LABEL,
            "NA",
        }:

            if (
                len(captured) > 80
                or re.search(
                    r"\b("
                    r"because|given|findings|however|"
                    r"therefore|analysis|assessment|"
                    r"volume|cortex|hippocampus|"
                    r"entorhinal"
                    r")\b",
                    captured,
                    flags=re.IGNORECASE,
                )
            ):
                continue

        if captured in {
            POSITIVE_LABEL,
            NEGATIVE_LABEL,
        }:
            return captured

    # Narrative classification
    if re.search(
        rf"\b("
        rf"classif(?:y|ied|ication)|"
        rf"diagnos(?:is|e|ed)|"
        rf"consistent\s+with|"
        rf"supports?"
        rf")\b"
        rf"[^.!?]{{0,180}}"
        rf"\b{NON_ALZHEIMER_RE}\b",
        text,
        flags=re.IGNORECASE,
    ):
        return NEGATIVE_LABEL

    if re.search(
        rf"\b("
        rf"classif(?:y|ied|ication)|"
        rf"diagnos(?:is|e|ed)|"
        rf"consistent\s+with|"
        rf"supports?"
        rf")\b"
        rf"[^.!?]{{0,180}}"
        rf"\b{ALZHEIMER_RE}\b",
        text,
        flags=re.IGNORECASE,
    ):
        return POSITIVE_LABEL

    # Conservative fallback
    fallback_match = re.search(
        rf"^\s*(.*?)\s*"
        rf"{stop_marker}",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if fallback_match:

        candidate = (
            clean_prediction_text(
                fallback_match.group(1)
            )
        )

        if candidate in {
            POSITIVE_LABEL,
            NEGATIVE_LABEL,
        }:
            return candidate

    return "NA"


# REASON EXTRACTION

def extract_reason(
    answer_value,
) -> str:
    """
    Extract the reasoning/rationale portion of an LLM response.

    The extractor supports explicit labels such as:

        Reason:
        Reasoning:
        Clinical Reasoning:
        Rationale:
        Explanation:
        Evidence:

    It also contains fallbacks for less structured responses.
    """

    if pd.isna(answer_value):
        return ""

    text = str(
        answer_value
    )

    text = extract_content_payload(
        text
    )

    raw_text = text

    text = (
        text
        .replace("\\\n", " ")
        .replace("\n", " ")
        .replace("\\'", "'")
        .replace("\\’", "’")
        .replace("**", "")
    )

    text = " ".join(
        text.split()
    )

    # Common case:
    # AD
    # reasoning text...

    normalised_raw = (
        raw_text
        .replace("\\\n", "\n")
    )

    normalised_raw = re.sub(
        r"^\s*Output\s*:\s*",
        "",
        normalised_raw,
        flags=re.IGNORECASE,
    )

    line_split = re.split(
        r"\r?\n",
        normalised_raw,
        maxsplit=1,
    )

    if len(line_split) == 2:

        first_line = (
            line_split[0]
            .strip()
        )

        first_line = re.sub(
            r"[*_`]+",
            "",
            first_line,
        )

        first_line = (
            first_line
            .strip()
            .strip(":-")
        )

        remainder = (
            line_split[1]
            .strip()
        )

        if (
            remainder
            and re.search(
                rf"^(?:"
                rf"{ALZHEIMER_RE}|"
                rf"{NON_ALZHEIMER_RE}|"
                rf"AD|CN"
                rf")$",
                first_line,
                flags=re.IGNORECASE,
            )
        ):
            return clean_reason_text(
                remainder
            )

    # Explicit FINDINGS:
    findings_match = re.search(
        r"\bFINDINGS?\s*:\s*(.*)$",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if findings_match:

        return clean_reason_text(
            findings_match.group(1)
        )

    # Explicit reason-like heading
    reason_heading = re.search(
        r"(?:"
        r"(?:Clinical\s+)?Reason(?:ing)?|"
        r"Reason\s+for\s+Decision|"
        r"Reason\s+behind\s+this\s+decision|"
        r"Rationale|"
        r"Why|"
        r"Explanation|"
        r"Short\s+[Rr]easoning|"
        r"Evidence"
        r")"
        r"(?:\s*:|\s{2,}|$)",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if reason_heading:

        start = (
            reason_heading.end()
        )

        # Stop before later decision/confidence fields
        end_match = re.search(
            r"(?:"
            r"Confidence"
            r"(?:\s+(?:Score|Level))?"
            r"\s*(?::|is|of)"
            r"|"
            r"(?:Final\s+)?"
            r"(?:Decision|Diagnosis|Assessment|"
            r"Status|Prediction)"
            r"\s*:"
            r")",
            text[start:],
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if end_match:

            reason = text[
                start:
                start + end_match.start()
            ]

        else:

            reason = text[
                start:
            ]

        return clean_reason_text(
            reason
        )

    # Final Answer at end:
    # reasoning...
    # Final Answer: AD
    final_answer_match = re.search(
        r"\bFinal\s+Answer\s*:\s*",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if final_answer_match:

        reason = text[
            :final_answer_match.start()
        ]

        reason = remove_leading_prediction(
            reason
        )

        return clean_reason_text(
            reason
        )

    # Decision/Diagnosis appears after reasoning
    decision_match = re.search(
        r"(?:Binary\s+)?"
        r"(?:Decision|Diagnosis|Assessment|Status|"
        r"Output|Prediction|Conclusion|Result|Summary)"
        r"\s*:\s*",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if decision_match:

        before = (
            text[
                :decision_match.start()
            ]
            .strip()
        )

        after = (
            text[
                decision_match.end():
            ]
            .strip()
        )

        # Remove diagnosis at beginning of the second part
        after = re.sub(
            rf"^\s*(?:"
            rf"{ALZHEIMER_RE}|"
            rf"{NON_ALZHEIMER_RE}|"
            rf"AD|CN|NA"
            rf")"
            rf"\s*[:\-.]?\s*",
            "",
            after,
            flags=re.IGNORECASE,
        )

        reason = " ".join(
            part
            for part in [
                before,
                after,
            ]
            if part
        )

        reason = remove_leading_prediction(
            reason
        )

        if reason:
            return clean_reason_text(
                reason
            )

    # Confidence appears after narrative reasoning
    confidence_match = re.search(
        r"(?:with\s+a\s+)?"
        r"(?:the\s+|my\s+)?"
        r"Confidence"
        r"(?:\s+(?:Score|Level))?"
        r"\s*(?::|is|of)",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if confidence_match:

        reason = text[
            :confidence_match.start()
        ]

        reason = remove_leading_prediction(
            reason
        )

        return clean_reason_text(
            reason
        )

    # Narrative fallback
    narrative_candidate = (
        remove_leading_prediction(
            text
        )
    )

    narrative_candidate = (
        clean_reason_text(
            narrative_candidate
        )
    )

    if narrative_candidate:

        sentence_count = len(
            re.findall(
                r"[.!?](?:\s|$)",
                narrative_candidate,
            )
        )

        rationale_cues = re.search(
            r"\b("
            r"because|given|findings|patterns|"
            r"suggest|suggests|indicative|"
            r"evidence|align|consistent|"
            r"conclude|classify|diagnosis|"
            r"therefore|however|volume|"
            r"thickness|hippocampus|entorhinal"
            r")\b",
            narrative_candidate,
            flags=re.IGNORECASE,
        )

        if (
            sentence_count >= 2
            and rationale_cues
        ):
            return (
                narrative_candidate
            )

    return ""


# REMOVE LEADING DIAGNOSIS

def remove_leading_prediction(
    text: str,
) -> str:
    """
    Remove a diagnosis label appearing at the start of a
    reasoning string.
    """

    return re.sub(
        rf"^\s*(?:"
        rf"{ALZHEIMER_RE}|"
        rf"{NON_ALZHEIMER_RE}|"
        rf"AD|CN|NA"
        rf")"
        rf"\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )


# FILE PROCESSOR

class ExtractionProcessor:
    """
    Add Prediction and Reason columns to raw LLM result files.
    """

    def is_known_non_answer_file(
        self,
        csv_path: str,
    ) -> bool:
        """
        Identify CSV files that are expected not to contain
        LLM answers.
        """

        file_name = (
            os.path.basename(
                csv_path
            )
            .lower()
        )

        return (
            file_name.startswith(
                "metrics"
            )
            or file_name
            == "info.csv"
        )

    def process_file(
        self,
        csv_path: str,
    ) -> None:
        """
        Extract Prediction and Reason from one result CSV.
        """

        df = pd.read_csv(
            csv_path
        )

        # Only raw LLM files contain Answer
        if "Answer" not in df.columns:

            if not self.is_known_non_answer_file(
                csv_path
            ):

                print(
                    f"Skipping "
                    f"{os.path.basename(csv_path)}: "
                    "missing 'Answer' column"
                )

            return

        # Create output columns
        if "Prediction" not in df.columns:
            df["Prediction"] = ""

        if "Reason" not in df.columns:
            df["Reason"] = ""

        df[
            "Prediction"
        ] = (
            df[
                "Prediction"
            ]
            .astype(object)
        )

        df[
            "Reason"
        ] = (
            df[
                "Reason"
            ]
            .astype(object)
        )

        # Process non-empty responses
        non_empty_answer_mask = (
            df["Answer"].notna()
            & (
                df[
                    "Answer"
                ]
                .astype(str)
                .str.strip()
                != ""
            )
        )

        df.loc[
            non_empty_answer_mask,
            "Prediction",
        ] = (
            df.loc[
                non_empty_answer_mask,
                "Answer",
            ]
            .apply(
                extract_prediction
            )
        )

        df.loc[
            non_empty_answer_mask,
            "Reason",
        ] = (
            df.loc[
                non_empty_answer_mask,
                "Answer",
            ]
            .apply(
                extract_reason
            )
        )

        # Save updated file
        df.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"Updated "
            f"{os.path.basename(csv_path)}"
        )

    def process_all_files(
        self,
        input_dir: str | os.PathLike,
    ) -> None:
        """
        Recursively process every result CSV under a directory.
        """

        csv_files = sorted(
            glob.glob(
                os.path.join(
                    str(input_dir),
                    "**",
                    "*.csv",
                ),
                recursive=True,
            )
        )

        if not csv_files:

            print(
                "No CSV files found under "
                f"{input_dir}"
            )

            return

        for csv_path in csv_files:

            self.process_file(
                csv_path
            )


# SIMPLE FUNCTION WRAPPERS

def process_file(
    csv_path: str,
) -> None:
    """
    Process a single LLM result file.
    """

    ExtractionProcessor().process_file(
        csv_path
    )


def process_all_files(
    input_dir: str,
) -> None:
    """
    Process every LLM result CSV below a directory.
    """

    ExtractionProcessor().process_all_files(
        input_dir
    )