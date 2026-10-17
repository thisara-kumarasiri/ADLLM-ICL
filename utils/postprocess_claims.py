from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import Settings as cfg


NULL_VALUES = {"", "null", "none", "nan"}

FALLBACK_COMPONENT_MAPPING = {
    "hippocampus": ["Hippocampus"],
    "hippocampal": ["Hippocampus"],
    "entorhinal cortex": ["Entorhinal Cortex"],
    "entorhinal": ["Entorhinal Cortex"],
    "fusiform": ["Fusiform Gyrus"],
    "fusiform gyrus": ["Fusiform Gyrus"],
    "lateral ventricle": ["Lateral Ventricle"],
    "lateral ventricles": ["Lateral Ventricle"],
    "inferior lateral ventricle": ["Inferior Lateral Ventricle"],
    "inferior lateral ventricles": ["Inferior Lateral Ventricle"],
    "inferior horn": ["Inferior Lateral Ventricle"],
    "temporal horn": ["Inferior Lateral Ventricle"],
    "inferior parietal": ["Inferior Parietal Cortex"],
    "inferior parietal cortex": ["Inferior Parietal Cortex"],
    "inferior temporal": ["Inferior Temporal Cortex"],
    "inferior temporal cortex": ["Inferior Temporal Cortex"],
    "lateral occipital": ["Lateral Occipital Cortex"],
    "lateral occipital cortex": ["Lateral Occipital Cortex"],
    "brain/icv ratio": ["Brain-to-ICV Ratio"],
    "brain to icv ratio": ["Brain-to-ICV Ratio"],
    "brain-to-icv ratio": ["Brain-to-ICV Ratio"],
    "icv": ["Brain-to-ICV Ratio"],
    "icv ratio": ["Brain-to-ICV Ratio"],
}


DIRECTION_RULES = (
    (
        "Not Preserved",
        (
            r"\bnot fully preserved\b",
            r"\bnot preserved\b",
            r"\bnot\b.*\bmaintain",
            r"\bno\b.*\bpreservation\b",
            r"\babsence of\b.*\bpreservation\b",
        ),
    ),
    (
        "Preserved",
        (
            r"\bpreserved\b",
            r"\bwell-preserved\b",
            r"\bmaintained\b",
            r"\bmaintain\b",
            r"\bpreservation\b",
            r"\bgood\b.*\bthickness\b",
            r"\bgood\b.*\bvolume",
            r"\bhigh\b.*\bthickness\b",
            r"\bhigh\b.*\bvolume",
            r"\breasonable\b.*\bthickness\b",
            r"\brobust\b.*\bthickness",
            r"\brobust\b.*\bvolume",
            r"\bstrong\b.*\bthickness\b",
            r"\bstrong\b.*\bvolume",
            r"\bthick\b",
            r"\bthicker\b",
            r"\brobust\b",
            r"\bintact\b",
            r"\brelative sparing\b",
            r"\brelatively spared\b",
            r"\bfairly good\b",
            r"\brelatively good\b",
            r"\bgood\b",
            r"\bstrong\b",
            r"\bnormal\b",
            r"\bnormal-range\b",
            r"\bwithin expected range\b",
            r"\bwithin normal range",
            r"\bwithin normal ranges",
            r"\bwithin the non-ad range\b",
            r"\bwithin or above non-ad reference range\b",
            r"\bfairly high\b",
            r"\bgenerally high\b",
            r"\brelatively high\b",
            r"\bhigh\b",
        ),
    ),
    (
        "Not Thinning",
        (
            r"\bno\b.*\bthin",
            r"\bnot\b.*\bthin",
            r"\bdo not\b.*\bthin",
            r"\bdoes not\b.*\bthin",
            r"\bwithout\b.*\bthin",
            r"\babsence of\b.*\bthin",
            r"\bthinning is absent\b",
            r"\brather than thinned\b",
        ),
    ),
    (
        "Thinning",
        (
            r"\bthinning\b",
            r"\bthinned\b",
            r"\bthin\b",
            r"\bthinner\b",
        ),
    ),
    (
        "Not Enlarged",
        (
            r"\bnot enlarged\b",
            r"\bnot markedly enlarged\b",
            r"\bnot prominently enlarged\b",
            r"\bnot substantially enlarged\b",
            r"\bnot strikingly enlarged\b",
            r"\bnot excessively enlarged\b",
            r"\bnot significantly enlarged\b",
            r"\babsence of\b.*\bdilatation\b",
            r"\bnot dilated\b",
            r"\bno\b.*\bventriculomegaly\b",
            r"\babsence\b.*\bdilatation\b",
            r"\black\b.*\bdilatation\b",
            r"\bdo not show\b.*\bdilat",
            r"\bdoes not show\b.*\bdilat",
        ),
    ),
    (
        "Enlarged",
        (
            r"\benlarged\b",
            r"\benlargement\b",
            r"\bexpanded\b",
            r"\bventriculomegaly\b",
            r"\bexpansion\b",
            r"\belevated\b",
            r"\bincreased\b",
            r"\blarge\b",
            r"\blarger\b",
            r"\blargest\b",
        ),
    ),
    (
        "Not Low",
        (
            r"\bnot markedly low\b",
            r"\bnot severely low\b",
            r"\bnot extremely low\b",
            r"\bnot strikingly low\b",
            r"\bnot notably low\b",
            r"\bnot disproportionately low\b",
        ),
    ),
    (
        "Low",
        (
            r"\blow\b",
            r"\blower\b",
            r"\blowest\b",
            r"\bbelow[- ]expected\b",
        ),
    ),
    (
        "Not Atrophy",
        (
            r"\bno\b.*\batroph",
            r"\bnot\b.*\batroph",
            r"\bdoes not\b.*\batroph",
            r"\bdo not\b.*\batroph",
            r"\bwithout\b.*\batroph",
            r"\babsence of\b.*\batroph",
            r"\black of\b.*\batroph",
            r"\blacks\b.*\batroph",
        ),
    ),
    (
        "Atrophy",
        (
            r"\batrophy\b",
            r"\batrophic\b",
            r"\batrophied\b",
        ),
    ),
    (
        "Not Reduced",
        (
            r"\bnot\b.*\breduc",
            r"\bno\b.*\bloss\b",
            r"\bnot\b.*\bloss\b",
            r"\bdoes not\b.*\bloss\b",
            r"\bdo not\b.*\bloss\b",
            r"\bwithout\b.*\bloss\b",
            r"\babsence of\b.*\bloss\b",
            r"\bloss\b.*\bnot present\b",
            r"\bno\b.*\bdegeneration\b",
            r"\bno\b.*\bneurodegeneration\b",
            r"\bnot\b.*\bdegeneration\b",
            r"\bnot\b.*\bneurodegeneration\b",
            r"\bdoes not\b.*\bdegeneration\b",
            r"\bdoes not\b.*\bneurodegeneration\b",
            r"\bdo not\b.*\bdegeneration\b",
            r"\bdo not\b.*\bneurodegeneration\b",
            r"\bnot\b.*\bsmall",
            r"\bnot\b.*\bshrunken\b",
            r"\bnot\b.*\bshrinkage\b",
            r"\bno\b.*\bshrinkage\b",
            r"\bnot\b.*\bcollapsed\b",
            r"\bnot\b.*\bdepressed\b",
        ),
    ),
    (
        "Reduced",
        (
            r"\breduced\b",
            r"\breduction\b",
            r"\breductions\b",
            r"\bloss\b",
            r"\bdegeneration\b",
            r"\bneurodegeneration\b",
            r"\bsmall\b",
            r"\bsmaller\b",
            r"\bsmallest\b",
            r"\bdiminished\b",
            r"\bdecreased\b",
            r"\bhalved\b",
            r"\bsubnormal\b",
            r"\bshrunken\b",
            r"\bshrinkage\b",
            r"\bcollapsed\b",
            r"\bdepressed\b",
        ),
    ),
    (
        "Asymmetric",
        (
            r"\basymmetric\b",
            r"\basymmetry\b",
            r"\basymmetries\b",
            r"\bright-sided\b",
            r"\bleft-sided\b",
            r"\bon the right\b",
            r"\bon the left\b",
        ),
    ),
    (
        "Above Average",
        (
            r"\babove average\b",
            r"\babove-average\b",
            r"\babove the range\b",
        ),
    ),
)


# Preserve the effective behavior of the original dictionary:
# Atrophy maps to the separate "Atrophic" group.
DIRECTION_GROUP_MAP = {
    "Preserved": "Preserved",
    "Not Reduced": "Preserved",
    "Not Thinning": "Preserved",
    "Not Atrophy": "Preserved",
    "Not Enlarged": "Preserved",
    "Not Low": "Preserved",
    "Above Average": "Preserved",
    "Not Preserved": "Reduced",
    "Reduced": "Reduced",
    "Thinning": "Reduced",
    "Low": "Reduced",
    "Atrophy": "Atrophic",
    "Enlarged": "Increased",
    "Asymmetric": "Asymmetry",
}


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode CSV: {path}")


def clean_null(value):
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in NULL_VALUES:
        return None

    return value


def normalize_region(region):
    region = clean_null(region)

    if region is None:
        return None

    region = (
        region.replace("**", "")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )

    region = re.sub(r"\s+", " ", region)

    return region.strip().lower()


def load_region_mapping(mapping_file: str | Path) -> dict[str, list[str]]:
    mapping_df = read_csv_safe(mapping_file)

    required_columns = {"Original_Region", "Component_Regions"}
    missing = required_columns - set(mapping_df.columns)

    if missing:
        raise ValueError(
            "Region mapping file is missing columns: "
            f"{sorted(missing)}"
        )

    region_mapping: dict[str, list[str]] = {}

    for _, row in mapping_df.iterrows():
        key = normalize_region(row["Original_Region"])

        if key is None or pd.isna(row["Component_Regions"]):
            continue

        components = [
            component.strip()
            for component in str(row["Component_Regions"]).split(";")
            if component.strip()
        ]

        if components:
            region_mapping[key] = components

    return region_mapping


def clean_region_for_fallback(region):
    region = normalize_region(region)

    if region is None:
        return None

    # Remove parenthetical example/reference annotations.
    region = re.sub(
        r"\([^)]*(?:example|examples|ex\s*\d+|ad|non[-\s]?ad|"
        r"alzheimer|case|cases)[^)]*\)",
        "",
        region,
        flags=re.IGNORECASE,
    )

    prefix_patterns = (
        r"^(?:ad|non[-\s]?ad)\s+examples?\s*\d*\s*",
        r"^examples?\s*\d+\s*(?:ad|non[-\s]?ad)?\s*",
        r"^ex\s*\d+\s*(?:ad|non[-\s]?ad)?\s*",
        r"^(?:ad|non[-\s]?ad)\s+cases?\s*",
        r"^(?:ad|non[-\s]?ad)\s+",
    )

    for pattern in prefix_patterns:
        region = re.sub(
            pattern,
            "",
            region,
            flags=re.IGNORECASE,
        )

    # Remove demographic prefixes such as "AD female".
    region = re.sub(
        r"^(?:ad|non[-\s]?ad)\s+(?:male|female)\s+",
        "",
        region,
        flags=re.IGNORECASE,
    )

    # Component regions are non-lateralized in the final output.
    region = re.sub(
        r"^(?:left|right)\s+",
        "",
        region,
        flags=re.IGNORECASE,
    )

    # Remove explanatory parenthetical wording.
    region = re.sub(
        r"\s*\(\s*temporal horn\s*\)\s*",
        "",
        region,
        flags=re.IGNORECASE,
    )

    if (
        "left hippocampus" in region
        and "right hippocampus" in region
    ):
        region = "hippocampus"

    return re.sub(r"\s+", " ", region).strip()


def infer_special_components(text):
    text = clean_null(text)

    if text is None:
        return None

    text = text.lower()

    if (
        re.search(r"\bage\b", text)
        or re.search(r"\byear[- ]old\b", text)
        or re.search(r"\byears?\s+old\b", text)
        or re.search(r"\b\d+(?:\.\d+)?\s+years?\b", text)
    ):
        return ["Age"]

    if re.search(
        r"\bbrain(?:\s+volume)?\s*"
        r"(?:/|-?\s*to\s*-?)\s*icv\s+ratios?\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ["Brain-to-ICV Ratio"]

    return None


def get_region_components(
    original_region,
    region_mapping: dict[str, list[str]],
):
    special_components = infer_special_components(original_region)

    if special_components:
        return special_components

    region_key = normalize_region(original_region)

    if region_key is None:
        return None

    components = region_mapping.get(region_key)

    if components:
        return components

    cleaned_region = clean_region_for_fallback(original_region)

    if cleaned_region is None:
        return None

    components = region_mapping.get(cleaned_region)

    if components:
        return components

    return FALLBACK_COMPONENT_MAPPING.get(cleaned_region)


def parse_brain_region_extraction(text) -> list[dict[str, str | None]]:
    if pd.isna(text):
        return []

    rows = []

    for line in str(text).splitlines():
        line = line.strip()

        if not line.startswith("|"):
            continue

        cells = [
            cell.strip().replace("**", "")
            for cell in line.strip("|").split("|")
        ]

        if len(cells) != 5:
            continue

        claim, claim_type, region, value, direction = cells

        if claim.lower() == "claim":
            continue

        if all(
            re.fullmatch(r":?-+:?", cell)
            for cell in cells
        ):
            continue

        rows.append(
            {
                "claim": clean_null(claim),
                "claim_type": clean_null(claim_type),
                "original_region": clean_null(region),
                "value": clean_null(value),
                "direction": clean_null(direction),
            }
        )

    return rows


def clean_direction(value):
    value = clean_null(value)

    if value is None:
        return None

    text = value.lower()

    for label, patterns in DIRECTION_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return label

    return value


def get_file_metadata(csv_path: str | Path):
    csv_path = Path(csv_path)

    method = csv_path.parent.name
    model = csv_path.parent.parent.name

    match = re.search(
        r"run[_-]?(\d+)",
        csv_path.stem,
        flags=re.IGNORECASE,
    )

    run = int(match.group(1)) if match else csv_path.stem

    return model, method, run


def build_subject_to_group(test_file: str | Path) -> dict[str, str]:
    test_df = read_csv_safe(test_file)

    required_columns = {"Subject", "Group"}
    missing = required_columns - set(test_df.columns)

    if missing:
        raise ValueError(
            f"Test file is missing columns: {sorted(missing)}"
        )

    labels = (
        test_df[["Subject", "Group"]]
        .drop_duplicates(subset=["Subject"])
        .copy()
    )

    labels["Subject"] = labels["Subject"].astype(str)

    return dict(
        zip(
            labels["Subject"],
            labels["Group"],
        )
    )


def process_reasoning_files(
    reasoning_dir: str | Path,
    mapping_file: str | Path,
    test_file: str | Path,
) -> tuple[pd.DataFrame, set[str]]:
    reasoning_dir = Path(reasoning_dir)

    if not reasoning_dir.exists():
        raise FileNotFoundError(
            f"Reasoning directory does not exist: {reasoning_dir}"
        )

    region_mapping = load_region_mapping(mapping_file)
    subject_to_group = build_subject_to_group(test_file)

    output_rows = []
    unmapped_regions: set[str] = set()

    csv_files = sorted(
        reasoning_dir.rglob(
            "run_*.csv"
        )
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No run_*.csv files found in: {reasoning_dir}"
        )

    for csv_path in csv_files:
        model, method, run = get_file_metadata(csv_path)

        # print(
        #     f"Processing: {model} | {method} | run {run}"
        # )

        df = read_csv_safe(csv_path)

        required_columns = {
            "Subject",
            "Brain_Region_Extraction",
        }

        missing = required_columns - set(df.columns)

        # if missing:
        #     print(
        #         f"  SKIPPED - missing columns: {sorted(missing)}"
        #     )
        #     continue

        for _, subject_row in df.iterrows():
            subject = str(subject_row["Subject"])
            prediction = subject_row.get("Prediction")
            true_label = subject_to_group.get(subject)

            claims = parse_brain_region_extraction(
                subject_row["Brain_Region_Extraction"]
            )

            for claim_row in claims:
                original_region = (
                    claim_row["original_region"]
                    or claim_row["claim"]
                )

                components = get_region_components(
                    original_region=original_region,
                    region_mapping=region_mapping,
                )

                if not components:
                    components = [None]

                    if original_region is not None:
                        unmapped_regions.add(
                            str(original_region)
                        )

                for component in components:
                    output_rows.append(
                        {
                            "model": model,
                            "method": method,
                            "run": run,
                            "subject": subject,
                            "True_Label": true_label,
                            "Prediction": prediction,
                            "original_region": original_region,
                            "component_region": component,
                            "claim": claim_row["claim"],
                            "claim_type": claim_row["claim_type"],
                            "value": claim_row["value"],
                            "direction": claim_row["direction"],
                        }
                    )

    result_df = pd.DataFrame(
        output_rows,
        columns=[
            "model",
            "method",
            "run",
            "subject",
            "True_Label",
            "Prediction",
            "original_region",
            "component_region",
            "claim",
            "claim_type",
            "value",
            "direction",
        ],
    )

    if result_df.empty:
        result_df["direction_clean"] = pd.Series(dtype="object")
        result_df["direction_group"] = pd.Series(dtype="object")
        return result_df, unmapped_regions

    result_df["direction_clean"] = (
        result_df["direction"].apply(clean_direction)
    )

    result_df["direction_group"] = (
        result_df["direction_clean"]
        .map(DIRECTION_GROUP_MAP)
        .where(lambda series: series.notna(), None)
    )

    result_df = (
        result_df
        .sort_values(
            by=[
                "model",
                "method",
                "run",
                "subject",
                "component_region",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return result_df, unmapped_regions


def create_reasoning_claims(
    reasoning_dir: str | Path,
    mapping_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> Path:
    output_file = Path(output_file)

    result_df, unmapped_regions = process_reasoning_files(
        reasoning_dir=reasoning_dir,
        mapping_file=mapping_file,
        test_file=test_file,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    # print(f"Saved: {output_file}")

    # if unmapped_regions:
    #     print("\nUnmapped regions:")
    #     for region in sorted(
    #         unmapped_regions,
    #         key=str.lower,
    #     ):
    #         print(f"  - {region}")

    return output_file


def main():
    reasoning_dir = (
        cfg.BRAIN_REGION_EXTRACTION_ROOT
        / cfg.MODEL
        / cfg.METHOD
    )

    mapping_file = (
        cfg.PROJECT_ROOT
        / "config"
        / "brain_region_mapping.csv"
    )

    output_file = (
        cfg.REASONING_OUTPUT_ROOT
        / cfg.MODEL
        / cfg.METHOD
        / "reasoning_claims.csv"
    )

    create_reasoning_claims(
        reasoning_dir=reasoning_dir,
        mapping_file=mapping_file,
        test_file=cfg.TEST_CSV,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()

