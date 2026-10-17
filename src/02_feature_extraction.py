"""
Extract regional MRI features from FreeSurfer outputs and create a
subject-level feature vector table.

For each subject, the script reads:
    stats/lh.aparc.stats
    stats/rh.aparc.stats
    stats/aseg.stats

Extracted features
------------------
1. Cortical regional gray-matter volumes from lh/rh.aparc.stats
2. Cortical regional mean thicknesses from lh/rh.aparc.stats
3. Subcortical structure volumes from aseg.stats
4. Brain-volume-to-ICV-ratio from aseg.stats
5. Age, Sex, and Group from the subject information file

Output
------
One CSV where:
    rows    = subjects
    columns = demographic information + MRI features
"""

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import Settings as cfg

# ============================================================
# CORTICAL FEATURES
# ============================================================

CORTICAL_REGION_NAMES = {
    "bankssts": "Banks of Superior Temporal Sulcus",
    "caudalanteriorcingulate": "Caudal Anterior Cingulate Cortex",
    "caudalmiddlefrontal": "Caudal Middle Frontal Cortex",
    "cuneus": "Cuneus Cortex",
    "entorhinal": "Entorhinal Cortex",
    "frontalpole": "Frontal Pole",
    "fusiform": "Fusiform Gyrus",
    "inferiorparietal": "Inferior Parietal Cortex",
    "inferiortemporal": "Inferior Temporal Cortex",
    "insula": "Insular Cortex",
    "isthmuscingulate": "Isthmus Cingulate Cortex",
    "lateraloccipital": "Lateral Occipital Cortex",
    "lateralorbitofrontal": "Lateral Orbitofrontal Cortex",
    "lingual": "Lingual Gyrus",
    "medialorbitofrontal": "Medial Orbitofrontal Cortex",
    "middletemporal": "Middle Temporal Cortex",
    "paracentral": "Paracentral Lobule",
    "parahippocampal": "Parahippocampal Gyrus",
    "parsopercularis": "Pars Opercularis",
    "parsorbitalis": "Pars Orbitalis",
    "parstriangularis": "Pars Triangularis",
    "pericalcarine": "Pericalcarine Cortex",
    "postcentral": "Postcentral Gyrus",
    "posteriorcingulate": "Posterior Cingulate Cortex",
    "precentral": "Precentral Gyrus",
    "precuneus": "Precuneus Cortex",
    "rostralanteriorcingulate": "Rostral Anterior Cingulate Cortex",
    "rostralmiddlefrontal": "Rostral Middle Frontal Cortex",
    "superiorfrontal": "Superior Frontal Cortex",
    "superiorparietal": "Superior Parietal Cortex",
    "superiortemporal": "Superior Temporal Cortex",
    "supramarginal": "Supramarginal Gyrus",
    "temporalpole": "Temporal Pole",
    "transversetemporal": "Transverse Temporal Cortex",
}

def extract_aparc_features(stats_file: Path, hemisphere: str):
    """
    Extract cortical gray-matter volume and mean cortical thickness
    from a FreeSurfer aparc.stats file.

    Parameters
    ----------
    stats_file : Path
        Path to lh.aparc.stats or rh.aparc.stats.

    hemisphere : str
        Either "lh" or "rh".

    Returns
    -------
    dict
        Feature-name/value pairs.
    """

    features = {}

    hemisphere_name = {
        "lh": "Left",
        "rh": "Right",
    }[hemisphere]

    with stats_file.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            columns = line.split()

            if len(columns) < 5:
                continue

            region = columns[0]

            try:
                gray_volume = float(columns[3])
                mean_thickness = float(columns[4])

            except ValueError:
                continue
            
            # --------------------------------------------------------
            # Replace hyphens with spaces
            # --------------------------------------------------------

            region_name = CORTICAL_REGION_NAMES.get(
                region,
                region.replace("-", " ").title()
            )

            features[
                f"{hemisphere_name} {region_name} Volume"
            ] = gray_volume

            features[
                f"{hemisphere_name} {region_name} Cortical Thickness"
            ] = mean_thickness

    return features

# ============================================================
# SUBCORTICAL FEATURES / Brain to ICV ratio
# ============================================================

ASEG_STRUCTURE_NAMES = {
    "Left-Inf-Lat-Vent": "Left Inferior Lateral Ventricle",
    "Right-Inf-Lat-Vent": "Right Inferior Lateral Ventricle",

    "Left-VentralDC": "Left Ventral Diencephalon",
    "Right-VentralDC": "Right Ventral Diencephalon",

    "Left-Accumbens-area": "Left Accumbens Area",
    "Right-Accumbens-area": "Right Accumbens Area",

    "Left-choroid-plexus": "Left Choroid Plexus",
    "Right-choroid-plexus": "Right Choroid Plexus",

    "Left-vessel": "Left Vessel",
    "Right-vessel": "Right Vessel",

    "Brain-Stem": "Brain Stem",
    "Optic-Chiasm": "Optic Chiasm",

    "CC_Anterior": "Anterior Corpus Callosum",
    "CC_Central": "Central Corpus Callosum",
    "CC_Mid_Anterior": "Mid Anterior Corpus Callosum",
    "CC_Mid_Posterior": "Mid Posterior Corpus Callosum",
    "CC_Posterior": "Posterior Corpus Callosum",
}

def extract_aseg_features(stats_file: Path):
    """
    Extract subcortical structure volumes from aseg.stats and
    calculate the Brain Volume-to-ICV ratio.

    The ratio is calculated as:
        BrainSegVol / eTIV
    """

    features = {}

    brain_seg_vol = None
    etiv = None

    with stats_file.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:

            line = line.strip()

            # --------------------------------------------------------
            # Extract BrainSegVol and eTIV from # Measure lines
            # --------------------------------------------------------

            if line.startswith("# Measure"):

                parts = [
                    part.strip()
                    for part in line.split(",")
                ]

                if len(parts) >= 4:

                    measure_name = parts[1]

                    try:
                        value = float(parts[3])

                    except ValueError:
                        continue

                    if measure_name == "BrainSegVol":
                        brain_seg_vol = value

                    elif measure_name == "eTIV":
                        etiv = value

                continue

            # --------------------------------------------------------
            # Skip other comments / empty lines
            # --------------------------------------------------------

            if not line or line.startswith("#"):
                continue

            columns = line.split()

            # --------------------------------------------------------
            # Expected aseg.stats structure:
            #
            # Index|SegId|NVoxels|Volume_mm3|StructName|normMean|normStdDev|normMin|normMax|normRange
            # --------------------------------------------------------

            if len(columns) < 5:
                continue

            structure = columns[4]

            try:
                volume = float(columns[3])

            except ValueError:
                continue

            structure_name = ASEG_STRUCTURE_NAMES.get(
                structure,
                structure.replace("-", " ").replace("_", " ")
            )

            features[
                f"{structure_name} Volume"
            ] = volume

    # --------------------------------------------------------
    # Calculate Brain Volume-to-ICV ratio
    # --------------------------------------------------------

    if (
        brain_seg_vol is not None
        and etiv is not None
        and etiv != 0
    ):
        features[
            "Brain volume to ICV ratio"
        ] = brain_seg_vol / etiv

    else:
        features[
            "Brain volume to ICV ratio"
        ] = None

    return features

# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def extract_subject_features(
    subject_dir: Path
):
    """
    Extract all MRI features for one FreeSurfer subject.
    """

    stats_dir = subject_dir / "stats"

    required_files = {
        "lh_aparc": stats_dir / "lh.aparc.stats",
        "rh_aparc": stats_dir / "rh.aparc.stats",
        "aseg": stats_dir / "aseg.stats",
    }

    missing_files = [
        path.name
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required FreeSurfer files: "
            + ", ".join(missing_files)
        )

    features = {}

    # --------------------------------------------------------
    # Cortical volume + thickness
    # --------------------------------------------------------

    features.update(
        extract_aparc_features(
            required_files["lh_aparc"],
            hemisphere="lh",
        )
    )

    features.update(
        extract_aparc_features(
            required_files["rh_aparc"],
            hemisphere="rh",
        )
    )

    # --------------------------------------------------------
    # Subcortical volumes
    # --------------------------------------------------------

    features.update(
        extract_aseg_features(
            required_files["aseg"]
        )
    )

    return features

# ============================================================
# LOAD SUBJECT INFORMATION
# ============================================================

def load_subject_info(
    info_file: Path,
    subject_column: str,
    age_column: str,
    sex_column: str,
    group_column: str,
):
    """
    Load subject demographic and diagnostic information.

    The metadata file must contain:
        subject identifier
        age
        sex
        diagnostic group
    """

    suffix = info_file.suffix.lower()

    if suffix == ".csv":
        info_df = pd.read_csv(info_file)

    elif suffix in {".xlsx", ".xls"}:
        info_df = pd.read_excel(info_file)

    elif suffix in {".tsv", ".txt"}:
        info_df = pd.read_csv(
            info_file,
            sep="\t",
        )

    else:
        raise ValueError(
            f"Unsupported metadata file format: {suffix}"
        )

    required_columns = [
        subject_column,
        age_column,
        sex_column,
        group_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in info_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing metadata columns: "
            + ", ".join(missing_columns)
        )

    info_df = info_df[
        required_columns
    ].copy()

    info_df = info_df.rename(
        columns={
            subject_column: "Subject",
            age_column: "Age",
            sex_column: "Sex",
            group_column: "Group",
        }
    )

    # --------------------------------------------------------
    # Make subject IDs strings so they can safely be matched
    # --------------------------------------------------------

    info_df["Subject"] = (
        info_df["Subject"]
        .astype(str)
        .str.strip()
    )

    return info_df

# ============================================================
# BUILD FEATURE VECTOR SPACE
# ============================================================

def build_feature_table(
    subjects_dir: Path,
    info_df: pd.DataFrame,
):
    """
    Extract FreeSurfer features for every subject and create the
    subject × feature matrix.
    """

    rows = []
    failed_subjects = []

    # --------------------------------------------------------
    # Dictionary makes metadata lookup faster
    # --------------------------------------------------------

    metadata = (
        info_df
        .set_index("Subject")
        .to_dict(orient="index")
    )

    for subject_dir in sorted(subjects_dir.iterdir()):

        if not subject_dir.is_dir():
            continue

        subject_name = subject_dir.name

        # --------------------------------------------------------
        # Skip FreeSurfer template directories if present
        # --------------------------------------------------------

        if subject_name.lower() in {
            "fsaverage",
            "fsaverage_sym",
        }:
            continue

        print(
            f"Extracting features: {subject_name}",
            flush=True,
        )

        # --------------------------------------------------------
        # Find subject metadata
        # --------------------------------------------------------

        if subject_name not in metadata:

            print(
                f"  Metadata not found: {subject_name}",
                flush=True,
            )

            failed_subjects.append(
                (
                    subject_name,
                    "Subject not found in metadata file",
                )
            )

            continue

        try:

            feature_values = extract_subject_features(
                subject_dir
            )

            subject_info = metadata[subject_name]

            row = {
                "Subject": subject_name,
                "Age": subject_info["Age"],
                "Sex": subject_info["Sex"],
                "Group": subject_info["Group"],
            }

            row.update(feature_values)

            rows.append(row)

        except Exception as exc:

            print(
                f"  Failed: {subject_name}: {exc}",
                flush=True,
            )

            failed_subjects.append(
                (
                    subject_name,
                    str(exc),
                )
            )

    feature_df = pd.DataFrame(rows)

    return feature_df, failed_subjects

# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Extract FreeSurfer cortical thickness, regional "
            "volumes, subcortical volumes, and global brain "
            "measurements into a subject-level feature table."
        )
    )

    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=cfg.FREESURFER_SUBJECTS_DIR,
        help=(
            "FreeSurfer SUBJECTS_DIR containing one directory "
            "per subject. Defaults to "
            "Settings.FREESURFER_SUBJECTS_DIR."
        ),
    )

    parser.add_argument(
        "--info-file",
        type=Path,
        default=cfg.SUBJECT_INFO_FILE,
        help=(
            "CSV, TSV, TXT, or Excel file containing subject "
            "metadata. Defaults to Settings.SUBJECT_INFO_FILE."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.ALL_FEATURES_CSV,
        help=(
            "Output feature-vector CSV. Defaults to "
            "Settings.ALL_FEATURES_CSV."
        ),
    )

    parser.add_argument(
        "--subject-column",
        default="Subject",
        help="Subject-ID column in the metadata file.",
    )

    parser.add_argument(
        "--age-column",
        default="Age",
        help="Age column in the metadata file.",
    )

    parser.add_argument(
        "--sex-column",
        default="Sex",
        help="Sex column in the metadata file.",
    )

    parser.add_argument(
        "--group-column",
        default="Group",
        help="Diagnostic group column in the metadata file.",
    )

    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if not args.subjects_dir.exists():
        raise FileNotFoundError(
            f"SUBJECTS_DIR does not exist: "
            f"{args.subjects_dir}"
        )

    if not args.info_file.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: "
            f"{args.info_file}"
        )

    # --------------------------------------------------------
    # Load demographics / labels
    # --------------------------------------------------------

    info_df = load_subject_info(
        info_file=args.info_file,
        subject_column=args.subject_column,
        age_column=args.age_column,
        sex_column=args.sex_column,
        group_column=args.group_column,
    )

    print(
        f"Loaded metadata for {len(info_df)} subjects."
    )

    # --------------------------------------------------------
    # Create feature-vector space
    # --------------------------------------------------------

    feature_df, failed_subjects = build_feature_table(
        subjects_dir=args.subjects_dir,
        info_df=info_df,
    )

    # --------------------------------------------------------
    # Keep metadata columns first
    # --------------------------------------------------------
    
    metadata_columns = [
        "Subject",
        "Age",
        "Sex",
        "Group",
    ]

    feature_columns = sorted(
        [
            column
            for column in feature_df.columns
            if column not in metadata_columns
        ]
    )

    feature_df = feature_df[
        metadata_columns + feature_columns
    ]

    # --------------------------------------------------------
    # Remove feature columns containing only zeros
    # --------------------------------------------------------

    zero_only_columns = [
        column
        for column in feature_columns
        if (feature_df[column].fillna(0) == 0).all()
    ]

    if zero_only_columns:

        print(
            f"Removing {len(zero_only_columns)} "
            "feature columns containing only zeros:"
        )

        for column in zero_only_columns:
            print(f"  - {column}")

        feature_df = feature_df.drop(
            columns=zero_only_columns
        )

    # --------------------------------------------------------
    # Exclude white-matter hypointensity measures
    # --------------------------------------------------------

    wm_hypointensity_columns = [
        column
        for column in feature_df.columns
        if "WM hypointensities" in column
    ]

    feature_df = feature_df.drop(
        columns=wm_hypointensity_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        args.output,
        index=False,
    )

    # --------------------------------------------------------
    # Recalculate feature count after removing zero columns
    # --------------------------------------------------------

    final_feature_columns = [
        column
        for column in feature_df.columns
        if column not in metadata_columns
    ]

    print()

    print(
        f"Feature vectors created for "
        f"{len(feature_df)} subjects."
    )

    print(
        f"Number of MRI features: "
        f"{len(final_feature_columns)}"
    )

    print(
        f"Saved to: {args.output}"
    )

    # --------------------------------------------------------
    # Failed-subject log
    # --------------------------------------------------------
    
    if failed_subjects:

        failure_file = (
            args.output.parent
            / "feature_extraction_failures.csv"
        )

        pd.DataFrame(
            failed_subjects,
            columns=[
                "Subject",
                "Error",
            ],
        ).to_csv(
            failure_file,
            index=False,
        )

        print(
            f"{len(failed_subjects)} subjects "
            f"could not be processed."
        )

        print(
            f"Failure log: {failure_file}"
        )


if __name__ == "__main__":
    main()