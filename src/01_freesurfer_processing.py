"""
Run FreeSurfer anatomical reconstruction and brainstem segmentation
for a folder of T1-weighted MRI scans.

Input
-----
A directory containing NIfTI files (.nii or .nii.gz).

Output
------
A FreeSurfer SUBJECTS_DIR containing one reconstructed directory
per subject.

Requirements
------------
- FreeSurfer installed and configured
- recon-all available on PATH
- segmentBS.sh available on PATH
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import Settings as cfg


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    input_path: Path,
    subject_name: str,
    subjects_dir: Path,
    threads_per_subject: int,
    run_brainstem: bool = True,
):
    """
    Run FreeSurfer recon-all and optional brainstem segmentation
    for a single MRI subject.
    """

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(subjects_dir)

    print(
        f"Starting: {subject_name}",
        flush=True,
    )

    recon_cmd = [
        "recon-all",
        "-i",
        str(input_path),
        "-s",
        subject_name,
        "-all",
        "-parallel",
        "-openmp",
        str(threads_per_subject),
    ]

    try:
        subprocess.run(
            recon_cmd,
            check=True,
            env=env,
            text=True,
            capture_output=True,
        )

        print(
            f"recon-all completed: {subject_name}",
            flush=True,
        )

        if run_brainstem:
            print(
                f"Running brainstem segmentation: {subject_name}",
                flush=True,
            )

            brainstem_cmd = [
                "segmentBS.sh",
                subject_name,
            ]

            subprocess.run(
                brainstem_cmd,
                check=True,
                env=env,
                text=True,
                capture_output=True,
            )

            print(
                f"Brainstem segmentation completed: {subject_name}",
                flush=True,
            )

        return subject_name, True, None

    except subprocess.CalledProcessError as exc:
        error_message = (
            f"Command failed with return code {exc.returncode}\n"
            f"STDOUT:\n{exc.stdout}\n"
            f"STDERR:\n{exc.stderr}"
        )

        return subject_name, False, error_message


# ============================================================
# FIND MRI FILES
# ============================================================

def find_subjects(
    input_dir: Path,
):
    """
    Return a list of (MRI path, subject name) pairs.
    """

    subjects = []

    for file_path in sorted(
        input_dir.iterdir()
    ):

        if file_path.name.endswith(
            ".nii.gz"
        ):
            subject_name = (
                file_path.name[:-7]
            )

        elif file_path.suffix == ".nii":
            subject_name = (
                file_path.stem
            )

        else:
            continue

        subjects.append(
            (
                file_path,
                subject_name,
            )
        )

    return subjects


# ============================================================
# CHECK EXISTING OUTPUT
# ============================================================

def subject_is_complete(
    subject_name: str,
    subjects_dir: Path,
):
    """
    Check whether FreeSurfer recon-all has completed previously.
    """

    done_file = (
        subjects_dir
        / subject_name
        / "scripts"
        / "recon-all.done"
    )

    return done_file.exists()


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run FreeSurfer recon-all and optional brainstem "
            "segmentation for NIfTI MRI scans."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=cfg.MRI_INPUT_DIR,
        help=(
            "Directory containing .nii or .nii.gz MRI files. "
            "Defaults to Settings.MRI_INPUT_DIR."
        ),
    )

    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=cfg.FREESURFER_SUBJECTS_DIR,
        help=(
            "FreeSurfer SUBJECTS_DIR output directory. "
            "Defaults to Settings.FREESURFER_SUBJECTS_DIR."
        ),
    )

    parser.add_argument(
        "--max-subjects",
        type=int,
        default=cfg.FREESURFER_MAX_SUBJECTS,
        help=(
            "Maximum number of subjects processed simultaneously."
        ),
    )

    parser.add_argument(
        "--threads-per-subject",
        type=int,
        default=cfg.FREESURFER_THREADS_PER_SUBJECT,
        help=(
            "OpenMP threads used by FreeSurfer for each subject."
        ),
    )

    parser.add_argument(
        "--skip-brainstem",
        action="store_true",
        help=(
            "Skip segmentBS.sh brainstem segmentation."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess subjects even if recon-all.done exists."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_dir = args.input_dir
    subjects_dir = args.subjects_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    subjects_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subjects = find_subjects(
        input_dir
    )

    print(
        f"Found {len(subjects)} MRI scans"
    )

    if not args.force:

        subjects_to_process = []

        for input_path, subject_name in subjects:

            if subject_is_complete(
                subject_name,
                subjects_dir,
            ):
                print(
                    f"Skipping completed subject: "
                    f"{subject_name}"
                )

            else:
                subjects_to_process.append(
                    (
                        input_path,
                        subject_name,
                    )
                )

    else:
        subjects_to_process = subjects

    run_brainstem = (
        cfg.FREESURFER_RUN_BRAINSTEM
        and not args.skip_brainstem
    )

    print(
        f"Subjects to process: "
        f"{len(subjects_to_process)}"
    )

    print(
        f"Concurrent subjects: "
        f"{args.max_subjects}"
    )

    print(
        f"Threads per subject: "
        f"{args.threads_per_subject}"
    )

    print(
        f"Brainstem segmentation: "
        f"{run_brainstem}"
    )

    print(
        "Maximum requested threads: "
        f"{args.max_subjects * args.threads_per_subject}"
    )

    failures = []

    with ProcessPoolExecutor(
        max_workers=args.max_subjects
    ) as executor:

        futures = {
            executor.submit(
                process_subject,
                input_path,
                subject_name,
                subjects_dir,
                args.threads_per_subject,
                run_brainstem,
            ): subject_name
            for input_path, subject_name
            in subjects_to_process
        }

        for future in as_completed(
            futures
        ):

            subject_name = futures[
                future
            ]

            try:
                name, success, error = (
                    future.result()
                )

                if success:
                    print(
                        f"✓ COMPLETED: {name}",
                        flush=True,
                    )

                else:
                    print(
                        f"✗ FAILED: {name}",
                        flush=True,
                    )

                    failures.append(
                        (
                            name,
                            error,
                        )
                    )

            except Exception as exc:
                print(
                    f"✗ UNEXPECTED ERROR: "
                    f"{subject_name}",
                    flush=True,
                )

                failures.append(
                    (
                        subject_name,
                        str(exc),
                    )
                )

    if failures:

        error_log = (
            subjects_dir
            / "failed_subjects.log"
        )

        with error_log.open(
            "w",
            encoding="utf-8",
        ) as file:

            for (
                subject_name,
                error,
            ) in failures:

                file.write(
                    f"===== "
                    f"{subject_name} "
                    f"=====\n"
                )

                file.write(
                    f"{error}\n\n"
                )

        print(
            f"\n{len(failures)} "
            f"subjects failed."
        )

        print(
            f"Failure details written to: "
            f"{error_log}"
        )

    else:
        print(
            "\nAll subjects completed successfully."
        )


if __name__ == "__main__":
    main()