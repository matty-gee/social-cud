#!/usr/bin/env python
"""Classify diagnosis from the two dimension-average feature sets.

The script runs two across-subject searchlights:

1. ``affiliation_minus_power`` uses one contrast map per subject.
2. ``affiliation_and_power`` uses both condition maps as separate channels.

Both models use the same five diagnosis-by-sex-stratified folds, a standardized
linear SVM, and balanced accuracy.  Run from this directory with:

    conda run -n brainiak python run_dimension_classification_searchlights.py \
        --workers 12
    conda run -n brainiak python run_dimension_classification_searchlights.py \
        --smoothing-fwhm 4 --workers 12

Inputs:

    lss_decision/voxelwise/dimension_average/
        manifest.csv
        grey_matter_mask.nii.gz
        maps/

Outputs:

    lss_decision/searchlights/statistical-maps/dimension_classification/
        subjects_and_folds.csv
        analysis.json
        affiliation_minus_power/diagnosis_balanced_accuracy.nii.gz
        affiliation_and_power/diagnosis_balanced_accuracy.nii.gz
"""

from __future__ import annotations

# BrainIAK workers provide the parallelism.  Limiting numerical libraries to
# one thread prevents every worker from creating another pool of CPU threads.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# The reusable searchlight engine lives at code/searchlights.  Resolve it from
# this file so the runner works regardless of the current working directory.
SEARCHLIGHT_DIR = Path(__file__).resolve().parents[3] / "searchlights"
sys.path.insert(0, str(SEARCHLIGHT_DIR))

from classifier_searchlight import (  # noqa: E402
    ClassificationSearchlight,
    load_image_stack,
    make_stratified_folds,
)


PROJECT_DATA = Path("/Users/matty_gee/Desktop/Social/SocialCUD")
LSS_ROOT = PROJECT_DATA / "results" / "glms" / "lss_decision"
INPUT_DIR = LSS_ROOT / "voxelwise" / "dimension_average"
DEFAULT_MASK = INPUT_DIR / "grey_matter_mask.nii.gz"
DEFAULT_METADATA = PROJECT_DATA / "data" / "questionnaire_data.xlsx"
DEFAULT_OUTPUT_DIR = (
    LSS_ROOT
    / "searchlights"
    / "statistical-maps"
    / "dimension_classification"
)

MODEL_FOLDERS = {
    "contrast": "affiliation_minus_power",
    "two_channel": "affiliation_and_power",
}
MAP_NAME = "diagnosis_balanced_accuracy.nii.gz"
RANDOM_STATE = 2026


def smoothing_suffix(fwhm: float | None) -> str:
    """Return the suffix shared by smoothed inputs and result files."""
    if fwhm is None:
        return ""
    if fwhm <= 0:
        raise ValueError("Smoothing FWHM must be positive")
    label = f"{fwhm:g}".replace(".", "p")
    return f"_smoothed{label}fwhm"


def normalize_subject_id(value: object) -> str:
    """Return a subject identifier in ``sub-18001`` form."""

    if pd.isna(value):
        raise ValueError("Subject IDs cannot be missing")
    if isinstance(value, (int, np.integer)):
        value = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = str(int(value))
    else:
        value = str(value).strip().removeprefix("sub-")
    return f"sub-{value}"


def prepare_subject_table(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Align map paths, diagnosis, and sex in one explicit subject order."""

    map_columns = [
        "subject_id",
        "n_affiliation_trials",
        "n_power_trials",
        "affiliation_map",
        "power_map",
    ]
    metadata_columns = ["sub_id", "dx", "sex"]
    if missing := set(map_columns) - set(manifest.columns):
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    if missing := set(metadata_columns) - set(metadata.columns):
        raise KeyError(f"Metadata are missing columns: {sorted(missing)}")

    maps = manifest[map_columns].copy()
    maps["subject_id"] = maps["subject_id"].map(normalize_subject_id)
    participant_data = metadata[metadata_columns].copy()
    participant_data["subject_id"] = participant_data["sub_id"].map(
        normalize_subject_id
    )
    participant_data = participant_data.drop(columns="sub_id")

    if maps["subject_id"].duplicated().any():
        raise ValueError("The manifest contains duplicate subjects")
    if participant_data["subject_id"].duplicated().any():
        raise ValueError("The metadata contain duplicate subjects")

    table = maps.merge(
        participant_data,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    if table[["dx", "sex"]].isna().any().any():
        missing = table.loc[table[["dx", "sex"]].isna().any(axis=1), "subject_id"]
        raise ValueError("Missing diagnosis or sex for: " + ", ".join(missing))

    table["dx"] = pd.to_numeric(table["dx"], errors="raise").astype(int)
    table["sex"] = pd.to_numeric(table["sex"], errors="raise").astype(int)
    if set(table["dx"]) != {0, 1}:
        raise ValueError("Diagnosis must contain both binary values 0 and 1")
    if not set(table["sex"]).issubset({0, 1}):
        raise ValueError("Sex must use binary values 0 and 1")

    subject_number = table["subject_id"].str.removeprefix("sub-").astype(int)
    return table.iloc[np.argsort(subject_number)].reset_index(drop=True)


def load_subject_table(manifest_path: Path, metadata_path: Path) -> pd.DataFrame:
    """Load and validate the subject table and every required map path."""

    table = prepare_subject_table(
        pd.read_csv(manifest_path),
        pd.read_excel(metadata_path, usecols=["sub_id", "dx", "sex"]),
    )
    for column in ("affiliation_map", "power_map"):
        missing = [path for path in table[column] if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {column}: {missing[0]}")
    return table


def make_classifier(table: pd.DataFrame) -> tuple[ClassificationSearchlight, np.ndarray]:
    """Build the fixed folds and standardized linear classifier."""

    labels = table["dx"].to_numpy()
    folds = make_stratified_folds(
        labels,
        balance_by=table["sex"].to_numpy(),
        n_splits=5,
        random_state=RANDOM_STATE,
    )
    estimator = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10_000,
            random_state=RANDOM_STATE,
        ),
    )
    classifier = ClassificationSearchlight(
        labels=labels,
        folds=folds,
        estimator=estimator,
        score="balanced_accuracy",
    )

    # Save a one-based fold number for every subject for exact reproducibility.
    fold_number = np.empty(len(table), dtype=int)
    for number, (_, test) in enumerate(folds, start=1):
        fold_number[test] = number
    return classifier, fold_number


def selected_models(choice: str) -> list[str]:
    """Return models in the memory-efficient execution order."""

    if choice == "both":
        return ["two_channel", "contrast"]
    return [choice]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--radius", type=int, default=3, help="Ball radius in voxels")
    parser.add_argument("--min-active", type=float, default=0.10)
    parser.add_argument(
        "--model",
        choices=("both", "contrast", "two_channel"),
        default="both",
    )
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=None,
        help="Use inputs smoothed with this FWHM in millimeters, such as 4.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the manifest selected from --smoothing-fwhm.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check subjects, maps, labels, and folds without running BrainIAK",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = smoothing_suffix(args.smoothing_fwhm)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.radius < 1:
        raise ValueError("--radius must be positive")
    if not 0 <= args.min_active <= 1:
        raise ValueError("--min-active must be between 0 and 1")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    manifest = args.manifest or INPUT_DIR / f"manifest{suffix}.csv"
    table = load_subject_table(manifest, args.metadata)
    classifier, fold_number = make_classifier(table)
    models = selected_models(args.model)

    counts = table.groupby(["dx", "sex"]).size()
    print(f"Subjects: {len(table)}", flush=True)
    print("Diagnosis-by-sex counts:", flush=True)
    print(counts.to_string(), flush=True)
    model_folders = {name: f"{MODEL_FOLDERS[name]}{suffix}" for name in models}
    print(f"Smoothing FWHM: {args.smoothing_fwhm}", flush=True)
    print(f"Models: {', '.join(model_folders.values())}", flush=True)
    if args.validate_only:
        print("Validation complete; no maps were written.", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_output = table.copy()
    subject_output.insert(1, "fold", fold_number)
    subject_output = subject_output[
        [
            "subject_id",
            "fold",
            "dx",
            "sex",
            "n_affiliation_trials",
            "n_power_trials",
            "affiliation_map",
            "power_map",
        ]
    ]
    subject_output.to_csv(
        args.output_dir / f"subjects_and_folds{suffix}.csv",
        index=False,
    )
    configuration = {
        "models": list(model_folders.values()),
        "n_subjects": len(table),
        "n_folds": len(classifier.folds),
        "fold_stratification": ["diagnosis", "sex"],
        "classifier": "StandardScaler + LinearSVC(C=1, class_weight='balanced')",
        "score": classifier.score,
        "radius_voxels": args.radius,
        "shape": "ball",
        "min_active_proportion": args.min_active,
        "workers": args.workers,
        "random_state": RANDOM_STATE,
        "smoothing_fwhm": args.smoothing_fwhm,
        "manifest": str(manifest),
        "mask": str(args.mask),
    }
    with (args.output_dir / f"analysis{suffix}.json").open("w") as stream:
        json.dump(configuration, stream, indent=2)

    pending = []
    for model in models:
        output_name = MAP_NAME.replace(".nii.gz", f"{suffix}.nii.gz")
        output = args.output_dir / model_folders[model] / output_name
        if output.exists() and not args.overwrite:
            print(f"Reusing existing map: {output}", flush=True)
        else:
            pending.append((model, output))
    if not pending:
        return

    # Subject order is identical in both stacks and in the label/fold arrays.
    affiliation, reference = load_image_stack(table["affiliation_map"].tolist())
    power, _ = load_image_stack(
        table["power_map"].tolist(),
        reference_img=reference,
    )

    for model, output in pending:
        if model == "two_channel":
            feature_stacks = [affiliation, power]
        else:
            # The two-channel model runs first when both are requested.  Reuse
            # the affiliation allocation for the contrast to avoid a third
            # approximately 290 MB whole-brain subject stack.
            affiliation -= power
            feature_stacks = [affiliation]

        print(f"Running {model_folders[model]}...", flush=True)
        started = perf_counter()
        result = classifier.run(
            feature_stacks=feature_stacks,
            reference_img=reference,
            analysis_mask_img=args.mask,
            radius_voxels=args.radius,
            shape="ball",
            min_active_proportion=args.min_active,
            pool_size=args.workers,
        )
        result.save_map(output)
        elapsed = perf_counter() - started
        centers = int(np.isfinite(result.values).sum())
        print(
            f"Saved {output} ({centers:,} centers; {elapsed / 60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
