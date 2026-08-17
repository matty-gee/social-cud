#!/usr/bin/env python
"""Average decision-trial LSS betas within affiliation and power.

Run from this directory with:

    conda run -n brainiak python make_dimension_average_maps.py
    conda run -n brainiak python make_dimension_average_maps.py \
        --smoothing-fwhm 4

The output structure is:

    results/glms/lss_decision/
        voxelwise/
            dimension_average/
                grey_matter_mask.nii.gz
                manifest.csv
                maps/
                    sub-18001_affiliation.nii.gz
                    sub-18001_power.nii.gz
                    sub-18001_affiliation_smoothed4fwhm.nii.gz
                    sub-18001_power_smoothed4fwhm.nii.gz
                    ...

Only responded affiliation and power trials are averaged. Neutral trials and
missed responses are excluded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import re

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from nilearn.image import smooth_img
import pandas as pd


PROJECT_DATA = Path("/Users/matty_gee/Desktop/Social/SocialCUD")
LSS_ROOT = PROJECT_DATA / "results" / "glms" / "lss_decision"
GLM_DIR = LSS_ROOT / "glms"
SUBJECT_DATA_FILE = LSS_ROOT / "subject_data_Tavares.pkl" # TODO just load behavior file directly
GREY_MATTER_MASK_CANDIDATES = (
    PROJECT_DATA / "masks" / "ROIs" / "GM.nii.gz",
    PROJECT_DATA / "results" / "masks" / "ROIs" / "GM.nii.gz",
)

ANALYSIS_DIR = LSS_ROOT / "voxelwise" / "dimension_average"
MAP_DIR = ANALYSIS_DIR / "maps"


def smoothing_suffix(fwhm: float | None) -> str:
    """Return the filename suffix for one optional smoothing kernel."""
    if fwhm is None:
        return ""
    if fwhm <= 0:
        raise ValueError("Smoothing FWHM must be positive")
    label = f"{fwhm:g}".replace(".", "p")
    return f"_smoothed{label}fwhm"


def normalize_subject_id(value) -> str:
    """Return IDs in the filename form ``sub-18001``."""
    value = re.sub(r"^sub-", "", str(value).strip(), flags=re.IGNORECASE)
    return f"sub-{value}"


def load_subject_data() -> dict[str, dict]:
    """Load behavior and index it by normalized subject ID."""
    with SUBJECT_DATA_FILE.open("rb") as stream:
        raw = pickle.load(stream)
    return {
        normalize_subject_id(subject.get("sub_id", key)): subject
        for key, subject in raw.items()
    }


def find_beta_images() -> dict[str, Path]:
    """Find the flat 4D LSS files requested for this analysis."""
    return {
        normalize_subject_id(path.name.split("_", 1)[0]): path
        for path in GLM_DIR.glob("sub-*_decision_trials_beta.nii.gz")
    }


def responded_mask(behavior: pd.DataFrame) -> np.ndarray:
    """Convert the stored response column to a reliable boolean mask."""
    if "responded" not in behavior:
        return np.ones(len(behavior), dtype=bool)
    values = behavior["responded"]
    if values.dtype == object:
        return values.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()
    return values.fillna(False).astype(bool).to_numpy()


def condition_masks(behavior: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned, responded affiliation and power trial masks."""
    behavior = behavior.reset_index(drop=True)
    if "dimension" not in behavior:
        raise KeyError("Behavior is missing the 'dimension' column.")

    # The LSS merge was written in decision_num order. This assertion protects
    # against silently averaging the wrong 4D volumes if behavior is reordered.
    if "decision_num" in behavior:
        decision_num = pd.to_numeric(behavior["decision_num"], errors="coerce")
        expected = np.arange(1, len(behavior) + 1)
        if not np.array_equal(decision_num.to_numpy(), expected):
            raise ValueError("Behavior rows are not in consecutive decision_num order.")

    dimension = behavior["dimension"].astype(str).str.lower()
    responded = responded_mask(behavior)
    affiliation = dimension.isin({"affil", "affiliation"}).to_numpy() & responded
    power = dimension.eq("power").to_numpy() & responded
    if not affiliation.any() or not power.any():
        raise ValueError("No responded trials remain for one or both dimensions.")
    return affiliation, power


def make_group_mask(reference: nib.Nifti1Image) -> np.ndarray:
    """Resample the project grey-matter mask once to the common LSS grid."""
    mask_path = next(
        (path for path in GREY_MATTER_MASK_CANDIDATES if path.exists()),
        None,
    )
    if mask_path is None:
        searched = "\n".join(str(path) for path in GREY_MATTER_MASK_CANDIDATES)
        raise FileNotFoundError(f"Could not find the grey-matter mask in:\n{searched}")
    print(f"Grey-matter mask: {mask_path}", flush=True)
    source = nib.load(str(mask_path))
    resampled = resample_from_to(
        source,
        (reference.shape[:3], reference.affine),
        order=0,
    )
    mask = np.asarray(resampled.dataobj) > 0

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    mask_image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header)
    nib.save(mask_image, ANALYSIS_DIR / "grey_matter_mask.nii.gz")
    return mask


def save_map(values: np.ndarray, reference: nib.Nifti1Image, path: Path):
    """Save a float32 map, using a temporary file to avoid partial outputs."""
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(values.astype(np.float32, copy=False), reference.affine, header)
    temporary = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(image, temporary)
    temporary.replace(path)


def smooth_condition_map(
    values: np.ndarray,
    reference: nib.Nifti1Image,
    fwhm: float,
) -> np.ndarray:
    """Smooth one full-volume condition mean before gray-matter masking."""
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(values.astype(np.float32, copy=False), reference.affine, header)
    return smooth_img(image, fwhm=fwhm).get_fdata(dtype=np.float32)


def make_subject_maps(
    subject_id: str,
    subject: dict,
    beta_path: Path,
    grey_matter: np.ndarray,
    reference: nib.Nifti1Image,
    overwrite: bool,
    smoothing_fwhm: float | None,
) -> dict:
    """Average one subject's LSS images and save the two condition maps."""
    suffix = smoothing_suffix(smoothing_fwhm)
    affiliation_path = MAP_DIR / f"{subject_id}_affiliation{suffix}.nii.gz"
    power_path = MAP_DIR / f"{subject_id}_power{suffix}.nii.gz"
    behavior = subject["behavior"].reset_index(drop=True)
    affiliation_trials, power_trials = condition_masks(behavior)

    if not overwrite and affiliation_path.exists() and power_path.exists():
        status = "reused"
    else:
        image = nib.load(str(beta_path))
        if image.shape[:3] != reference.shape[:3] or not np.allclose(
            image.affine, reference.affine
        ):
            raise ValueError(f"{subject_id}: LSS image is not on the common grid.")
        if image.shape[-1] != len(behavior):
            raise ValueError(
                f"{subject_id}: {image.shape[-1]} LSS volumes but "
                f"{len(behavior)} behavior rows."
            )

        # Average first because averaging and Gaussian smoothing are linear.
        # Smoothing only two condition means is equivalent to smoothing every
        # selected trial, while avoiding dozens of redundant operations.
        data = np.asarray(image.dataobj, dtype=np.float32)
        affiliation = data[..., affiliation_trials].mean(axis=-1, dtype=np.float32)
        power = data[..., power_trials].mean(axis=-1, dtype=np.float32)
        if smoothing_fwhm is not None:
            # Smooth the full volume before masking so zeros outside the mask
            # do not attenuate values along the gray-matter boundary.
            affiliation = smooth_condition_map(affiliation, image, smoothing_fwhm)
            power = smooth_condition_map(power, image, smoothing_fwhm)

        affiliation[~grey_matter] = 0
        power[~grey_matter] = 0
        if not np.isfinite(affiliation[grey_matter]).all():
            raise ValueError(f"{subject_id}: non-finite affiliation values in grey matter.")
        if not np.isfinite(power[grey_matter]).all():
            raise ValueError(f"{subject_id}: non-finite power values in grey matter.")

        save_map(affiliation, image, affiliation_path)
        save_map(power, image, power_path)
        status = "created"

    return {
        "subject_id": subject_id,
        "n_affiliation_trials": int(affiliation_trials.sum()),
        "n_power_trials": int(power_trials.sum()),
        "smoothing_fwhm": smoothing_fwhm,
        "affiliation_map": str(affiliation_path),
        "power_map": str(power_path),
        "status": status,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject to generate; repeat as needed. Default: all available subjects.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace maps that already exist.",
    )
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=None,
        help="Optional spatial smoothing kernel in millimeters, such as 4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = smoothing_suffix(args.smoothing_fwhm)
    subjects = load_subject_data()
    beta_images = find_beta_images()
    available = set(subjects) & set(beta_images)
    selected = (
        [normalize_subject_id(value) for value in args.subject]
        if args.subject
        else sorted(available, key=lambda value: int(value.removeprefix("sub-")))
    )
    missing = sorted(set(selected) - available)
    if missing:
        raise FileNotFoundError(
            "Missing subject data or LSS image: " + ", ".join(missing)
        )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(beta_images[selected[0]]))
    grey_matter = make_group_mask(reference)

    rows = []
    for index, subject_id in enumerate(selected, start=1):
        row = make_subject_maps(
            subject_id,
            subjects[subject_id],
            beta_images[subject_id],
            grey_matter,
            reference,
            args.overwrite,
            args.smoothing_fwhm,
        )
        rows.append(row)
        print(
            f"[{index:02d}/{len(selected):02d}] {subject_id}: "
            f"affiliation={row['n_affiliation_trials']}, "
            f"power={row['n_power_trials']} ({row['status']})",
            flush=True,
        )

    manifest_path = ANALYSIS_DIR / f"manifest{suffix}.csv"
    temporary_manifest = ANALYSIS_DIR / f"manifest{suffix}.tmp.csv"
    with temporary_manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest_path)

    print(f"Saved {2 * len(rows)} maps to {MAP_DIR}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
