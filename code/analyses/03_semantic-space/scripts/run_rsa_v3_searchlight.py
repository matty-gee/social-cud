#!/usr/bin/env python
"""Run the final semantic model from ``rsa_roi_v3`` as a searchlight.

The analysis is the notebook's terminal three-predictor semantic model:

    neural dissimilarity ~ nuisance RDVs
                           + previous semantic direction
                           + current semantic choice
                           + current-minus-history semantic direction

By default, every subject shared by ``subject_data_Tavares.pkl`` and the LSS
GLM directory is run sequentially.  Within each subject, BrainIAK distributes
searchlight centers over ``--workers`` local processes.

Example from this directory:

    conda run -n brainiak python run_rsa_v3_searchlight.py \
        --subject 18001 --workers 12

Omit ``--subject`` to run every available subject.  Use ``--validate-only``
first to check trial alignment and model construction without writing maps.
"""

from __future__ import annotations

# The worker pool is the source of parallelism.  Giving every worker its own
# multithreaded BLAS pool oversubscribes laptop cores and is usually slower.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
from collections import OrderedDict
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import re
import sys
from typing import Mapping

import nibabel as nib
import numpy as np
import pandas as pd

# Resolve the shared engine from this file so the relocated runner works from
# any current working directory.
SEARCHLIGHT_DIR = Path(__file__).resolve().parents[3] / "searchlights"
sys.path.insert(0, str(SEARCHLIGHT_DIR))

from rsa_searchlight import RSASearchlight  # noqa: E402


# ---------------------------------------------------------------------------
# Analysis choices copied from rsa_roi_v3
# ---------------------------------------------------------------------------

ANALYSIS_NAME = "rsa_roi_v3_semantic_full"

LSS_ROOT = Path(
    "/Users/matty_gee/Desktop/Social/SocialCUD/results/glms/lss_decision"
)
DEFAULT_GLM_DIR = LSS_ROOT / "glms"
DEFAULT_SUBJECT_DATA = LSS_ROOT / "subject_data_Tavares.pkl"
DEFAULT_MASK = Path(
    "/Users/matty_gee/Desktop/Social/SocialCUD/masks/ROIs/GM.nii.gz"
)
DEFAULT_OUTPUT_ROOT = LSS_ROOT / "searchlights" / "searchlight-maps"

NUISANCE_NAMES = (
    "time",
    "time_sq",
    "dimension",
    "char_1",
    "char_2",
    "char_3",
    "char_4",
    "char_5",
    "character_decision_num",
    "reaction_time",
)

FOCAL_NAMES = (
    "semantic_previous_direction",
    "semantic_choice",
    "semantic_choice_update_direction",
)

# Only the terminal full model is needed here.  Adding the whole subset ladder
# would produce incremental-R2 maps too, but would not change these five maps.
FULL_MODEL = OrderedDict({"full": FOCAL_NAMES})
OUTPUTS = ("r2", "adjusted_r2", "beta")


# ---------------------------------------------------------------------------
# Exact RDVs needed by the final rsa_roi_v3 model
# ---------------------------------------------------------------------------

def _first_column(frame: pd.DataFrame, candidates, label: str) -> str:
    column = next((name for name in candidates if name in frame.columns), None)
    if column is None:
        raise KeyError(f"{label} requires one of these columns: {candidates}")
    return column


def _numeric_column(frame: pd.DataFrame, candidates, label: str) -> np.ndarray:
    column = _first_column(frame, candidates, label)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def _pair_helpers(n_trials: int):
    """Return condensed-order pair constructors shared by all RDVs."""
    first, second = np.triu_indices(n_trials, k=1)

    def pair_absolute(values):
        values = np.asarray(values, float).reshape(-1)
        if values.size != n_trials:
            raise ValueError(f"Expected {n_trials} values; got {values.size}.")
        result = np.abs(values[first] - values[second])
        valid = np.isfinite(values[first]) & np.isfinite(values[second])
        result[~valid] = np.nan
        return result

    def pair_category(values):
        codes, _ = pd.factorize(pd.Series(values), sort=True)
        codes = codes.astype(float)
        codes[codes < 0] = np.nan
        result = (codes[first] != codes[second]).astype(float)
        valid = np.isfinite(codes[first]) & np.isfinite(codes[second])
        result[~valid] = np.nan
        return result

    def pair_cosine(vectors, eps: float = 1e-12):
        vectors = np.asarray(vectors, float)
        if vectors.ndim != 2 or vectors.shape[0] != n_trials:
            raise ValueError(
                f"Expected vectors with shape ({n_trials}, D); got {vectors.shape}."
            )
        norms = np.linalg.norm(vectors, axis=1)
        valid_rows = (
            np.isfinite(vectors).all(axis=1)
            & np.isfinite(norms)
            & (norms > eps)
        )
        units = np.full_like(vectors, np.nan, dtype=float)
        units[valid_rows] = vectors[valid_rows] / norms[valid_rows, None]
        valid_pairs = valid_rows[first] & valid_rows[second]
        result = np.full(first.size, np.nan, dtype=float)
        similarities = np.sum(
            units[first[valid_pairs]] * units[second[valid_pairs]], axis=1
        )
        result[valid_pairs] = 1.0 - np.clip(similarities, -1.0, 1.0)
        return result

    def both_role(roles, role):
        roles = np.asarray(roles, float)
        valid = np.isfinite(roles[first]) & np.isfinite(roles[second])
        result = (
            (roles[first] == role) & (roles[second] == role)
        ).astype(float)
        result[~valid] = np.nan
        return result

    return pair_absolute, pair_category, pair_cosine, both_role


def _semantic_history(
    choice_embeddings: np.ndarray,
    roles: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Past-only, equal-weighted, same-character history from rsa_roi_v3."""
    choice = np.asarray(choice_embeddings, float)
    if choice.ndim != 2 or choice.shape[0] != len(roles):
        raise ValueError(
            "embeddings['choice'] must be a trials-by-features matrix aligned "
            "with behavior."
        )

    norms = np.linalg.norm(choice, axis=1)
    valid = (
        np.isfinite(choice).all(axis=1)
        & np.isfinite(norms)
        & (norms > eps)
    )
    choice_unit = np.full_like(choice, np.nan, dtype=float)
    choice_unit[valid] = choice[valid] / norms[valid, None]

    running_mean = np.full_like(choice_unit, np.nan)
    update = np.full_like(choice_unit, np.nan)

    # The first encounter with each character has no prior history and remains
    # NaN.  RSASearchlight therefore excludes all pairs involving those trials
    # once, before any spatial computation begins.
    for role in pd.unique(pd.Series(roles).dropna()):
        role_trials = np.flatnonzero(roles == role)
        for position in range(1, len(role_trials)):
            trial = role_trials[position]
            if not valid[trial]:
                continue
            previous_trials = role_trials[:position]
            previous_trials = previous_trials[valid[previous_trials]]
            if previous_trials.size == 0:
                continue
            mean = choice_unit[previous_trials].mean(axis=0)
            running_mean[trial] = mean
            update[trial] = choice_unit[trial] - mean

    return running_mean, update


def make_full_model_rdvs(subject: Mapping) -> tuple[OrderedDict, OrderedDict]:
    """Construct exactly the nuisance and focal RDVs used by the full model."""
    behavior = subject.get("behavior")
    embeddings = subject.get("embeddings")
    if not isinstance(behavior, pd.DataFrame):
        raise TypeError("Subject behavior must be a pandas DataFrame.")
    if not isinstance(embeddings, Mapping) or "choice" not in embeddings:
        raise KeyError("Subject embeddings must contain embeddings['choice'].")

    behavior = behavior.reset_index(drop=True)
    n_trials = len(behavior)
    if n_trials < 2:
        raise ValueError("At least two trials are required.")

    role_column = _first_column(
        behavior,
        ("character_role_num", "char_role_num"),
        "character identity",
    )
    roles = pd.to_numeric(behavior[role_column], errors="coerce").to_numpy(float)

    # rsa_roi_v3 prefers onset, then decision/trial order, and falls back to a
    # simple sequence only when none of those columns exists.
    time_column = next(
        (name for name in ("onset", "decision_num", "trial_num") if name in behavior),
        None,
    )
    time = (
        pd.to_numeric(behavior[time_column], errors="coerce").to_numpy(float)
        if time_column is not None
        else np.arange(n_trials, dtype=float)
    )

    if "dimension" in behavior:
        dimension = behavior["dimension"].to_numpy()
    else:
        affiliation = _numeric_column(
            behavior, ("affil_decision",), "affiliation decision"
        )
        power = _numeric_column(behavior, ("power_decision",), "power decision")
        dimension = np.full(n_trials, np.nan, dtype=object)
        dimension[np.isfinite(affiliation) & (np.abs(affiliation) > 1e-12)] = (
            "affiliation"
        )
        dimension[np.isfinite(power) & (np.abs(power) > 1e-12)] = "power"

    if "character_decision_num" in behavior:
        character_decision_num = pd.to_numeric(
            behavior["character_decision_num"], errors="coerce"
        ).to_numpy(float)
    else:
        character_decision_num = np.full(n_trials, np.nan)
    if not np.isfinite(character_decision_num).any():
        for role in pd.unique(pd.Series(roles).dropna()):
            indices = np.flatnonzero(roles == role)
            character_decision_num[indices] = np.arange(1, len(indices) + 1)

    reaction_time = _numeric_column(
        behavior,
        ("reaction_time", "rt", "response_time"),
        "reaction time",
    )
    choice = np.asarray(embeddings["choice"], float)
    running_mean, update = _semantic_history(choice, roles)

    pair_absolute, pair_category, pair_cosine, both_role = _pair_helpers(n_trials)
    nuisance = OrderedDict(
        time=pair_absolute(time),
        time_sq=pair_absolute(time) ** 2,
        dimension=pair_category(dimension),
    )
    for role in range(1, 6):
        # Notebook coding: 0 only when both trials are the named character;
        # 1 otherwise. Binary columns are deliberately not z-scored.
        nuisance[f"char_{role}"] = 1.0 - both_role(roles, role)
    nuisance["character_decision_num"] = pair_absolute(character_decision_num)
    nuisance["reaction_time"] = pair_absolute(reaction_time)

    focal = OrderedDict(
        semantic_previous_direction=pair_cosine(running_mean),
        semantic_choice=pair_cosine(choice),
        semantic_choice_update_direction=pair_cosine(update),
    )
    return nuisance, focal


def prepare_subject_model(subject: Mapping) -> RSASearchlight:
    """Precompute this subject's fixed full-model projections once."""
    nuisance, focal = make_full_model_rdvs(subject)
    return RSASearchlight.from_rdvs(
        nuisance_rdvs=nuisance,
        focal_rdvs=focal,
        models=FULL_MODEL,
        comparisons=None,
        outputs=OUTPUTS,
        beta_models="full",
        zscore_x=True,
        zscore_binary=False,
        zscore_y=True,
        neural_rdm_method="matrix",
        neural_rdm_dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Files, validation, execution, and provenance
# ---------------------------------------------------------------------------

def normalize_subject_id(value) -> str:
    value = str(value).strip()
    value = re.sub(r"^sub-", "", value, flags=re.IGNORECASE)
    if not value:
        raise ValueError("Subject ID cannot be empty.")
    return f"sub-{value}"


def load_subjects(path: Path) -> dict[str, Mapping]:
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must contain a subject mapping.")

    subjects = {}
    for key, subject in raw.items():
        if not isinstance(subject, Mapping):
            continue
        subject_id = normalize_subject_id(subject.get("sub_id", key))
        if subject_id in subjects:
            raise ValueError(f"Duplicate subject metadata for {subject_id}.")
        subjects[subject_id] = subject
    return subjects


def find_beta_images(glm_dir: Path) -> dict[str, Path]:
    images = {}
    pattern = re.compile(r"^(sub-[^_]+)_decision_trials_beta\.nii\.gz$")
    for path in sorted(glm_dir.glob("sub-*_decision_trials_beta.nii.gz")):
        match = pattern.match(path.name)
        if match:
            images[normalize_subject_id(match.group(1))] = path
    return images


def _subject_sort_key(subject_id: str):
    label = subject_id.removeprefix("sub-")
    return (0, int(label)) if label.isdigit() else (1, label)


def select_subjects(
    requested: list[str],
    metadata: Mapping[str, Mapping],
    beta_images: Mapping[str, Path],
) -> list[str]:
    available = set(metadata) & set(beta_images)
    if requested:
        selected = [normalize_subject_id(value) for value in requested]
        missing = sorted(set(selected) - available, key=_subject_sort_key)
        if missing:
            raise FileNotFoundError(
                "Subjects missing metadata or an LSS beta image: " + ", ".join(missing)
            )
        return list(dict.fromkeys(selected))
    return sorted(available, key=_subject_sort_key)


def validate_subject(subject_id: str, subject: Mapping, beta_path: Path):
    behavior = subject.get("behavior")
    if not isinstance(behavior, pd.DataFrame):
        raise TypeError(f"{subject_id}: behavior is not a DataFrame.")
    beta_image = nib.load(str(beta_path))
    if len(beta_image.shape) != 4:
        raise ValueError(f"{subject_id}: beta image is not 4D: {beta_image.shape}")
    if beta_image.shape[-1] != len(behavior):
        raise ValueError(
            f"{subject_id}: {beta_image.shape[-1]} beta trials != "
            f"{len(behavior)} behavior trials."
        )
    model = prepare_subject_model(subject)
    return model, beta_image.shape


def analysis_config(args) -> dict:
    return {
        "analysis_name": args.analysis_name,
        "model": "rsa_roi_v3 terminal semantic full model",
        "nuisance_rdvs": list(NUISANCE_NAMES),
        "focal_rdvs": list(FOCAL_NAMES),
        "outputs": list(OUTPUTS),
        "neural_rdm": "correlation distance, float32 matrix implementation",
        "zscore_x": True,
        "zscore_binary": False,
        "zscore_y": True,
        "history_window": None,
        "history_tau": None,
        "searchlight_shape": "true Ball",
        "radius_voxels": args.radius_voxels,
        "min_active_proportion_of_ball": args.min_active_proportion,
        "mask": str(args.mask),
        "glm_dir": str(args.glm_dir),
        "subject_data": str(args.subject_data),
    }


def write_json_atomic(path: Path, payload: Mapping):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ensure_analysis_config(output_dir: Path, config: Mapping):
    """Prevent scientifically different runs from sharing one output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis_config.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != dict(config):
            raise ValueError(
                f"{path} describes different analysis settings. Use a new "
                "--analysis-name rather than mixing maps."
            )
    else:
        write_json_atomic(path, config)


def run_subject(
    *,
    subject_id: str,
    subject: Mapping,
    beta_path: Path,
    model: RSASearchlight,
    output_dir: Path,
    mask: Path,
    radius_voxels: int,
    min_active_proportion: float,
    workers: int,
    overwrite: bool,
):
    marker = output_dir / f"{subject_id}_complete.json"
    if marker.exists() and not overwrite:
        previous = json.loads(marker.read_text())
        expected = [Path(path) for path in previous.get("maps", [])]
        if expected and all(path.exists() for path in expected):
            print(f"Skipping {subject_id}: complete outputs already exist.", flush=True)
            return "skipped"

    print(
        f"Running {subject_id}: {model.n_trials} trials, {model.n_pairs} pairs, "
        f"{workers} workers",
        flush=True,
    )
    result = model.run(
        beta_img=beta_path,
        analysis_mask_img=mask,
        radius_voxels=radius_voxels,
        min_active_proportion=min_active_proportion,
        pool_size=workers,
    )
    saved = result.save_maps(output_dir, subject_id)
    completed_centers = int(np.isfinite(result.values[..., 0]).sum())
    payload = {
        "subject_id": subject_id,
        "beta_image": str(beta_path),
        "n_trials": model.n_trials,
        "n_pairs": model.n_pairs,
        "completed_centers": completed_centers,
        "output_names": list(result.output_names),
        "maps": [str(path) for path in saved],
        "timings_seconds": result.timings,
    }
    # The marker is written last, so an interrupted subject is never mistaken
    # for a complete one on the next invocation.
    write_json_atomic(marker, payload)
    print(
        f"Finished {subject_id}: {completed_centers} centers, "
        f"{result.timings['total']:.1f} s, {len(saved)} maps",
        flush=True,
    )
    return "completed"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject ID to run; repeat for multiple subjects. Default: all.",
    )
    parser.add_argument("--analysis-name", default=ANALYSIS_NAME)
    parser.add_argument("--glm-dir", type=Path, default=DEFAULT_GLM_DIR)
    parser.add_argument("--subject-data", type=Path, default=DEFAULT_SUBJECT_DATA)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--radius-voxels",
        type=int,
        default=5,
        help="Ball radius in voxels (default: 5, approximately 10 mm here).",
    )
    parser.add_argument("--min-active-proportion", type=float, default=0.10)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and subject models without running or writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun subjects whose complete outputs already exist.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.analysis_name):
        raise ValueError(
            "--analysis-name may contain only letters, numbers, dot, dash, and underscore."
        )
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.radius_voxels < 0:
        raise ValueError("--radius-voxels must be non-negative.")
    if not 0 <= args.min_active_proportion < 1:
        raise ValueError("--min-active-proportion must be in [0, 1).")
    for path, label in (
        (args.glm_dir, "GLM directory"),
        (args.subject_data, "subject-data pickle"),
        (args.mask, "analysis mask"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    mp.set_start_method("spawn", force=True)

    metadata = load_subjects(args.subject_data)
    beta_images = find_beta_images(args.glm_dir)
    selected = select_subjects(args.subject, metadata, beta_images)
    if not selected:
        raise RuntimeError("No subjects have both metadata and an LSS beta image.")

    print(
        f"Selected {len(selected)} subject(s); metadata={len(metadata)}, "
        f"LSS images={len(beta_images)}.",
        flush=True,
    )

    if args.validate_only:
        for subject_id in selected:
            model, shape = validate_subject(
                subject_id, metadata[subject_id], beta_images[subject_id]
            )
            print(
                f"OK {subject_id}: beta shape={shape}, "
                f"valid pairs={model.n_pairs}, outputs={model.n_outputs}",
                flush=True,
            )
        print("Validation complete; no outputs were written.", flush=True)
        return

    output_dir = args.output_root / args.analysis_name
    ensure_analysis_config(output_dir, analysis_config(args))
    completed = skipped = 0
    for subject_id in selected:
        model, _ = validate_subject(
            subject_id, metadata[subject_id], beta_images[subject_id]
        )
        status = run_subject(
            subject_id=subject_id,
            subject=metadata[subject_id],
            beta_path=beta_images[subject_id],
            model=model,
            output_dir=output_dir,
            mask=args.mask,
            radius_voxels=args.radius_voxels,
            min_active_proportion=args.min_active_proportion,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        completed += status == "completed"
        skipped += status == "skipped"

    print(
        f"Done: completed={completed}, skipped={skipped}, outputs={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
