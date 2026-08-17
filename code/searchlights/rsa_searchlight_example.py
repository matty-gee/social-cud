"""Runnable toy example for ``RSASearchlight``.

From this directory, run:

    conda run -n brainiak python rsa_searchlight_example.py --workers 4

The script creates synthetic trialwise beta data, runs a parallelized Ball
searchlight, and writes named NIfTI maps to a temporary directory.  Pass
``--output-dir PATH`` to keep the outputs somewhere specific.
"""

from __future__ import annotations

# Prevent each BrainIAK worker from starting its own multithreaded BLAS pool.
# These variables must be set before importing NumPy.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import multiprocessing as mp
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from rsa_searchlight import RSASearchlight


def make_toy_inputs(output_dir: Path, seed: int = 20260817):
    """Create a small beta image, GM-like mask, and behavioral RDVs."""
    # This setup-only import stays out of the spawned searchlight workers.
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(seed)
    n_trials = 18
    shape = (11, 11, 11)

    # Two trial features will be represented in different spatial gradients.
    trial_axis = np.linspace(-1.0, 1.0, n_trials)
    feature_a = np.sin(np.linspace(0, 2 * np.pi, n_trials, endpoint=False))
    feature_b = np.cos(np.linspace(0, 3 * np.pi, n_trials, endpoint=False))

    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    centered_grid = grid - center[:, None, None, None]
    radius = np.sqrt(np.sum(centered_grid**2, axis=0))

    # This acts like a small grey-matter analysis mask.
    analysis_mask = radius <= 4.5

    # Spatially varying loadings make correlation distances between trial
    # patterns sensitive to feature_a and feature_b.
    loading_a = centered_grid[0] / 4.5
    loading_b = centered_grid[1] / 4.5
    data = rng.normal(0, 0.45, size=shape + (n_trials,)).astype(np.float32)
    data += loading_a[..., None] * feature_a
    data += loading_b[..., None] * feature_b

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    beta_path = output_dir / "toy_trial_betas.nii.gz"
    mask_path = output_dir / "toy_analysis_mask.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), beta_path)
    nib.save(nib.Nifti1Image(analysis_mask.astype(np.uint8), affine), mask_path)

    # Every RDV uses SciPy condensed order: one value for each trial pair.
    nuisance_rdvs = {
        "time": pdist(trial_axis[:, None], metric="euclidean"),
    }
    focal_rdvs = {
        "feature_a": pdist(feature_a[:, None], metric="euclidean"),
        "feature_b": pdist(feature_b[:, None], metric="euclidean"),
    }
    return beta_path, mask_path, nuisance_rdvs, focal_rdvs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of BrainIAK worker processes (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for toy inputs and result maps; defaults to a new temp directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive.")

    # macOS uses spawn.  Keeping all executable work inside main() prevents
    # child workers from rerunning the script while importing it.
    mp.set_start_method("spawn", force=True)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="socialcud-rsa-toy-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create/load one subject's beta image, mask, and behavioral RDVs.
    beta_path, mask_path, nuisance_rdvs, focal_rdvs = make_toy_inputs(output_dir)

    # 2. Define the model family.  With two focal predictors this creates the
    # nuisance model, both one-predictor models, the full model, and all valid
    # one-predictor nested comparisons.
    models, comparisons = RSASearchlight.make_subset_models(
        {"a": "feature_a", "b": "feature_b"}
    )

    # 3. Prepare everything that is constant across searchlight centers.
    rsa = RSASearchlight.from_rdvs(
        nuisance_rdvs=nuisance_rdvs,
        focal_rdvs=focal_rdvs,
        models=models,
        comparisons=comparisons,
        outputs=("r2", "delta_r2", "beta"),
        beta_models="full",
        neural_rdm_method="matrix",
        neural_rdm_dtype=np.float32,
    )

    # 4. Run one corrected Ball searchlight.  BrainIAK distributes spatial
    # blocks across args.workers processes; the prepared RSA object is reused
    # at every center.
    result = rsa.run(
        beta_img=beta_path,
        analysis_mask_img=mask_path,
        radius_voxels=1,
        min_active_proportion=0.50,
        pool_size=args.workers,
    )

    # 5. Save one named 3D map for every requested statistic.
    saved_maps = result.save_maps(output_dir / "maps", "toy")

    completed_centers = int(np.isfinite(result.values[..., 0]).sum())
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Completed centers: {completed_centers}")
    print(f"Outputs ({len(result.output_names)}):")
    for name in result.output_names:
        print(f"  - {name}")
    print("Timings (seconds):")
    for name, seconds in result.timings.items():
        print(f"  {name}: {seconds:.3f}")
    print(f"Saved {len(saved_maps)} maps to: {output_dir / 'maps'}")


if __name__ == "__main__":
    main()
