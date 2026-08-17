"""Tests for smoothing the dimension-average feature maps."""

from pathlib import Path
import sys

import nibabel as nib
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "voxelwise"))

from make_dimension_average_maps import (  # noqa: E402
    smooth_condition_map,
    smoothing_suffix,
)


def test_smoothed_map_uses_requested_kernel_and_preserves_grid() -> None:
    values = np.zeros((7, 7, 7), dtype=np.float32)
    values[3, 3, 3] = 1
    reference = nib.Nifti1Image(values, np.diag([2.0, 2.0, 2.0, 1.0]))

    smoothed = smooth_condition_map(values, reference, fwhm=4)

    assert smoothed.shape == values.shape
    assert smoothed.dtype == np.float32
    assert 0 < smoothed[3, 3, 3] < 1
    assert smoothed[3, 3, 2] > 0
    np.testing.assert_allclose(smoothed.sum(), 1, rtol=0.02)


def test_feature_map_suffix_matches_classifier_suffix() -> None:
    assert smoothing_suffix(4) == "_smoothed4fwhm"
