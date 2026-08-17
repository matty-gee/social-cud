"""Tests for the dimension classification runner's alignment choices."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[2]
SEARCHLIGHT_DIR = CODE_DIR / "searchlights"
RUNNER_DIR = CODE_DIR / "analyses" / "02_affiliation-power" / "scripts"
sys.path.insert(0, str(SEARCHLIGHT_DIR))
sys.path.insert(0, str(RUNNER_DIR))

from run_dimension_classification_searchlights import (  # noqa: E402
    make_classifier,
    normalize_subject_id,
    prepare_subject_table,
    selected_models,
    smoothing_suffix,
)


def test_normalize_subject_id_handles_numeric_and_prefixed_ids() -> None:
    assert normalize_subject_id(18001) == "sub-18001"
    assert normalize_subject_id(18001.0) == "sub-18001"
    assert normalize_subject_id("sub-18001") == "sub-18001"


def test_prepare_subject_table_aligns_metadata_and_sorts_subjects() -> None:
    subjects = [18005, 18001, 18003, 18002]
    manifest = pd.DataFrame(
        {
            "subject_id": [f"sub-{subject}" for subject in subjects],
            "n_affiliation_trials": [30] * 4,
            "n_power_trials": [30] * 4,
            "affiliation_map": [f"a-{subject}.nii.gz" for subject in subjects],
            "power_map": [f"p-{subject}.nii.gz" for subject in subjects],
        }
    )
    metadata = pd.DataFrame(
        {
            "sub_id": [18001, 18002, 18003, 18005],
            "dx": [0, 1, 0, 1],
            "sex": [1, 0, 0, 1],
        }
    )

    table = prepare_subject_table(manifest, metadata)

    assert table["subject_id"].tolist() == [
        "sub-18001",
        "sub-18002",
        "sub-18003",
        "sub-18005",
    ]
    assert table["dx"].tolist() == [0, 1, 0, 1]
    assert table.columns.tolist() == [
        "subject_id",
        "n_affiliation_trials",
        "n_power_trials",
        "affiliation_map",
        "power_map",
        "dx",
        "sex",
    ]


def test_classifier_uses_one_complete_fixed_fold_assignment() -> None:
    # Five members of each diagnosis-by-sex group support five joint folds.
    table = pd.DataFrame(
        {
            "dx": np.repeat([0, 0, 1, 1], 5),
            "sex": np.repeat([0, 1, 0, 1], 5),
        }
    )

    classifier, fold_number = make_classifier(table)

    assert classifier.n_subjects == 20
    assert set(fold_number) == {1, 2, 3, 4, 5}
    assert np.bincount(fold_number)[1:].tolist() == [4, 4, 4, 4, 4]


def test_both_models_use_memory_efficient_order() -> None:
    assert selected_models("both") == ["two_channel", "contrast"]


def test_smoothing_suffix_is_stable_and_filename_safe() -> None:
    assert smoothing_suffix(None) == ""
    assert smoothing_suffix(4) == "_smoothed4fwhm"
    assert smoothing_suffix(4.5) == "_smoothed4p5fwhm"
