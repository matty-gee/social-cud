"""Tests for the reusable classification-searchlight building blocks."""

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEARCHLIGHT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEARCHLIGHT_DIR))

from classifier_searchlight import (  # noqa: E402
    ClassificationResult,
    ClassificationSearchlight,
    load_image_stack,
    make_stratified_folds,
)


def _classifier(n_subjects: int = 20) -> ClassificationSearchlight:
    labels = np.tile([0, 1], n_subjects // 2)
    folds = make_stratified_folds(labels, n_splits=5, random_state=1)
    estimator = make_pipeline(StandardScaler(), LinearSVC(C=1.0))
    return ClassificationSearchlight(labels=labels, folds=folds, estimator=estimator)


def test_load_image_stack_preserves_subject_order_and_grid(tmp_path: Path) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    paths = []
    for subject, value in enumerate([3.0, 7.0, 11.0]):
        path = tmp_path / f"sub-{subject}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((3, 4, 5), value), affine), path)
        paths.append(path)

    stack, reference = load_image_stack(paths)

    assert stack.shape == (3, 4, 5, 3)
    assert stack.dtype == np.float32
    np.testing.assert_array_equal(stack[0, 0, 0], [3.0, 7.0, 11.0])
    np.testing.assert_array_equal(reference.affine, affine)


def test_load_image_stack_rejects_misaligned_images(tmp_path: Path) -> None:
    first = tmp_path / "first.nii.gz"
    second = tmp_path / "second.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)), first)
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.diag([2, 2, 2, 1])), second)

    with pytest.raises(ValueError, match="not aligned"):
        load_image_stack([first, second])


def test_stratified_folds_are_fixed_disjoint_partition() -> None:
    # Four equal diagnosis-by-sex groups allow five-fold joint stratification.
    labels = np.repeat([0, 0, 1, 1], 5)
    sex = np.repeat([0, 1, 0, 1], 5)
    folds = make_stratified_folds(labels, balance_by=sex, n_splits=5, random_state=9)

    tested = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(tested), np.arange(20))
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        assert set(zip(labels[test], sex[test])) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_score_features_finds_signal_without_fitting_original_estimator() -> None:
    classifier = _classifier()
    signal = np.where(classifier.labels == 1, 10.0, -10.0)
    features = np.column_stack([signal, signal * 0.5])

    score = classifier.score_features(features)

    assert score == pytest.approx(1.0)
    assert not hasattr(classifier.estimator.named_steps["linearsvc"], "coef_")


def test_classifier_rejects_folds_that_omit_training_subjects() -> None:
    labels = np.tile([0, 1], 5)
    incomplete_folds = [
        (np.arange(2, 9), np.asarray([0, 1])),
        (np.asarray([0, 1, 4, 5, 6, 7, 8, 9]), np.asarray([2, 3])),
        (np.asarray([0, 1, 2, 3, 6, 7, 8, 9]), np.asarray([4, 5])),
        (np.asarray([0, 1, 2, 3, 4, 5, 8, 9]), np.asarray([6, 7])),
        (np.arange(8), np.asarray([8, 9])),
    ]

    with pytest.raises(ValueError, match="do not cover all subjects"):
        ClassificationSearchlight(
            labels=labels,
            folds=incomplete_folds,
            estimator=LinearSVC(),
        )


def test_kernel_uses_true_mask_and_concatenates_channels() -> None:
    classifier = _classifier()
    first = np.arange(3 * 3 * 3 * 20).reshape(3, 3, 3, 20)
    second = first + 1000
    ball = np.zeros((3, 3, 3), dtype=bool)
    ball[1, 1, 1] = True
    ball[1, 1, 2] = True
    captured = []

    def record(features: np.ndarray) -> float:
        captured.append(features.copy())
        return 0.75

    classifier.score_features = record  # type: ignore[method-assign]
    score = classifier._kernel([first, second], ball, 1, classifier)

    expected = np.concatenate([first[ball, :].T, second[ball, :].T], axis=1)
    assert score == 0.75
    np.testing.assert_array_equal(captured[0], expected)
    assert captured[0].shape == (20, 4)


def test_result_saves_float32_nifti(tmp_path: Path) -> None:
    reference = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    result = ClassificationResult(
        values=np.full((3, 3, 3), 0.6),
        reference_img=reference,
        score_name="balanced_accuracy",
    )

    output = result.save_map(tmp_path / "balanced_accuracy.nii.gz")
    saved = nib.load(output)

    assert output.exists()
    assert saved.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_allclose(saved.get_fdata(), 0.6)
