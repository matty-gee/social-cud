"""Small, reusable building blocks for classification searchlights.

The module deliberately separates two concerns:

1. A scikit-learn estimator describes *which classifier* to fit.
2. ``ClassificationSearchlight`` describes *how to evaluate it* at every
   searchlight center using the same subject-level cross-validation folds.

Each input feature stack has shape ``(x, y, z, subjects)``.  Pass one stack
for a single-map analysis (for example, affiliation minus power), or several
stacks to concatenate multiple local feature sets (for example, affiliation
and power as separate channels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


ImageLike = str | Path | nib.spatialimages.SpatialImage
Fold = tuple[np.ndarray, np.ndarray]


def _load_image(image: ImageLike) -> nib.spatialimages.SpatialImage:
    """Return a nibabel image without changing its data."""

    return nib.load(str(image)) if isinstance(image, (str, Path)) else image


def load_image_stack(
    image_paths: Sequence[str | Path],
    *,
    reference_img: ImageLike | None = None,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    """Load aligned 3D images into one ``(x, y, z, images)`` array.

    Images are checked, not resampled.  Failing on a grid mismatch protects a
    group searchlight from silently comparing different anatomical locations.
    The returned reference image can be supplied when loading another channel.
    """

    paths = tuple(Path(path) for path in image_paths)
    if not paths:
        raise ValueError("image_paths must contain at least one image")

    reference = (
        _load_image(reference_img)
        if reference_img is not None
        else nib.load(str(paths[0]))
    )
    if len(reference.shape) != 3:
        raise ValueError(f"Reference image must be 3D; got shape {reference.shape}")

    stack = np.empty((*reference.shape, len(paths)), dtype=dtype)
    for index, path in enumerate(paths):
        image = nib.load(str(path))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise ValueError(f"Image is not aligned to the reference grid: {path}")
        stack[..., index] = image.get_fdata(dtype=dtype)

    return stack, reference


def make_stratified_folds(
    labels: Sequence[object],
    *,
    balance_by: Sequence[object] | np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 2026,
) -> tuple[Fold, ...]:
    """Create fixed subject folds, optionally balancing extra variables.

    ``balance_by`` may be one vector (such as sex) or a two-dimensional array
    of several categorical variables.  Fold strata are the joint combinations
    of the outcome label and these variables.  Every joint group must therefore
    contain at least ``n_splits`` subjects.
    """

    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(y) < 2:
        raise ValueError("At least two subjects are required")

    strata_columns = [y.astype(str)]
    if balance_by is not None:
        balance = np.asarray(balance_by)
        if balance.ndim == 1:
            balance = balance[:, np.newaxis]
        if balance.ndim != 2 or balance.shape[0] != len(y):
            raise ValueError("balance_by must have one row per subject")
        strata_columns.extend(
            balance[:, column].astype(str) for column in range(balance.shape[1])
        )

    # Tuple representations make unambiguous joint labels such as ("1", "0").
    rows = zip(*(column.tolist() for column in strata_columns))
    strata = np.asarray([repr(tuple(row)) for row in rows])

    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < n_splits:
        raise ValueError(
            "Every joint stratification group must contain at least "
            f"n_splits={n_splits} subjects; smallest group has {counts.min()}"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return tuple((train, test) for train, test in splitter.split(np.zeros(len(y)), strata))


def _shape_relative_brainiak_threshold(
    shape: str,
    radius: int,
    min_active_proportion: float,
) -> float:
    """Adjust BrainIAK's threshold so a Ball is judged relative to its Ball."""

    if shape == "cube":
        return min_active_proportion
    if shape != "ball":
        raise ValueError("shape must be 'ball' or 'cube'")

    # BrainIAK 0.12 divides active Ball voxels by the surrounding cube size.
    # This conversion preserves the requested proportion of the actual Ball.
    grid = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    ball_size = int((sum(axis * axis for axis in grid) <= radius * radius).sum())
    cube_size = (2 * radius + 1) ** 3
    return min_active_proportion * ball_size / cube_size


@dataclass
class ClassificationResult:
    """One whole-brain cross-validated classification score map."""

    values: np.ndarray
    reference_img: nib.spatialimages.SpatialImage
    score_name: str

    def save_map(self, output_path: str | Path) -> Path:
        """Save the score map as a float32 NIfTI image."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        header = self.reference_img.header.copy()
        header.set_data_dtype(np.float32)
        image = nib.Nifti1Image(
            np.asarray(self.values, dtype=np.float32),
            self.reference_img.affine,
            header,
        )
        nib.save(image, str(output))
        return output


class ClassificationSearchlight:
    """Evaluate any scikit-learn classifier in a subject-level searchlight."""

    _SCORES = {
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
    }

    def __init__(
        self,
        *,
        labels: Sequence[object],
        folds: Sequence[Fold],
        estimator: object,
        score: str = "balanced_accuracy",
    ) -> None:
        self.labels = np.asarray(labels)
        self.folds = tuple(
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in folds
        )
        self.estimator = estimator
        self.score = score
        self._validate()

    @property
    def n_subjects(self) -> int:
        return len(self.labels)

    def _validate(self) -> None:
        """Fail early when folds could produce an invalid or biased map."""

        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if np.unique(self.labels).size < 2:
            raise ValueError("labels must contain at least two classes")
        if self.score not in self._SCORES:
            raise ValueError(f"score must be one of {tuple(self._SCORES)}")
        if not self.folds:
            raise ValueError("At least one cross-validation fold is required")

        test_counts = np.zeros(self.n_subjects, dtype=int)
        all_indices = np.arange(self.n_subjects)
        for fold_number, (train, test) in enumerate(self.folds, start=1):
            if train.size == 0 or test.size == 0:
                raise ValueError(f"Fold {fold_number} has an empty train or test set")
            if np.any(train < 0) or np.any(test < 0):
                raise ValueError(f"Fold {fold_number} contains a negative subject index")
            if np.any(train >= self.n_subjects) or np.any(test >= self.n_subjects):
                raise ValueError(f"Fold {fold_number} contains an out-of-range subject index")
            if np.intersect1d(train, test).size:
                raise ValueError(f"Fold {fold_number} has overlapping train and test subjects")
            if np.unique(train).size != train.size or np.unique(test).size != test.size:
                raise ValueError(f"Fold {fold_number} repeats a subject index")
            if not np.array_equal(np.union1d(train, test), all_indices):
                raise ValueError(
                    f"Fold {fold_number} train and test sets do not cover all subjects"
                )
            if np.unique(self.labels[train]).size < 2:
                raise ValueError(f"Fold {fold_number} training data do not contain both classes")
            test_counts[test] += 1

        if not np.array_equal(np.flatnonzero(test_counts), all_indices) or np.any(test_counts != 1):
            raise ValueError("The test folds must include every subject exactly once")

    def score_features(self, features: np.ndarray) -> float:
        """Cross-validate the estimator on one local subject-by-feature matrix."""

        x = np.asarray(features)
        if x.ndim != 2 or x.shape[0] != self.n_subjects:
            raise ValueError(
                "features must have shape (subjects, features); "
                f"expected {self.n_subjects} rows, got {x.shape}"
            )
        if x.shape[1] == 0 or not np.isfinite(x).all():
            return np.nan

        predictions = np.empty(self.labels.shape, dtype=self.labels.dtype)
        for train, test in self.folds:
            # clone() gives every fold an unfitted model.  Any scaling in a
            # Pipeline is consequently learned from training subjects only.
            fitted = clone(self.estimator).fit(x[train], self.labels[train])
            predictions[test] = fitted.predict(x[test])

        return float(self._SCORES[self.score](self.labels, predictions))

    @staticmethod
    def _kernel(
        data: Sequence[np.ndarray],
        searchlight_mask: np.ndarray,
        radius: int,
        classifier: "ClassificationSearchlight",
    ) -> float:
        """Build and score the local matrix for one searchlight center."""

        del radius  # BrainIAK supplies it, but the explicit mask defines the Ball.
        local_channels = [volume[searchlight_mask, :].T for volume in data]
        features = np.concatenate(local_channels, axis=1)
        return classifier.score_features(features)

    def run(
        self,
        *,
        feature_stacks: Sequence[np.ndarray],
        reference_img: ImageLike,
        analysis_mask_img: ImageLike,
        radius_voxels: int = 3,
        shape: str = "ball",
        min_active_proportion: float = 0.10,
        pool_size: int = 1,
        max_block_edge: int = 10,
    ) -> ClassificationResult:
        """Run the classifier at every valid center in ``analysis_mask_img``.

        For parallel runs, keep any internal estimator thread count at one to
        avoid nesting model threads inside the BrainIAK worker processes.
        """

        # Import lazily because importing BrainIAK initializes MPI on some
        # systems, which is surprising when callers only load this module.
        from brainiak.searchlight.searchlight import Ball, Cube, Searchlight

        reference = _load_image(reference_img)
        if len(reference.shape) != 3:
            raise ValueError("reference_img must be 3D")
        if not feature_stacks:
            raise ValueError("feature_stacks must contain at least one 4D array")

        stacks = [np.asarray(stack) for stack in feature_stacks]
        expected_shape = (*reference.shape, self.n_subjects)
        for index, stack in enumerate(stacks):
            if stack.shape != expected_shape:
                raise ValueError(
                    f"Feature stack {index} has shape {stack.shape}; expected {expected_shape}"
                )

        requested_mask_img = resample_to_img(
            _load_image(analysis_mask_img),
            reference,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        requested_mask = requested_mask_img.get_fdata() > 0

        # A center is used only when every subject and every feature channel is
        # finite there.  This is an input-validity rule, not feature selection.
        finite_mask = np.logical_and.reduce(
            [np.isfinite(stack).all(axis=-1) for stack in stacks]
        )
        analysis_mask = requested_mask & finite_mask
        if not analysis_mask.any():
            raise ValueError("The analysis mask contains no finite feature voxels")

        if radius_voxels < 1:
            raise ValueError("radius_voxels must be at least 1")
        if not 0 <= min_active_proportion <= 1:
            raise ValueError("min_active_proportion must be between 0 and 1")
        if pool_size < 1:
            raise ValueError("pool_size must be at least 1")

        sl_shape = {"ball": Ball, "cube": Cube}.get(shape)
        if sl_shape is None:
            raise ValueError("shape must be 'ball' or 'cube'")
        threshold = _shape_relative_brainiak_threshold(
            shape,
            radius_voxels,
            min_active_proportion,
        )

        searchlight = Searchlight(
            sl_rad=radius_voxels,
            max_blk_edge=max_block_edge,
            shape=sl_shape,
            min_active_voxels_proportion=threshold,
            pool_size=pool_size,
        )
        # Each distributed item is one feature channel.  BrainIAK sends the
        # matching spatial neighborhood from every channel to the same kernel.
        searchlight.distribute(stacks, analysis_mask)
        searchlight.broadcast(self)
        raw_result = searchlight.run_searchlight(self._kernel)

        values = np.full(reference.shape, np.nan, dtype=np.float32)
        for index in np.ndindex(reference.shape):
            value = raw_result[index]
            if value is not None:
                values[index] = float(np.asarray(value).squeeze())

        return ClassificationResult(
            values=values,
            reference_img=reference,
            score_name=self.score,
        )


__all__ = [
    "ClassificationResult",
    "ClassificationSearchlight",
    "load_image_stack",
    "make_stratified_folds",
]
