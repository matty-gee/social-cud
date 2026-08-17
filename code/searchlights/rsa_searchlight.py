"""One-class interface for fast, reusable BrainIAK searchlight RSA.

``RSASearchlight`` owns the complete workflow: prepare fixed behavioral models,
score neural patterns, run a corrected BrainIAK Ball, and return saveable maps.
Only a tiny top-level kernel remains because multiprocessing must be able to
pickle the function BrainIAK calls.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from time import perf_counter
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

Array = np.ndarray
VALID_OUTPUTS = frozenset(
    {"r2", "adjusted_r2", "delta_r2", "partial_r2", "beta"}
)

def _make_subset_model_family(
    focal_predictors: Mapping[str, str],
    *,
    nuisance_name: str = "nuisance",
    full_name: str = "full",
) -> tuple[OrderedDict[str, tuple[str, ...]], OrderedDict[str, tuple[str, str]]]:
    """Create all focal-predictor subsets and one-predictor comparisons.

    Parameters
    ----------
    focal_predictors
        Ordered mapping from compact model keys to RDV names.  For example,
        ``{"history": "semantic_previous_direction", "choice": ...}``.
    nuisance_name, full_name
        Stable names for the nuisance-only and terminal full models.
    """
    keys = tuple(focal_predictors)
    predictors = tuple(focal_predictors.values())
    if not predictors:
        raise ValueError("At least one focal predictor is required.")
    if len(set(keys)) != len(keys) or len(set(predictors)) != len(predictors):
        raise ValueError("Focal keys and RDV names must each be unique.")
    if nuisance_name == full_name:
        raise ValueError("nuisance_name and full_name must differ.")

    models: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    models[nuisance_name] = ()
    subset_to_name: dict[tuple[int, ...], str] = {(): nuisance_name}
    model_number = 1
    for size in range(1, len(predictors) + 1):
        for subset in combinations(range(len(predictors)), size):
            if size == len(predictors):
                name = full_name
            else:
                name = f"M{model_number}_{'_'.join(keys[index] for index in subset)}"
                model_number += 1
            models[name] = tuple(predictors[index] for index in subset)
            subset_to_name[subset] = name

    comparisons: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for subset, larger_name in tuple(subset_to_name.items())[1:]:
        for added_index in subset:
            reduced_subset = tuple(index for index in subset if index != added_index)
            reduced_name = subset_to_name[reduced_subset]
            added_key = keys[added_index]
            if reduced_subset:
                conditioned = "_".join(keys[index] for index in reduced_subset)
                comparison_name = f"{added_key}_given_{conditioned}"
            else:
                comparison_name = f"{added_key}_vs_{nuisance_name}"
            comparisons[comparison_name] = (larger_name, reduced_name)
    return models, comparisons

def _infer_n_trials(n_pairs: int) -> int:
    n_trials = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    if n_trials * (n_trials - 1) // 2 != n_pairs:
        raise ValueError(f"RDV length {n_pairs} is not a condensed distance length.")
    return n_trials

def _is_binary(values: Array) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size > 0 and np.unique(finite).size <= 2

def _standardize_predictor(
    values: Array,
    *,
    zscore_x: bool,
    zscore_binary: bool,
    eps: float,
) -> Array:
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std(ddof=1))
    if not np.isfinite(sd) or sd <= eps:
        return np.zeros_like(values)
    if zscore_x and (zscore_binary or not _is_binary(values)):
        return (values - values.mean()) / sd
    return values

def _column_space_basis(design: Array) -> tuple[Array, int, float]:
    """Return an orthonormal basis that is safe for rank-deficient designs."""
    u, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((design.shape[0], 0)), 0, np.inf
    tolerance = np.finfo(singular_values.dtype).eps * max(design.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if rank == design.shape[1] and singular_values[-1] > 0
        else np.inf
    )
    return np.ascontiguousarray(u[:, :rank]), rank, condition_number

def _correlation_rdm(
    patterns: Array,
    *,
    method: str = "matrix",
    dtype=np.float32,
    eps: float = 1e-12,
) -> Array | None:
    """Correlation-distance RDV between trial patterns.

    ``patterns`` must be trials by voxels.  ``method='matrix'`` is the fast
    path; ``method='scipy'`` is retained for validation.
    """
    patterns = np.asarray(patterns)
    if patterns.ndim != 2:
        raise ValueError("patterns must have shape (trials, voxels).")
    if patterns.shape[0] < 2 or patterns.shape[1] < 2:
        return None
    if not np.isfinite(patterns).all():
        return None

    if method == "scipy":
        # Reference-only dependency: keeping this import lazy avoids making all
        # spawned production workers pay SciPy's import cost.
        from scipy.spatial.distance import pdist

        result = pdist(patterns, metric="correlation")
        return result if np.isfinite(result).all() else None
    if method != "matrix":
        raise ValueError("method must be 'matrix' or 'scipy'.")

    # Correlation distance is 1 - cosine similarity after centering each trial
    # pattern.  One matrix multiplication is much faster here than calling a
    # general pairwise-distance routine for every center.
    work = np.asarray(patterns, dtype=dtype)
    centered = work - work.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= eps):
        return None
    normalized = centered / norms[:, None]
    triangle = np.triu_indices(patterns.shape[0], 1)
    # Some BLAS backends can warn rather than raise for a pathological local
    # product.  Treat any such center as invalid, just as the SciPy path does.
    with np.errstate(invalid="ignore", over="ignore"):
        result = (1.0 - normalized @ normalized.T)[triangle]
    return result if np.isfinite(result).all() else None

@dataclass(frozen=True)
class RSASearchlight:
    """One subject's fixed RSA models plus the reusable searchlight runner.

    Construct with :meth:`from_rdvs`.  The object contains only arrays and
    metadata, so BrainIAK can broadcast it once and reuse it at every center.
    """

    n_trials: int
    pair_mask: Array
    model_names: tuple[str, ...]
    model_predictors: tuple[tuple[str, ...], ...]
    comparison_names: tuple[str, ...]
    comparison_indices: tuple[tuple[int, int], ...]
    designs: tuple[Array, ...]
    bases: tuple[Array, ...]
    pseudoinverses: tuple[Array, ...]
    ranks: tuple[int, ...]
    condition_numbers: tuple[float, ...]
    nuisance_count: int
    outputs: tuple[str, ...]
    beta_model_indices: tuple[int, ...]
    output_names: tuple[str, ...]
    zscore_y: bool = True
    neural_rdm_method: str = "matrix"
    neural_rdm_dtype: object = np.float32
    eps: float = 1e-12

    @staticmethod
    def make_subset_models(
        focal_predictors: Mapping[str, str],
        *,
        nuisance_name: str = "nuisance",
        full_name: str = "full",
    ) -> tuple[
        OrderedDict[str, tuple[str, ...]],
        OrderedDict[str, tuple[str, str]],
    ]:
        """Build the complete subset ladder used by ``rsa_roi_v3``.

        Custom analyses may instead pass their own ``models`` and
        ``comparisons`` mappings directly to :meth:`from_rdvs`.
        """
        return _make_subset_model_family(
            focal_predictors,
            nuisance_name=nuisance_name,
            full_name=full_name,
        )

    @classmethod
    def from_rdvs(
        cls,
        *,
        nuisance_rdvs: Mapping[str, Array],
        focal_rdvs: Mapping[str, Array],
        models: Mapping[str, Sequence[str]],
        comparisons: Mapping[str, tuple[str, str]] | None = None,
        pair_mask: Array | None = None,
        outputs: Sequence[str] = ("r2", "delta_r2"),
        beta_models: Sequence[str] | str = "full",
        zscore_x: bool = True,
        zscore_binary: bool = False,
        zscore_y: bool = True,
        neural_rdm_method: str = "matrix",
        neural_rdm_dtype=np.float32,
        eps: float = 1e-12,
    ) -> "RSASearchlight":
        nuisance_rdvs = OrderedDict(nuisance_rdvs)
        focal_rdvs = OrderedDict(focal_rdvs)
        models = OrderedDict((name, tuple(values)) for name, values in models.items())
        comparisons = OrderedDict(comparisons or {})
        outputs = tuple(outputs)

        unknown_outputs = set(outputs) - VALID_OUTPUTS
        if unknown_outputs:
            raise ValueError(f"Unknown outputs: {sorted(unknown_outputs)}")
        if not models:
            raise ValueError("At least one model is required.")
        overlap = set(nuisance_rdvs) & set(focal_rdvs)
        if overlap:
            raise ValueError(f"Predictors cannot be nuisance and focal: {sorted(overlap)}")

        all_rdvs = OrderedDict(nuisance_rdvs)
        all_rdvs.update(focal_rdvs)
        if not all_rdvs:
            raise ValueError("At least one RDV is required.")
        lengths = {np.asarray(values).size for values in all_rdvs.values()}
        if len(lengths) != 1:
            raise ValueError("Every RDV must have the same length.")
        n_all_pairs = lengths.pop()
        n_trials = _infer_n_trials(n_all_pairs)

        # Scientific choice: every center uses one fixed trial-pair set.  A
        # center with an additional neural NaN is rejected rather than fitted
        # with a subtly different design matrix.
        valid = np.ones(n_all_pairs, dtype=bool)
        for name, values in all_rdvs.items():
            values = np.asarray(values)
            if values.ndim != 1 or values.size != n_all_pairs:
                raise ValueError(f"RDV {name!r} must be one-dimensional.")
            valid &= np.isfinite(values)
        if pair_mask is not None:
            pair_mask = np.asarray(pair_mask, dtype=bool)
            if pair_mask.shape != valid.shape:
                raise ValueError("pair_mask must match the RDV length.")
            valid &= pair_mask
        if int(valid.sum()) < 2:
            raise ValueError("Fewer than two valid trial pairs remain.")

        # Predictor scaling is fixed per subject.  Binary columns retain their
        # original coding by default, matching rsa_roi_v3.
        lookup: dict[str, Array] = {}
        for name, values in all_rdvs.items():
            lookup[name] = _standardize_predictor(
                np.asarray(values, dtype=np.float64)[valid],
                zscore_x=zscore_x,
                zscore_binary=zscore_binary,
                eps=eps,
            )

        focal_names = set(focal_rdvs)
        model_names = tuple(models)
        model_index = {name: index for index, name in enumerate(model_names)}
        designs = []
        bases = []
        pseudoinverses = []
        ranks = []
        condition_numbers = []
        n_pairs = int(valid.sum())
        nuisance_names = tuple(nuisance_rdvs)
        # Pay for the decompositions once.  The SVD basis makes projection
        # exact even for rank-deficient designs; pseudoinverses are needed only
        # when coefficient maps are requested.
        for model_name, focal_names_in_model in models.items():
            unknown = set(focal_names_in_model) - focal_names
            if unknown:
                raise ValueError(f"Model {model_name!r} has unknown focal RDVs: {sorted(unknown)}")
            if len(focal_names_in_model) != len(set(focal_names_in_model)):
                raise ValueError(f"Model {model_name!r} repeats a focal predictor.")
            predictor_names = nuisance_names + tuple(focal_names_in_model)
            design = np.column_stack(
                [np.ones(n_pairs)] + [lookup[name] for name in predictor_names]
            )
            basis, rank, condition_number = _column_space_basis(design)
            designs.append(np.ascontiguousarray(design))
            bases.append(basis)
            pseudoinverses.append(np.ascontiguousarray(np.linalg.pinv(design)))
            ranks.append(rank)
            condition_numbers.append(condition_number)

        comparison_names = []
        comparison_indices = []
        for comparison_name, (larger_name, reduced_name) in comparisons.items():
            if larger_name not in model_index or reduced_name not in model_index:
                raise ValueError(f"Comparison {comparison_name!r} names an unknown model.")
            larger_predictors = set(models[larger_name])
            reduced_predictors = set(models[reduced_name])
            if not reduced_predictors < larger_predictors:
                raise ValueError(
                    f"Comparison {comparison_name!r} must be strictly nested."
                )
            comparison_names.append(comparison_name)
            comparison_indices.append(
                (model_index[larger_name], model_index[reduced_name])
            )

        if "beta" in outputs:
            if isinstance(beta_models, str):
                beta_models = (beta_models,)
            unknown_beta_models = set(beta_models) - set(model_names)
            if unknown_beta_models:
                raise ValueError(f"Unknown beta models: {sorted(unknown_beta_models)}")
            beta_model_indices = tuple(model_index[name] for name in beta_models)
        else:
            beta_model_indices = ()

        # Each center returns one fixed-order numeric array—no per-center dicts,
        # estimator objects, or DataFrames.
        output_names = []
        if "r2" in outputs:
            output_names.extend(f"r2__{name}" for name in model_names)
        if "adjusted_r2" in outputs:
            output_names.extend(f"adjusted_r2__{name}" for name in model_names)
        if "delta_r2" in outputs:
            output_names.extend(f"delta_r2__{name}" for name in comparison_names)
        if "partial_r2" in outputs:
            output_names.extend(f"partial_r2__{name}" for name in comparison_names)
        if "beta" in outputs:
            for index in beta_model_indices:
                output_names.extend(
                    f"beta__{model_names[index]}__{predictor}"
                    for predictor in models[model_names[index]]
                )

        return cls(
            n_trials=n_trials,
            pair_mask=valid,
            model_names=model_names,
            model_predictors=tuple(models.values()),
            comparison_names=tuple(comparison_names),
            comparison_indices=tuple(comparison_indices),
            designs=tuple(designs),
            bases=tuple(bases),
            pseudoinverses=tuple(pseudoinverses),
            ranks=tuple(ranks),
            condition_numbers=tuple(condition_numbers),
            nuisance_count=len(nuisance_names),
            outputs=outputs,
            beta_model_indices=beta_model_indices,
            output_names=tuple(output_names),
            zscore_y=zscore_y,
            neural_rdm_method=neural_rdm_method,
            neural_rdm_dtype=neural_rdm_dtype,
            eps=eps,
        )

    @property
    def n_pairs(self) -> int:
        return int(self.pair_mask.sum())

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    def empty_result(self) -> Array:
        return np.full(self.n_outputs, np.nan, dtype=np.float32)

    def _prepare_y(self, neural_rdm: Array) -> tuple[Array, Array, float] | None:
        neural_rdm = np.asarray(neural_rdm)
        if neural_rdm.shape != self.pair_mask.shape:
            raise ValueError("neural_rdm length does not match the prepared pair mask.")
        y = np.asarray(neural_rdm[self.pair_mask], dtype=np.float64)
        if not np.isfinite(y).all():
            return None
        centered = y - y.mean()
        sst = float(centered @ centered)
        if not np.isfinite(sst) or sst <= self.eps:
            return None
        if self.zscore_y:
            sd = float(centered.std(ddof=1))
            if not np.isfinite(sd) or sd <= self.eps:
                return None
            coefficient_y = centered / sd
        else:
            coefficient_y = centered
        return centered, coefficient_y, sst

    def _assemble_output(
        self,
        r2: Array,
        coefficient_y: Array,
        reference_coefficients: Mapping[int, Array] | None = None,
    ) -> Array:
        chunks = []
        n = self.n_pairs
        if "r2" in self.outputs:
            chunks.append(r2)
        if "adjusted_r2" in self.outputs:
            adjusted = np.empty_like(r2)
            for index, rank in enumerate(self.ranks):
                p_effective = max(rank - 1, 0)
                adjusted[index] = (
                    1.0 - (1.0 - r2[index]) * (n - 1) / (n - p_effective - 1)
                    if n > p_effective + 1
                    else np.nan
                )
            chunks.append(adjusted)
        if "delta_r2" in self.outputs or "partial_r2" in self.outputs:
            deltas = np.asarray(
                [r2[larger] - r2[reduced] for larger, reduced in self.comparison_indices]
            )
            if "delta_r2" in self.outputs:
                chunks.append(deltas)
            if "partial_r2" in self.outputs:
                partial = np.asarray(
                    [
                        delta / (1.0 - r2[reduced])
                        if r2[reduced] < 1.0 - self.eps
                        else np.nan
                        for delta, (_, reduced) in zip(deltas, self.comparison_indices)
                    ]
                )
                chunks.append(partial)
        if "beta" in self.outputs:
            beta_values = []
            for index in self.beta_model_indices:
                coefficients = (
                    reference_coefficients[index]
                    if reference_coefficients is not None
                    else self.pseudoinverses[index] @ coefficient_y
                )
                first_focal = 1 + self.nuisance_count
                beta_values.extend(coefficients[first_focal:])
            chunks.append(np.asarray(beta_values))
        if not chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def score_neural_rdm(self, neural_rdm: Array) -> Array:
        """Score one neural RDV using the precomputed OLS projections.

        The behavioral design never changes across centers.  Reusing its
        column-space basis avoids thousands of repeated least-squares fits.
        """
        prepared_y = self._prepare_y(neural_rdm)
        if prepared_y is None:
            return self.empty_result()
        centered, coefficient_y, sst = prepared_y
        r2 = np.empty(len(self.bases), dtype=np.float64)
        for index, basis in enumerate(self.bases):
            coordinates = basis.T @ centered
            r2[index] = float(coordinates @ coordinates / sst)
        return self._assemble_output(r2, coefficient_y)

    def score_neural_rdm_reference(self, neural_rdm: Array) -> Array:
        """Repeat ordinary least squares for parity tests, not production."""
        prepared_y = self._prepare_y(neural_rdm)
        if prepared_y is None:
            return self.empty_result()
        centered, coefficient_y, sst = prepared_y
        r2 = np.empty(len(self.designs), dtype=np.float64)
        reference_coefficients = {}
        for index, design in enumerate(self.designs):
            coefficients, *_ = np.linalg.lstsq(design, centered, rcond=None)
            residual = centered - design @ coefficients
            r2[index] = 1.0 - float(residual @ residual) / sst
            if index in self.beta_model_indices:
                reference_coefficients[index], *_ = np.linalg.lstsq(
                    design, coefficient_y, rcond=None
                )
        return self._assemble_output(
            r2,
            coefficient_y,
            reference_coefficients=reference_coefficients,
        )

    def score_patterns(self, patterns: Array) -> Array:
        """Convert trial patterns to one neural RDV, then score every model."""
        neural_rdm = _correlation_rdm(
            patterns,
            method=self.neural_rdm_method,
            dtype=self.neural_rdm_dtype,
            eps=self.eps,
        )
        return self.empty_result() if neural_rdm is None else self.score_neural_rdm(neural_rdm)

    def run(
        self,
        *,
        beta_img: str | Path | nib.spatialimages.SpatialImage,
        analysis_mask_img: str | Path | nib.spatialimages.SpatialImage,
        radius_voxels: int = 5,
        min_active_proportion: float = 0.10,
        pool_size: int = 12,
        max_block_edge: int = 10,
        data_dtype=np.float32,
    ) -> "SearchlightResult":
        """Run one corrected Ball searchlight and return all requested maps."""
        return _run_subject(
            beta_img=beta_img,
            analysis_mask_img=analysis_mask_img,
            rsa=self,
            radius_voxels=radius_voxels,
            min_active_proportion=min_active_proportion,
            pool_size=pool_size,
            max_block_edge=max_block_edge,
            data_dtype=data_dtype,
        )

    @staticmethod
    def recommended_workers(
        *, total_cores: int = 12, concurrent_subjects: int = 1
    ) -> tuple[int, int]:
        """Return ``(subjects, workers_each)`` without CPU oversubscription."""
        if total_cores < 1 or concurrent_subjects < 1:
            raise ValueError("total_cores and concurrent_subjects must be positive.")
        subjects = min(concurrent_subjects, total_cores)
        return subjects, max(1, total_cores // subjects)

    @staticmethod
    def _kernel(data, searchlight_mask, radius, rsa: "RSASearchlight") -> Array:
        """Pickle-safe bridge between BrainIAK and the prepared model."""
        del radius
        # BrainIAK provides a bounding cube plus the intended Ball/brain mask.
        # Applying this mask avoids the legacy bug that analyzed cube corners.
        patterns = data[0][searchlight_mask].T
        return rsa.score_patterns(patterns)

@dataclass
class SearchlightResult:
    values: Array
    output_names: tuple[str, ...]
    affine: Array
    header: nib.Nifti1Header
    analysis_mask: Array
    timings: dict[str, float]

    def save_maps(self, output_dir: str | Path, prefix: str) -> list[Path]:
        """Save one compressed 3D NIfTI for each named result."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for index, name in enumerate(self.output_names):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
            path = output_dir / f"{prefix}_{safe_name}.nii.gz"
            header = self.header.copy()
            header.set_data_dtype(np.float32)
            image = nib.Nifti1Image(
                self.values[..., index].astype(np.float32, copy=False),
                self.affine,
                header,
            )
            nib.save(image, path)
            saved.append(path)
        return saved

def _shape_relative_brainiak_threshold(radius: int, proportion: float) -> float:
    """Translate Ball-relative coverage to BrainIAK 0.12's cube denominator."""
    grid = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    ball_size = int((sum(axis * axis for axis in grid) <= radius * radius).sum())
    cube_size = (2 * radius + 1) ** 3
    return float(proportion * ball_size / cube_size)

def _numeric_result(raw_result: Array, n_outputs: int) -> Array:
    values = np.full(raw_result.shape + (n_outputs,), np.nan, dtype=np.float32)
    for index, result in np.ndenumerate(raw_result):
        if result is None:
            continue
        result = np.asarray(result, dtype=np.float32)
        if result.shape == (n_outputs,):
            values[index] = result
    return values

def _run_subject(
    *,
    beta_img: str | Path | nib.spatialimages.SpatialImage,
    analysis_mask_img: str | Path | nib.spatialimages.SpatialImage,
    rsa: RSASearchlight,
    radius_voxels: int = 5,
    min_active_proportion: float = 0.10,
    pool_size: int = 12,
    max_block_edge: int = 10,
    data_dtype=np.float32,
) -> SearchlightResult:
    
    """Run one corrected Ball searchlight for one subject.

    ``min_active_proportion`` is defined relative to the Ball, correcting the
    cube-denominator behavior in BrainIAK 0.12.
    """
    if radius_voxels < 0 or int(radius_voxels) != radius_voxels:
        raise ValueError("radius_voxels must be a non-negative integer.")
    if not 0 <= min_active_proportion < 1:
        raise ValueError("min_active_proportion must be in [0, 1).")
    if pool_size < 1:
        raise ValueError("pool_size must be positive.")

    # Import lazily: importing BrainIAK initializes MPI, which should happen
    # only when a real spatial run starts—not when a notebook imports the class.
    from brainiak.searchlight.searchlight import Ball, Searchlight
    from nilearn.image import resample_to_img

    total_start = perf_counter()
    load_start  = perf_counter()
    beta_image = nib.load(str(beta_img)) if isinstance(beta_img, (str, Path)) else beta_img
    # float32 halves the 4D image memory and drives the fast matrix-RDM path.
    data = beta_image.get_fdata(dtype=data_dtype)
    if data.ndim != 4 or data.shape[-1] != rsa.n_trials:
        raise ValueError(
            f"Beta image must be 4D with {rsa.n_trials} trials; got {data.shape}."
        )
    load_seconds = perf_counter() - load_start

    mask_start = perf_counter()
    mask_image = (
        nib.load(str(analysis_mask_img))
        if isinstance(analysis_mask_img, (str, Path))
        else analysis_mask_img
    )
    mask_image = resample_to_img(
        mask_image,
        beta_image,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    requested_mask = np.asarray(mask_image.dataobj) > 0
    # Voxel validity is global, so compute it once rather than constructing a
    # VarianceThreshold estimator hundreds of thousands of times.
    valid_voxels = np.isfinite(data).all(axis=-1) & (np.var(data, axis=-1) > rsa.eps)
    analysis_mask = requested_mask & valid_voxels
    mask_seconds = perf_counter() - mask_start

    setup_start = perf_counter()
    # BrainIAK 0.12 divides active Ball voxels by the bounding-cube size.  This
    # translation makes the public argument mean a proportion of the Ball.
    threshold = _shape_relative_brainiak_threshold(
        radius_voxels, min_active_proportion
    )
    searchlight = Searchlight(
        sl_rad=radius_voxels,
        max_blk_edge=max_block_edge,
        shape=Ball,
        min_active_voxels_proportion=threshold,
        pool_size=pool_size,
    )
    searchlight.distribute([data], analysis_mask)
    # Broadcast the fixed model object once; the center kernel receives it by
    # reference and performs only neural-RDM construction plus projections.
    searchlight.broadcast(rsa)
    setup_seconds = perf_counter() - setup_start

    searchlight_start = perf_counter()
    raw_result = searchlight.run_searchlight(RSASearchlight._kernel)
    searchlight_seconds = perf_counter() - searchlight_start

    conversion_start = perf_counter()
    values = _numeric_result(raw_result, rsa.n_outputs)
    conversion_seconds = perf_counter() - conversion_start
    timings = {
        "load": load_seconds,
        "mask": mask_seconds,
        "setup": setup_seconds,
        "searchlight": searchlight_seconds,
        "conversion": conversion_seconds,
        "total": perf_counter() - total_start,
    }
    return SearchlightResult(
        values=values,
        output_names=rsa.output_names,
        affine=beta_image.affine,
        header=beta_image.header.copy(),
        analysis_mask=analysis_mask,
        timings=timings,
    )

__all__ = [
    "RSASearchlight",
    "SearchlightResult",
]
