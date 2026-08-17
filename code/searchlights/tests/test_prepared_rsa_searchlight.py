import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial.distance import pdist


SEARCHLIGHT_DIR = Path(__file__).resolve().parents[1]
if str(SEARCHLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(SEARCHLIGHT_DIR))

import rsa_searchlight as module  # noqa: E402
from rsa_searchlight import RSASearchlight, SearchlightResult  # noqa: E402


class RSASearchlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4103)
        cls.n_trials = 14
        cls.n_pairs = cls.n_trials * (cls.n_trials - 1) // 2
        cls.nuisance = {
            "time": np.abs(rng.normal(size=cls.n_pairs)),
            "same_character": rng.integers(0, 2, size=cls.n_pairs).astype(float),
        }
        cls.focal = {
            "history": rng.normal(size=cls.n_pairs),
            "choice": rng.normal(size=cls.n_pairs),
            "update": rng.normal(size=cls.n_pairs),
        }
        # These two pairs must be removed once, before any center is scored.
        cls.nuisance["time"][3] = np.nan
        cls.focal["update"][9] = np.nan
        models, comparisons = RSASearchlight.make_subset_models(
            {
                "history": "history",
                "choice": "choice",
                "update": "update",
            }
        )
        cls.rsa = RSASearchlight.from_rdvs(
            nuisance_rdvs=cls.nuisance,
            focal_rdvs=cls.focal,
            models=models,
            comparisons=comparisons,
            outputs=("r2", "adjusted_r2", "delta_r2", "partial_r2", "beta"),
            beta_models="full",
            neural_rdm_dtype=np.float64,
        )

    def test_fast_outputs_match_repeated_least_squares(self):
        """Covers R2, adjusted/partial R2, delta R2, and full-model betas."""
        rng = np.random.default_rng(718)
        for _ in range(20):
            neural_rdm = rng.normal(size=self.n_pairs)
            np.testing.assert_allclose(
                self.rsa.score_neural_rdm(neural_rdm),
                self.rsa.score_neural_rdm_reference(neural_rdm),
                rtol=1e-6,
                atol=1e-7,
            )

    def test_matrix_pattern_path_matches_scipy_rdm(self):
        rng = np.random.default_rng(91)
        patterns = rng.normal(size=(self.n_trials, 73))
        matrix_result = self.rsa.score_patterns(patterns)
        scipy_result = self.rsa.score_neural_rdm(
            pdist(patterns, metric="correlation")
        )
        np.testing.assert_allclose(matrix_result, scipy_result, rtol=1e-12, atol=1e-12)

    def test_float32_pattern_path_has_small_bounded_error(self):
        rng = np.random.default_rng(501)
        patterns = rng.normal(size=(self.n_trials, 73))
        rsa_float32 = RSASearchlight.from_rdvs(
            nuisance_rdvs=self.nuisance,
            focal_rdvs=self.focal,
            models=dict(zip(self.rsa.model_names, self.rsa.model_predictors)),
            comparisons=dict(
                zip(
                    self.rsa.comparison_names,
                    (
                        (self.rsa.model_names[larger], self.rsa.model_names[reduced])
                        for larger, reduced in self.rsa.comparison_indices
                    ),
                )
            ),
            outputs=("r2", "delta_r2"),
            neural_rdm_dtype=np.float32,
        )
        fast = rsa_float32.score_patterns(patterns)
        scipy = rsa_float32.score_neural_rdm(pdist(patterns, metric="correlation"))
        np.testing.assert_allclose(fast, scipy, rtol=1e-5, atol=1e-6)

    def test_kernel_uses_the_ball_mask_not_the_bounding_cube(self):
        rng = np.random.default_rng(20)
        data = rng.normal(size=(3, 3, 3, self.n_trials))
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        mask[1, 1, 2] = True
        baseline = RSASearchlight._kernel([data], mask, 1, self.rsa)

        changed_outside_mask = data.copy()
        changed_outside_mask[~mask] = rng.normal(0, 1e6, size=(~mask).sum() * self.n_trials).reshape(-1, self.n_trials)
        changed = RSASearchlight._kernel(
            [changed_outside_mask], mask, 1, self.rsa
        )
        np.testing.assert_array_equal(baseline, changed)

    def test_pair_mask_is_fixed_and_invalid_centers_are_rejected(self):
        rng = np.random.default_rng(63)
        neural_rdm = rng.normal(size=self.n_pairs)

        # NaNs in behavior-excluded pairs do not alter the fixed design.
        neural_rdm[[3, 9]] = np.nan
        self.assertTrue(np.isfinite(self.rsa.score_neural_rdm(neural_rdm)).all())

        # A NaN in an included pair invalidates the center instead of silently
        # fitting a different pair set at that location.
        neural_rdm[np.flatnonzero(self.rsa.pair_mask)[0]] = np.nan
        self.assertTrue(np.isnan(self.rsa.score_neural_rdm(neural_rdm)).all())

    def test_constant_trial_patterns_return_nan(self):
        patterns = np.ones((self.n_trials, 10))
        self.assertTrue(np.isnan(self.rsa.score_patterns(patterns)).all())

    def test_rank_deficient_design_matches_reference(self):
        duplicated_focal = dict(self.focal)
        duplicated_focal["choice_copy"] = duplicated_focal["choice"].copy()
        rsa = RSASearchlight.from_rdvs(
            nuisance_rdvs=self.nuisance,
            focal_rdvs=duplicated_focal,
            models={
                "nuisance": (),
                "full": ("history", "choice", "choice_copy", "update"),
            },
            comparisons={"all_vs_nuisance": ("full", "nuisance")},
            outputs=("r2", "delta_r2"),
            neural_rdm_dtype=np.float64,
        )
        neural_rdm = np.random.default_rng(19).normal(size=self.n_pairs)
        np.testing.assert_allclose(
            rsa.score_neural_rdm(neural_rdm),
            rsa.score_neural_rdm_reference(neural_rdm),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertLess(rsa.ranks[-1], rsa.designs[-1].shape[1])

    def test_model_family_and_output_order(self):
        self.assertEqual(len(self.rsa.model_names), 8)
        self.assertEqual(len(self.rsa.comparison_names), 12)
        self.assertEqual(self.rsa.n_pairs, self.n_pairs - 2)
        self.assertEqual(self.rsa.n_outputs, 43)
        self.assertEqual(self.rsa.output_names[0], "r2__nuisance")
        self.assertEqual(self.rsa.output_names[7], "r2__full")
        self.assertEqual(self.rsa.output_names[-1], "beta__full__update")

    def test_ball_coverage_is_relative_to_ball_not_cube(self):
        expected = 0.10 * 515 / 1331
        self.assertAlmostEqual(
            module._shape_relative_brainiak_threshold(5, 0.10), expected
        )

    def test_result_saves_named_float32_maps(self):
        values = np.zeros((2, 2, 2, 2), dtype=np.float32)
        result = SearchlightResult(
            values=values,
            output_names=("r2__full", "delta_r2__choice/given-history"),
            affine=np.eye(4),
            header=nib.Nifti1Header(),
            analysis_mask=np.ones((2, 2, 2), dtype=bool),
            timings={},
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = result.save_maps(directory, "sub-test")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertNotIn("/", paths[1].name)
            image = nib.load(paths[0])
            self.assertEqual(image.shape, (2, 2, 2))
            self.assertEqual(image.get_data_dtype(), np.dtype(np.float32))

    def test_worker_recommendations_avoid_oversubscription(self):
        self.assertEqual(RSASearchlight.recommended_workers(concurrent_subjects=1), (1, 12))
        self.assertEqual(RSASearchlight.recommended_workers(concurrent_subjects=2), (2, 6))
        self.assertEqual(RSASearchlight.recommended_workers(concurrent_subjects=3), (3, 4))
        self.assertEqual(RSASearchlight.recommended_workers(concurrent_subjects=5), (5, 2))

    def test_non_nested_comparison_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly nested"):
            RSASearchlight.from_rdvs(
                nuisance_rdvs=self.nuisance,
                focal_rdvs=self.focal,
                models={"history": ("history",), "choice": ("choice",)},
                comparisons={"invalid": ("history", "choice")},
            )


if __name__ == "__main__":
    unittest.main()
