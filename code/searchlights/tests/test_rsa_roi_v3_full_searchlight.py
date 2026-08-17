import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform


CODE_DIR = Path(__file__).resolve().parents[2]
SEARCHLIGHT_DIR = CODE_DIR / "searchlights"
RUNNER_DIR = CODE_DIR / "analyses" / "03_semantic-space" / "scripts"
sys.path.insert(0, str(SEARCHLIGHT_DIR))
sys.path.insert(0, str(RUNNER_DIR))

import run_rsa_v3_searchlight as runner  # noqa: E402


class RSAROIV3FullSearchlightTests(unittest.TestCase):
    def setUp(self):
        self.behavior = pd.DataFrame(
            {
                "onset": [0, 1, 2, 3, 4, 5],
                "dimension": ["affil", "power", "power", "affil", "affil", "power"],
                "character_role_num": [1, 2, 1, 2, 1, 2],
                "character_decision_num": [1, 1, 2, 2, 3, 3],
                "reaction_time": [1.0, 1.5, 1.1, 1.7, 1.2, 1.8],
            }
        )
        self.choice = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=float,
        )
        self.subject = {
            "sub_id": "sub-test",
            "behavior": self.behavior,
            "embeddings": {"choice": self.choice},
        }

    def test_rdvs_match_the_prespecified_full_model(self):
        nuisance, focal = runner.make_full_model_rdvs(self.subject)
        self.assertEqual(tuple(nuisance), runner.NUISANCE_NAMES)
        self.assertEqual(tuple(focal), runner.FOCAL_NAMES)
        self.assertTrue(all(values.shape == (15,) for values in nuisance.values()))
        self.assertTrue(all(values.shape == (15,) for values in focal.values()))

        history = squareform(focal["semantic_previous_direction"])
        update = squareform(focal["semantic_choice_update_direction"])
        # The first encounter with each character has no past-only history.
        self.assertTrue(np.isnan(history[0, 2]))
        self.assertTrue(np.isnan(update[1, 3]))
        # Role 1 history changes from [1, 0, 0] to mean([1,0,0], [0,1,0]).
        self.assertAlmostEqual(history[2, 4], 1.0 - 1.0 / np.sqrt(2.0))

    def test_prepared_model_has_fixed_pairs_and_five_outputs(self):
        model = runner.prepare_subject_model(self.subject)
        # Four trials have a valid history, so their six mutual pairs survive.
        self.assertEqual(model.n_pairs, 6)
        self.assertEqual(
            model.output_names,
            (
                "r2__full",
                "adjusted_r2__full",
                "beta__full__semantic_previous_direction",
                "beta__full__semantic_choice",
                "beta__full__semantic_choice_update_direction",
            ),
        )

    def test_subject_id_normalization_and_selection(self):
        metadata = {"sub-2": {}, "sub-10": {}}
        images = {"sub-2": Path("two.nii.gz"), "sub-10": Path("ten.nii.gz")}
        self.assertEqual(
            runner.select_subjects([], metadata, images), ["sub-2", "sub-10"]
        )
        self.assertEqual(
            runner.select_subjects(["10", "sub-2"], metadata, images),
            ["sub-10", "sub-2"],
        )


if __name__ == "__main__":
    unittest.main()
