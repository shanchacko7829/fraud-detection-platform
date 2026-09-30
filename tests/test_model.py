import unittest

import numpy as np

from frauddetect.config import LABEL_DELAY_DAYS
from frauddetect.model import (operating_point, recall_at_fpr, sample_weights, threshold_for_fpr,
                               time_splits)
from helpers import small_df, small_model


class SplitTests(unittest.TestCase):
    def test_splits_are_disjoint_and_ordered_in_time(self):
        df = small_df()
        sp = time_splits(df["timestamp"])
        ts = df["timestamp"].to_numpy()
        self.assertFalse((sp.train & sp.valid).any())
        self.assertFalse((sp.valid & sp.test).any())
        self.assertLess(ts[sp.train].max(), ts[sp.valid].min())
        self.assertLess(ts[sp.valid].max(), ts[sp.test].min())

    def test_label_delay_gap_between_train_and_validation(self):
        df = small_df()
        sp = time_splits(df["timestamp"])
        ts = df["timestamp"].to_numpy()
        self.assertGreaterEqual(ts[sp.valid].min() - ts[sp.train].max(), LABEL_DELAY_DAYS * 86_400 * 0.99)

    def test_burn_in_rows_are_not_trained_on(self):
        df = small_df()
        sp = time_splits(df["timestamp"], burn_in_days=7)
        self.assertGreaterEqual(df["timestamp"][sp.train].min(), df["timestamp"].min() + 7 * 86_400)


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = np.r_[np.zeros(9_000, int), np.ones(1_000, int)]
        self.s = np.r_[rng.normal(0, 1, 9_000), rng.normal(3, 1, 1_000)]
        self.amt = rng.uniform(10, 100, 10_000)

    def test_threshold_hits_target_fpr(self):
        thr = threshold_for_fpr(self.y, self.s, 0.01)
        fpr = (self.s[self.y == 0] >= thr).mean()
        self.assertAlmostEqual(fpr, 0.01, delta=0.002)

    def test_operating_point_arithmetic(self):
        op = operating_point(self.y, self.s, self.amt, 1.5)
        self.assertEqual(op["tp"] + op["fn"], 1_000)
        self.assertEqual(op["alerts"], op["tp"] + op["fp"])
        self.assertAlmostEqual(op["precision"], op["tp"] / op["alerts"], places=3)

    def test_recall_at_fpr_is_monotone(self):
        self.assertLessEqual(recall_at_fpr(self.y, self.s, 0.001), recall_at_fpr(self.y, self.s, 0.05))

    def test_sample_weights(self):
        y = np.array([0] * 100 + [1] * 4)
        self.assertTrue((sample_weights(y, "none") == 1).all())
        self.assertAlmostEqual(sample_weights(y, "balanced")[-1], 25.0)
        self.assertAlmostEqual(sample_weights(y, "sqrt")[-1], 5.0)
        self.assertEqual(sample_weights(y, "sqrt")[0], 1.0)


class TrainingSmokeTests(unittest.TestCase):
    """Trains on a small dataset; asserts loose sanity bounds, not exact numbers."""

    @classmethod
    def setUpClass(cls):
        cls.artifact, cls.report = small_model()

    def test_model_beats_random_by_a_wide_margin(self):
        prevalence = self.report["dataset"]["fraud_rate"]
        self.assertGreater(self.report["comparison_test"]["gradient_boosting"]["pr_auc"], 10 * prevalence)

    def test_block_threshold_is_stricter_than_review(self):
        t = self.artifact["thresholds"]
        self.assertGreater(t["block"], t["review"])

    def test_artifact_contents(self):
        for key in ("model", "features", "thresholds", "reference_values", "drift_reference", "meta"):
            self.assertIn(key, self.artifact)
        self.assertIn("score", self.artifact["drift_reference"])

    def test_report_covers_baselines_and_typologies(self):
        self.assertEqual(set(self.report["comparison_test"]), {"rules", "logistic_regression", "gradient_boosting"})
        self.assertTrue(self.report["recall_by_typology_at_review"])


if __name__ == "__main__":
    unittest.main()
