import unittest

import numpy as np

from frauddetect.drift import DriftMonitor, build_reference, drift_report, psi, status


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.ref_data = self.rng.normal(0, 1, 20_000)

    def test_same_distribution_is_stable(self):
        ref = build_reference({"x": self.ref_data})
        self.assertLess(psi(ref, "x", self.rng.normal(0, 1, 20_000)), 0.02)

    def test_shifted_distribution_is_flagged(self):
        ref = build_reference({"x": self.ref_data})
        v = psi(ref, "x", self.rng.normal(1.0, 1, 20_000))
        self.assertGreater(v, 0.25)
        self.assertEqual(status(v), "drift")

    def test_binary_feature_shift_is_detected(self):
        ref = build_reference({"flag": self.rng.binomial(1, 0.05, 20_000).astype(float)})
        self.assertLess(psi(ref, "flag", self.rng.binomial(1, 0.05, 20_000).astype(float)), 0.02)
        self.assertGreater(psi(ref, "flag", self.rng.binomial(1, 0.30, 20_000).astype(float)), 0.25)

    def test_missing_values_get_their_own_bin(self):
        ref = build_reference({"x": self.ref_data})
        cur = self.ref_data.copy()
        cur[: len(cur) // 2] = np.nan
        self.assertGreater(psi(ref, "x", cur), 0.25)

    def test_constant_feature_does_not_crash(self):
        ref = build_reference({"c": np.ones(1_000)})
        self.assertAlmostEqual(psi(ref, "c", np.ones(500)), 0.0, places=6)

    def test_report_and_monitor(self):
        ref = build_reference({"a": self.ref_data, "score": self.rng.uniform(0, 1, 20_000)})
        rep = drift_report(ref, {"a": self.rng.normal(2, 1, 5_000), "score": self.rng.uniform(0, 1, 5_000)})
        self.assertEqual(rep["max_feature"], "a")
        self.assertEqual(rep["status"], "drift")

        mon = DriftMonitor(ref, ["a"], window=1_000, min_samples=100)
        self.assertEqual(mon.report()["status"], "insufficient_data")
        for v in self.rng.normal(0, 1, 500):
            mon.observe([float(v)], float(self.rng.uniform()))
        self.assertEqual(mon.report()["status"], "stable")


if __name__ == "__main__":
    unittest.main()
