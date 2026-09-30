import unittest

import numpy as np

from frauddetect.data import GenConfig, generate_transactions
from helpers import SMALL, small_df


class DataTests(unittest.TestCase):
    def test_schema_ordering_and_fraud_rate(self):
        df = small_df()
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertTrue(df["txn_id"].is_unique)
        self.assertTrue((df["amount"] > 0).all())
        rate = df["is_fraud"].mean()
        self.assertGreater(rate, 0.003)
        self.assertLess(rate, 0.03)

    def test_all_fraud_typologies_present(self):
        kinds = set(small_df().query("is_fraud == 1")["fraud_type"])
        self.assertEqual(kinds, {"card_testing", "account_takeover", "atm_cashout", "low_and_slow"})

    def test_deterministic_for_same_seed(self):
        cfg = GenConfig(n_accounts=100, n_legit=2_000, days=20, fraud_events=10, legit_bursts=5, seed=11)
        a, b = generate_transactions(cfg), generate_transactions(cfg)
        self.assertTrue(a.equals(b))

    def test_seed_changes_data(self):
        cfg = GenConfig(n_accounts=100, n_legit=2_000, days=20, fraud_events=10, legit_bursts=5, seed=11)
        other = GenConfig(**{**cfg.__dict__, "seed": 12})
        self.assertFalse(np.array_equal(generate_transactions(cfg)["amount"].to_numpy(),
                                        generate_transactions(other)["amount"].to_numpy()))

    def test_coordinates_are_valid(self):
        df = small_df()
        self.assertTrue(df["lat"].between(-90, 90).all())
        self.assertTrue(df["lon"].between(-180, 180).all())
        self.assertEqual(SMALL.n_accounts, 600)


if __name__ == "__main__":
    unittest.main()
