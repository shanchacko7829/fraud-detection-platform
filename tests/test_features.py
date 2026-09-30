import math
import unittest

import numpy as np
import pandas as pd

from frauddetect.features import FEATURES, TXN_FIELDS, FeatureStore, compute_features, haversine_km
from helpers import make_txn, small_df


class FeatureTests(unittest.TestCase):
    def test_haversine_known_distance(self):
        # Chennai -> Singapore is roughly 2,900 km
        d = haversine_km(13.08, 80.27, 1.35, 103.82)
        self.assertAlmostEqual(d, 2_900, delta=100)

    def test_first_transaction_has_no_history(self):
        f = FeatureStore().compute(make_txn())
        self.assertEqual(f["cnt_10m"], 0)
        self.assertTrue(math.isnan(f["secs_since_last"]))
        self.assertEqual(f["new_device"], 1.0)

    def test_velocity_windows(self):
        s, t0 = FeatureStore(), 1_770_000_000
        for i, dt in enumerate([0, 30, 200, 3_000]):
            s.update(make_txn(txn_id=f"T{i}", timestamp=t0 + dt, merchant_id=f"M{i}"))
        f = s.compute(make_txn(timestamp=t0 + 3_030))
        self.assertEqual(f["cnt_1m"], 1)      # only the one at +3000 is within 60s of +3030
        self.assertEqual(f["cnt_10m"], 1)
        self.assertEqual(f["cnt_1h"], 4)
        self.assertEqual(f["cnt_24h"], 4)
        self.assertEqual(f["distinct_merchants_1h"], 4)

    def test_old_events_leave_the_24h_window(self):
        s, t0 = FeatureStore(), 1_770_000_000
        s.update(make_txn(timestamp=t0))
        f = s.compute(make_txn(timestamp=t0 + 90_000))
        self.assertEqual(f["cnt_24h"], 0)

    def test_new_device_and_country_flags(self):
        s = FeatureStore()
        s.update(make_txn())
        same = s.compute(make_txn(timestamp=1_770_000_100))
        self.assertEqual((same["new_device"], same["new_country"]), (0.0, 0.0))
        other = s.compute(make_txn(timestamp=1_770_000_100, device_id="X1", country="US"))
        self.assertEqual((other["new_device"], other["new_country"]), (1.0, 1.0))

    def test_device_shared_across_accounts(self):
        s = FeatureStore()
        for i in range(3):
            s.update(make_txn(account_id=f"V{i}", device_id="X007", timestamp=1_770_000_000 + i))
        f = s.compute(make_txn(account_id="V9", device_id="X007", timestamp=1_770_000_100))
        self.assertEqual(f["device_n_accounts"], 3)
        # an account's own past use of a device is not counted as "other accounts"
        self.assertEqual(s.compute(make_txn(account_id="V0", device_id="X007", timestamp=1_770_000_100))["device_n_accounts"], 2)

    def test_impossible_travel_speed(self):
        s = FeatureStore()
        s.update(make_txn(timestamp=1_770_000_000, lat=13.08, lon=80.27))                    # Chennai
        f = s.compute(make_txn(timestamp=1_770_000_000 + 600, lat=40.7, lon=-74.0, country="US"))  # New York, 10 min later
        self.assertGreater(f["speed_kmh_last"], 10_000)

    def test_compute_is_read_only(self):
        s = FeatureStore()
        s.update(make_txn())
        txn = make_txn(timestamp=1_770_000_050)
        a, b = s.compute(txn), s.compute(txn)
        self.assertEqual({k: v for k, v in a.items() if not math.isnan(v)}, {k: v for k, v in b.items() if not math.isnan(v)})
        self.assertEqual(s._accounts["A00001"].n, 1)

    def test_no_future_leakage_prefix_invariance(self):
        """Features of row k must be identical whether or not later rows exist."""
        df = small_df().head(1_500)
        full, _ = compute_features(df)
        prefix, _ = compute_features(df.head(800))
        pd.testing.assert_frame_equal(full.head(800), prefix)

    def test_empty_store_passed_in_is_used(self):
        """Regression: an empty store is falsy (len 0); `store or FeatureStore()` silently discarded it."""
        store = FeatureStore()
        compute_features(small_df().head(200), store)
        self.assertGreater(len(store), 0)

    def test_offline_online_parity_after_warmup(self):
        df = small_df().head(3_000)
        offline, _ = compute_features(df)
        store = FeatureStore()
        compute_features(df.head(2_000), store)                      # warm-up (the "history")
        online = pd.DataFrame(
            [store.features_and_update(r) for r in df.iloc[2_000:][TXN_FIELDS].to_dict("records")],
            columns=FEATURES, index=df.index[2_000:])
        pd.testing.assert_frame_equal(offline.iloc[2_000:], online)


if __name__ == "__main__":
    unittest.main()
