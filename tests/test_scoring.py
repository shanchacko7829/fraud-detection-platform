import unittest

from frauddetect.scoring import Scorer
from helpers import make_txn, small_df, small_model


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.artifact, _ = small_model()
        self.scorer = Scorer(self.artifact)

    def test_response_shape(self):
        r = self.scorer.score(make_txn())
        for key in ("txn_id", "risk_score", "decision", "reasons", "model_version", "latency_ms", "duplicate"):
            self.assertIn(key, r)
        self.assertGreaterEqual(r["risk_score"], 0)
        self.assertLessEqual(r["risk_score"], 1)
        self.assertIn(r["decision"], {"approve", "review", "decline"})

    def test_redelivered_message_is_idempotent(self):
        txn = make_txn(txn_id="dup-1")
        first = self.scorer.score(txn)
        n_after_first = self.scorer.store._accounts["A00001"].n
        second = self.scorer.score(txn)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["risk_score"], second["risk_score"])
        self.assertEqual(self.scorer.store._accounts["A00001"].n, n_after_first)   # state not updated twice

    def test_card_testing_burst_gets_flagged_with_reasons(self):
        t0, decisions = 1_770_000_000, []
        # a normal customer history first...
        for i in range(6):
            self.scorer.score(make_txn(txn_id=f"h{i}", timestamp=t0 + i * 20_000))
        # ...then a burst of tiny e-commerce charges from an unseen device in another country
        for i in range(14):
            decisions.append(self.scorer.score(make_txn(
                txn_id=f"b{i}", timestamp=t0 + 200_000 + i * 5, amount=1.5 + i * 0.1, mcc="digital_goods",
                channel="ecom", device_id="X999", country="RU", lat=55.7, lon=37.6)))
        flagged = [d for d in decisions if d["decision"] != "approve"]
        self.assertTrue(flagged, "a textbook card-testing burst should be flagged")
        self.assertTrue(all(d["reasons"] for d in flagged), "flagged transactions must carry reasons")
        self.assertIn(flagged[-1]["decision"], {"review", "decline"})

    def test_normal_customer_is_approved(self):
        t0 = 1_770_000_000
        out = [self.scorer.score(make_txn(txn_id=f"n{i}", timestamp=t0 + i * 43_200)) for i in range(5)]
        self.assertEqual(out[-1]["decision"], "approve")
        self.assertEqual(out[-1]["reasons"], [])

    def test_warm_up_populates_state_without_scoring(self):
        n = self.scorer.warm_up(small_df().head(500))
        self.assertEqual(n, 500)
        self.assertGreater(len(self.scorer.store), 0)
        self.assertEqual(self.scorer.metrics()["scored"], 0)

    def test_metrics(self):
        self.scorer.score(make_txn(txn_id="m1"))
        m = self.scorer.metrics()
        self.assertEqual(m["scored"], 1)
        self.assertIn("p99", m["latency_ms"])
        self.assertEqual(m["drift"]["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
