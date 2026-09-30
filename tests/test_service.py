import unittest

from starlette.testclient import TestClient

from frauddetect.scoring import Scorer
from frauddetect.service import create_app
from helpers import make_txn, small_model


class ServiceTests(unittest.TestCase):
    def setUp(self):
        artifact, _ = small_model()
        self.client = TestClient(create_app(Scorer(artifact)))

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_score_valid_transaction(self):
        r = self.client.post("/v1/score", json=make_txn())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["txn_id"], "T1")
        self.assertIn(body["decision"], {"approve", "review", "decline"})

    def test_duplicate_txn_id_returns_cached_decision(self):
        first = self.client.post("/v1/score", json=make_txn()).json()
        again = self.client.post("/v1/score", json=make_txn()).json()
        self.assertFalse(first["duplicate"])
        self.assertTrue(again["duplicate"])
        self.assertEqual(first["risk_score"], again["risk_score"])

    def test_validation_errors_return_422(self):
        for bad in (make_txn(mcc="not-a-category"), make_txn(channel="fax"), make_txn(amount=-5),
                    make_txn(lat=123), make_txn(country="X")):
            r = self.client.post("/v1/score", json=bad)
            self.assertEqual(r.status_code, 422, bad)
            self.assertEqual(r.json()["error"], "validation_error")

    def test_missing_field_returns_422(self):
        payload = make_txn()
        del payload["device_id"]
        self.assertEqual(self.client.post("/v1/score", json=payload).status_code, 422)

    def test_malformed_json_returns_400(self):
        r = self.client.post("/v1/score", content=b"{not json", headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 400)

    def test_metrics_endpoint(self):
        self.client.post("/v1/score", json=make_txn())
        m = self.client.get("/metrics").json()
        self.assertEqual(m["scored"], 1)
        self.assertIn("latency_ms", m)


if __name__ == "__main__":
    unittest.main()
