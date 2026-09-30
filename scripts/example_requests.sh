#!/usr/bin/env bash
# Try the API (start it first with `make serve`).
set -euo pipefail
URL=${URL:-http://127.0.0.1:8000}

curl -s "$URL/health"; echo

echo "--- a normal purchase"
curl -s -X POST "$URL/v1/score" -H 'content-type: application/json' -d '{
  "txn_id":"demo-1","timestamp":1772000000,"account_id":"A00042","amount":32.5,
  "merchant_id":"M00-010","mcc":"grocery","channel":"pos","device_id":"D00042-0",
  "country":"IN","lat":13.08,"lon":80.27}'; echo

echo "--- a burst of small e-commerce charges from a new device abroad (card testing)"
for i in $(seq 1 12); do
  curl -s -X POST "$URL/v1/score" -H 'content-type: application/json' -d "{
    \"txn_id\":\"burst-$i\",\"timestamp\":$((1772100000 + i * 6)),\"account_id\":\"A00042\",\"amount\":2.$i,
    \"merchant_id\":\"M10-00$i\",\"mcc\":\"digital_goods\",\"channel\":\"ecom\",\"device_id\":\"X-unknown\",
    \"country\":\"RU\",\"lat\":55.7,\"lon\":37.6}" > /tmp/last.json
done
cat /tmp/last.json; echo

echo "--- metrics"
curl -s "$URL/metrics"; echo
