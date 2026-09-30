.PHONY: install data train test stream benchmark drift plots serve all clean

export PYTHONPATH := src
PY ?= python3

install:
	$(PY) -m pip install -r requirements-dev.txt

data:            ## generate the synthetic dataset (~204k transactions)
	$(PY) -m frauddetect generate

train: data      ## features -> train -> evaluate -> save model + results/train_report.json
	$(PY) -m frauddetect train

test:            ## unit + integration tests (runs in a few seconds)
	$(PY) -m unittest discover -s tests -v

stream:          ## replay the held-out window through the scorer via a queue
	$(PY) -m frauddetect stream

benchmark:       ## in-process and HTTP latency
	$(PY) -m frauddetect benchmark

drift:           ## PSI on a same-distribution vs shifted population
	$(PY) -m frauddetect drift-demo

plots:           ## README charts from results/*.json
	$(PY) -m frauddetect plots

serve:           ## http://127.0.0.1:8000  (POST /v1/score, GET /health, GET /metrics)
	$(PY) -m frauddetect serve

all: test train stream benchmark drift plots

clean:
	rm -rf data/*.csv.gz models/*.joblib __pycache__ src/frauddetect/__pycache__ tests/__pycache__
