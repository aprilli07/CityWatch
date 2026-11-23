# CityWatch developer commands (run from repo root)
PY := backend/.venv/bin/python
B := cd backend &&

.PHONY: setup migrate data train-risk train-cv score bootstrap api app test lint

setup:            ## create the Python venv and install dependencies
	cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt -r requirements-dev.txt
	cd frontend && npm install

migrate:          ## apply SQL migrations
	$(B) .venv/bin/python -m db.migrate

data:             ## real OSM assets + weather archive + simulated incident history
	$(B) .venv/bin/python -m scripts.ingest_osm
	$(B) .venv/bin/python -m scripts.backfill_weather
	$(B) .venv/bin/python -m ml.risk.simulate

train-risk:       ## train 30/90/365-day risk models (writes ml/artifacts/risk_metrics.json)
	$(B) .venv/bin/python -m ml.risk.train

train-cv:         ## download dataset, prepare with OpenCV, fine-tune MobileNetV2
	$(B) .venv/bin/kaggle datasets download -d akinduhiman/urban-issues-dataset -p ml/data/raw --unzip
	$(B) .venv/bin/kaggle datasets download -d atulyakumar98/pothole-detection-dataset -p ml/data/raw2 --unzip
	$(B) .venv/bin/kaggle datasets download -d programmerrdai/road-issues-detection-dataset -p ml/data/raw3 --unzip
	$(B) .venv/bin/python -m ml.cv.prepare_dataset ml/data/raw ml/data/raw2 ml/data/raw3
	$(B) .venv/bin/python -m ml.cv.train

score:            ## score all assets/regions with the trained risk model
	$(B) .venv/bin/python -c "from app.services.risk import score_all; print(score_all())"

bootstrap: migrate data train-risk score

api:              ## run the API with autoreload on :8000
	$(B) .venv/bin/uvicorn app.main:app --reload --port 8000

app:              ## run the Expo app
	cd frontend && npx expo start

test:
	$(B) .venv/bin/python -m pytest -q

lint:
	$(B) .venv/bin/ruff check .
	cd frontend && npx tsc --noEmit && npx expo lint
