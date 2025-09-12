PY=python3
PIP=python3 -m pip

ENV=.venv
ACTIVATE=. $(ENV)/bin/activate

.PHONY: setup dev run fmt test clean models

setup:
	$(PY) -m venv $(ENV)
	$(ACTIVATE); $(PIP) install -U pip
	$(ACTIVATE); $(PIP) install -r requirements.txt
	$(ACTIVATE); $(PY) setup_models.py

run:
	$(ACTIVATE); VHYS_ENV=dev uvicorn server.main:app --host $${VHYS_HOST:-0.0.0.0} --port $${VHYS_PORT:-8080} --reload

dev: run

fmt:
	@echo "Keeping it minimal; add ruff/black if you like."

test:
	$(ACTIVATE); pytest -q

models:
	$(ACTIVATE); $(PY) setup_models.py

clean:
	rm -rf $(ENV) dist build .pytest_cache **/__pycache__ metrics/*.ndjson
