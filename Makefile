.PHONY: dev test lint up down ingest ask

dev:            ## Install dependencies into .venv
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

test:           ## Run the test suite (no model needed)
	.venv/bin/pytest -q

lint:           ## Lint and check formatting
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

up:             ## Start Ollama, the API and Prometheus
	docker compose up -d --build

down:
	docker compose down

ingest:         ## Load the sample handbook
	python3 -c "import json;print(json.dumps({'source':'handbook.md','text':open('sample_docs/handbook.md').read()}))" \
	  | curl -s -X POST localhost:8000/documents -H 'content-type: application/json' -d @- ; echo

ask:            ## Ask a question: make ask Q="How many vacation days do I get?"
	curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
	  -d "{\"question\": \"$(Q)\"}" ; echo
