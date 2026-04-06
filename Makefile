# Life Insurance Support Assistant - Makefile
# git commit: feat(build): add Makefile with common commands
# Module: project-foundation

.PHONY: install install-dev run-api run-cli index-kb test lint format clean help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install dev + production dependencies
	pip install -r requirements-dev.txt

run-api: ## Start the FastAPI server
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-cli: ## Start the CLI chat interface
	python -m src.cli.chat

index-kb: ## Index the knowledge base into Qdrant
	python -m src.knowledge.indexer

test: ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linting checks
	ruff check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
