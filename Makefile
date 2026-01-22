# Makefile
# Local quality gates you can run before committing.
# Assumes a venv at .venv (recommended). Adjust PY if needed.

SHELL := /bin/bash
PY := .venv/bin/python
PIP := .venv/bin/pip

# Tools (install via requirements.txt)
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest
MYPY := .venv/bin/mypy

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make venv        - create .venv and install deps"
	@echo "  make fmt         - auto-format (ruff)"
	@echo "  make lint        - lint (ruff)"
	@echo "  make type        - type-check (mypy, if installed)"
	@echo "  make test        - run tests (pytest)"
	@echo "  make check       - run fmt + lint + type + test"
	@echo "  make precommit   - same as check (use before git commit)"
	@echo "  make clean       - remove caches"

.PHONY: venv
venv:
	python -m venv .venv
	$(PIP) install -U pip wheel
	$(PIP) install -r requirements.txt

.PHONY: fmt
fmt:
	@$(RUFF) --version >/dev/null 2>&1 || (echo "ruff not found. Install: pip install ruff" && exit 1)
	$(RUFF) format .
	$(RUFF) check . --fix

.PHONY: lint
lint:
	@$(RUFF) --version >/dev/null 2>&1 || (echo "ruff not found. Install: pip install ruff" && exit 1)
	$(RUFF) check .

.PHONY: type
type:
	@if [ -x "$(MYPY)" ]; then \
		$(MYPY) src ; \
	else \
		echo "mypy not installed (skipping). To enable: pip install mypy"; \
	fi

.PHONY: test
test:
	@if [ -x "$(PYTEST)" ]; then \
		$(PYTEST) -q ; \
	else \
		echo "pytest not installed (skipping). To enable: pip install pytest"; \
	fi

.PHONY: check
check: fmt lint type test

.PHONY: precommit
precommit: check
	@echo "All checks passed."

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
