.PHONY: help test test-rust test-python examples example stats plots plot format lint clean

help:
	@echo "Brahe Makefile - Aliases for make.py commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test                  Run all tests (Rust + Python)"
	@echo "  make test-rust             Run Rust tests only"
	@echo "  make test-python           Run Python tests only"
	@echo "  make examples              Run all example tests"
	@echo "  make example NAME=<name>   Run specific example test"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format                Format all code (Rust + Python)"
	@echo "  make lint                  Run linters (clippy + ruff)"
	@echo ""
	@echo "Documentation:"
	@echo "  make plots                 Generate all plots"
	@echo "  make plot NAME=<name>      Generate specific plot"
	@echo ""
	@echo "Utilities:"
	@echo "  make stats                 Show example statistics"
	@echo "  make clean                 Clean build artifacts"
	@echo "  make help                  Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make examples"
	@echo "  make example NAME=access/basic_workflow"
	@echo "  make example NAME=access/basic_workflow LANG=rust"
	@echo "  make format"
	@echo "  make plots"
	@echo "  make plot NAME=attitude_representations"

test:
	uv run python make.py test

test-rust:
	uv run python make.py test-rust

test-python:
	uv run python make.py test-python

examples:
	uv run python make.py test-examples

example:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make example NAME=<example_name> [LANG=python|rust]"; \
		exit 1; \
	fi
	@if [ -n "$(LANG)" ]; then \
		uv run python make.py test-example $(NAME) --lang $(LANG); \
	else \
		uv run python make.py test-example $(NAME); \
	fi

format:
	uv run python make.py format

lint:
	uv run python make.py lint

stats:
	uv run python make.py stats

plots:
	uv run python make.py make-plots

plot:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make plot NAME=<plot_name>"; \
		echo "Example: make plot NAME=attitude_representations"; \
		exit 1; \
	fi
	uv run python make.py make-plot $(NAME)

clean:
	@echo "Cleaning build artifacts..."
	uv run make.py clean
	rm -rf target/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
