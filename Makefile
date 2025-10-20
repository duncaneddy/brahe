.PHONY: help install test test-rust test-python test-examples test-examples-strict format format-check lint ship-tests clean serve-docs build-docs figures all

# Configuration
UV := uv
PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
PRE_COMMIT := uv run pre-commit
MKDOCS := uv run mkdocs
REPO_ROOT := $(shell pwd)
FIGURE_OUTPUT_DIR := docs/figures
EXAMPLES_DIR := examples
PLOTS_DIR := plots

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install all dependencies (Rust, Python, dev tools)
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@if ! command -v rust-script &> /dev/null; then \
		echo "$(YELLOW)Installing rust-script...$(NC)"; \
		cargo install rust-script; \
	fi
	@echo "$(GREEN)Syncing Python dependencies with uv...$(NC)"
	@uv sync
	@echo "$(GREEN)All dependencies installed!$(NC)"

##@ Testing

test: test-rust test-python ## Run all tests (Rust + Python)
	@echo "$(GREEN)All tests passed!$(NC)"

test-rust: ## Run Rust tests
	@echo "$(BLUE)Running Rust tests...$(NC)"
	@cargo test

test-python: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	@$(UV) pip install -e .
	@$(PYTEST) tests/ -v

test-examples: ## Test all doc examples (warn on parity issues, skip IGNORE flags)
	@echo "$(BLUE)Testing documentation examples...$(NC)"
	@$(MAKE) _test_examples STRICT=0

test-examples-strict: ## Test all doc examples (fail on parity issues, skip IGNORE flags)
	@echo "$(BLUE)Testing documentation examples (strict mode)...$(NC)"
	@$(MAKE) _test_examples STRICT=1

test-examples-all: ## Test all examples including CI-ONLY (but not IGNORE)
	@echo "$(BLUE)Testing documentation examples (including CI-ONLY)...$(NC)"
	@$(MAKE) _test_examples STRICT=0 ENABLE_CI_ONLY=1

_test_examples:
	@echo "$(BLUE)=== Testing Rust Examples ===$(NC)"
	@RUST_COUNT=0; \
	RUST_PASS=0; \
	RUST_FAIL=0; \
	RUST_SKIP=0; \
	for example in $$(find $(EXAMPLES_DIR) -name "*.rs" -type f); do \
		if [ -f "$$example" ]; then \
			SKIP=0; \
			SKIP_REASON=""; \
			FLAGS=$$(head -n 10 "$$example" | grep -o "FLAGS = \[.*\]" | sed 's/FLAGS = \[//;s/\]//;s/ //g'); \
			if [ -n "$$FLAGS" ]; then \
				IFS=',' read -ra FLAG_ARRAY <<< "$$FLAGS"; \
				for flag in "$${FLAG_ARRAY[@]}"; do \
					if [ "$$flag" = "IGNORE" ]; then \
						SKIP=1; SKIP_REASON="ignored"; \
					elif [ "$$flag" = "CI-ONLY" ] && [ -z "$$ENABLE_CI_ONLY" ]; then \
						SKIP=1; SKIP_REASON="ci-only"; \
					elif [ "$$flag" = "SLOW" ] && [ -z "$$ENABLE_SLOW" ]; then \
						SKIP=1; SKIP_REASON="slow"; \
					fi; \
				done; \
			fi; \
			if [ $$SKIP -eq 1 ]; then \
				echo "Testing $$example...$(YELLOW)SKIP ($$SKIP_REASON)$(NC)"; \
				RUST_SKIP=$$((RUST_SKIP + 1)); \
			else \
				RUST_COUNT=$$((RUST_COUNT + 1)); \
				printf "Testing $$example..."; \
				TMPFILE=$$(mktemp); \
				echo "//! \`\`\`cargo" > $$TMPFILE; \
				echo "//! [dependencies]" >> $$TMPFILE; \
				echo "//! brahe = {path = \"$(REPO_ROOT)\"}" >> $$TMPFILE; \
				echo "//! approx = \"^0.5.0\"" >> $$TMPFILE; \
				echo "//! serde_json = \"1\"" >> $$TMPFILE; \
				echo "//! \`\`\`" >> $$TMPFILE; \
				cat $$example >> $$TMPFILE; \
				if rust-script $$TMPFILE > /dev/null 2>&1; then \
					echo "$(GREEN)PASS$(NC)"; \
					RUST_PASS=$$((RUST_PASS + 1)); \
				else \
					echo "$(RED)FAIL$(NC)"; \
					RUST_FAIL=$$((RUST_FAIL + 1)); \
				fi; \
				rm $$TMPFILE; \
			fi; \
		fi; \
	done; \
	echo "$(BLUE)Rust Examples: $$RUST_COUNT total, $$RUST_PASS passed, $$RUST_FAIL failed, $$RUST_SKIP skipped$(NC)"
	@echo ""
	@echo "$(BLUE)=== Testing Python Examples ===$(NC)"
	@PYTHON_COUNT=0; \
	PYTHON_PASS=0; \
	PYTHON_FAIL=0; \
	PYTHON_SKIP=0; \
	for example in $$(find $(EXAMPLES_DIR) -name "*.py" -type f); do \
		if [ -f "$$example" ]; then \
			SKIP=0; \
			SKIP_REASON=""; \
			FLAGS=$$(head -n 10 "$$example" | grep -o "FLAGS = \[.*\]" | sed 's/FLAGS = \[//;s/\]//;s/ //g'); \
			if [ -n "$$FLAGS" ]; then \
				IFS=',' read -ra FLAG_ARRAY <<< "$$FLAGS"; \
				for flag in "$${FLAG_ARRAY[@]}"; do \
					if [ "$$flag" = "IGNORE" ]; then \
						SKIP=1; SKIP_REASON="ignored"; \
					elif [ "$$flag" = "CI-ONLY" ] && [ -z "$$ENABLE_CI_ONLY" ]; then \
						SKIP=1; SKIP_REASON="ci-only"; \
					elif [ "$$flag" = "SLOW" ] && [ -z "$$ENABLE_SLOW" ]; then \
						SKIP=1; SKIP_REASON="slow"; \
					fi; \
				done; \
			fi; \
			if [ $$SKIP -eq 1 ]; then \
				echo "Testing $$example...$(YELLOW)SKIP ($$SKIP_REASON)$(NC)"; \
				PYTHON_SKIP=$$((PYTHON_SKIP + 1)); \
			else \
				PYTHON_COUNT=$$((PYTHON_COUNT + 1)); \
				printf "Testing $$example..."; \
				if $(PYTHON) $$example > /dev/null 2>&1; then \
					echo "$(GREEN)PASS$(NC)"; \
					PYTHON_PASS=$$((PYTHON_PASS + 1)); \
				else \
					echo "$(RED)FAIL$(NC)"; \
					PYTHON_FAIL=$$((PYTHON_FAIL + 1)); \
				fi; \
			fi; \
		fi; \
	done; \
	echo "$(BLUE)Python Examples: $$PYTHON_COUNT total, $$PYTHON_PASS passed, $$PYTHON_FAIL failed, $$PYTHON_SKIP skipped$(NC)"
	@echo ""
	@echo "$(BLUE)=== Checking Rust/Python Parity ===$(NC)"
	@MISSING_PY=0; \
	MISSING_RS=0; \
	for rs_file in $$(find $(EXAMPLES_DIR) -name "*.rs" -type f); do \
		if [ -f "$$rs_file" ]; then \
			py_file=$${rs_file%.rs}.py; \
			if [ ! -f "$$py_file" ]; then \
				echo "$(YELLOW)Warning: Missing Python equivalent for $$rs_file$(NC)"; \
				MISSING_PY=$$((MISSING_PY + 1)); \
			fi; \
		fi; \
	done; \
	for py_file in $$(find $(EXAMPLES_DIR) -name "*.py" -type f); do \
		if [ -f "$$py_file" ]; then \
			rs_file=$${py_file%.py}.rs; \
			if [ ! -f "$$rs_file" ]; then \
				echo "$(YELLOW)Warning: Missing Rust equivalent for $$py_file$(NC)"; \
				MISSING_RS=$$((MISSING_RS + 1)); \
			fi; \
		fi; \
	done; \
	if [ $$MISSING_PY -gt 0 ] || [ $$MISSING_RS -gt 0 ]; then \
		echo "$(YELLOW)Parity Issues: $$MISSING_PY missing Python, $$MISSING_RS missing Rust$(NC)"; \
		if [ "$(STRICT)" = "1" ]; then \
			echo "$(RED)STRICT MODE: Failing due to parity issues$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(GREEN)All examples have Rust/Python pairs!$(NC)"; \
	fi

##@ Code Quality

format: ## Auto-format all code (Rust + Python)
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	@cargo fmt
	@echo "$(BLUE)Formatting Python code...$(NC)"
	@$(RUFF) format

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking Rust formatting...$(NC)"
	@cargo fmt -- --check
	@echo "$(BLUE)Checking Python formatting...$(NC)"
	@$(RUFF) format --check

lint: ## Run linters (clippy + ruff)
	@echo "$(BLUE)Running Rust linter (clippy)...$(NC)"
	@cargo clippy --all-targets --all-features -- -D warnings
	@echo "$(BLUE)Running Python linter (ruff)...$(NC)"
	@$(RUFF) check

lint-fix: ## Auto-fix linter issues where possible
	@echo "$(BLUE)Auto-fixing Python linting issues...$(NC)"
	@$(RUFF) check --fix

precommit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	@$(PRE_COMMIT) run --all-files

##@ Documentation

figures: ## Generate all documentation figures
	@echo "$(BLUE)Generating documentation figures...$(NC)"
	@mkdir -p $(FIGURE_OUTPUT_DIR)
	@export BRAHE_FIGURE_OUTPUT_DIR=$(REPO_ROOT)/$(FIGURE_OUTPUT_DIR); \
	for plot in $(PLOTS_DIR)/*.py; do \
		if [ -f "$$plot" ]; then \
			echo "Generating $$(basename $$plot)..."; \
			$(PYTHON) $$plot; \
		fi; \
	done
	@echo "$(GREEN)Figures generated in $(FIGURE_OUTPUT_DIR)/$(NC)"

build-docs: figures ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@./scripts/generate_stubs.sh
	@cd docs && $(MKDOCS) build

build-docs-strict: figures ## Build documentation with strict mode (warnings as errors)
	@echo "$(BLUE)Building documentation (strict mode)...$(NC)"
	@./scripts/generate_stubs.sh
	@cd docs && $(MKDOCS) build --strict

serve-docs: figures ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://127.0.0.1:8000$(NC)"
	@./scripts/generate_stubs.sh
	@cd docs && $(MKDOCS) serve

##@ Ship Tests

ship-tests: ## Run all pre-ship validation (tests + format + lint + docs)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Running Pre-Ship Validation Suite$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)[1/8] Rust Tests$(NC)"
	@$(MAKE) test-rust
	@echo ""
	@echo "$(BLUE)[2/8] Rust Format Check$(NC)"
	@cargo fmt -- --check
	@echo ""
	@echo "$(BLUE)[3/8] Rust Lint (Clippy)$(NC)"
	@cargo clippy --all-targets --all-features -- -D warnings
	@echo ""
	@echo "$(BLUE)[4/8] Python Package Install$(NC)"
	@$(UV) pip install -e .
	@echo ""
	@echo "$(BLUE)[5/8] Python Tests$(NC)"
	@$(PYTEST) tests/ -v
	@echo ""
	@echo "$(BLUE)[6/8] Python Format Check$(NC)"
	@$(RUFF) format --check
	@echo ""
	@echo "$(BLUE)[7/8] Python Lint (Ruff)$(NC)"
	@$(RUFF) check
	@echo ""
	@echo "$(BLUE)[8/8] Documentation Build (Strict)$(NC)"
	@$(MAKE) test-examples-strict
	@$(MAKE) build-docs-strict
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)All Ship Tests Passed! ✓$(NC)"
	@echo "$(GREEN)========================================$(NC)"

##@ Utilities

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@cargo clean
	@rm -rf target/
	@rm -rf docs/site/
	@rm -rf $(FIGURE_OUTPUT_DIR)/*.html
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleaned!$(NC)"

all: install test lint format-check build-docs ## Run full build (install + test + lint + docs)
	@echo "$(GREEN)Full build complete!$(NC)"
