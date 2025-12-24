.PHONY: help install-linters lint lint-rust lint-python fmt fmt-rust fmt-python fix fix-rust fix-python clean clippy lint-rust

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)KubeTrainer Linting Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install-linters: ## Install all linting tools (Rust + Python)
	@echo "$(CYAN)Installing Rust linters...$(NC)"
	@rustup component add rustfmt clippy 2>/dev/null || echo "Rust toolchain not found. Install from https://rustup.rs"
	@echo "$(CYAN)Installing Python linters...$(NC)"
	@pip install --upgrade ruff black isort mypy pylint

lint: lint-rust lint-python ## Run all linters

lint-python: ## Lint Python code (ruff + black check)
	@echo "$(CYAN)Running ruff...$(NC)"
	@ruff check infra_training/
	@echo "$(CYAN)Checking Python formatting (black)...$(NC)"
	@black --check infra_training/ || (echo "$(RED)Python formatting issues found. Run 'make fmt-python' to fix$(NC)" && exit 1)
	@echo "$(CYAN)Checking import order (isort)...$(NC)"
	@isort --check-only infra_training/ || (echo "$(RED)Import order issues found. Run 'make fmt-python' to fix$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python linting passed!$(NC)"

lint-python-strict: ## Lint Python with mypy type checking
	@echo "$(CYAN)Running strict Python linting...$(NC)"
	@make lint-python
	@echo "$(CYAN)Running mypy type checking...$(NC)"
	@mypy infra_training/ --ignore-missing-imports
	@echo "$(GREEN)✓ Strict Python linting passed!$(NC)"

fmt: fmt-rust fmt-python ## Auto-fix formatting for all code

fmt-rust: ## Auto-fix Rust formatting
	@echo "$(CYAN)Formatting Rust code...$(NC)"
	@cd k8s-data-collector && cargo fmt --all
	@cd gpu-detect && cargo fmt --all
	@echo "$(GREEN)✓ Rust code formatted!$(NC)"

fmt-python: ## Auto-fix Python formatting
	@echo "$(CYAN)Formatting Python code with black...$(NC)"
	@black infra_training/
	@echo "$(CYAN)Sorting Python imports...$(NC)"
	@isort infra_training/
	@echo "$(GREEN)✓ Python code formatted!$(NC)"

fix: fix-rust fix-python ## Auto-fix all linting issues

fix-rust: ## Auto-fix Rust linting issues
	@echo "$(CYAN)Auto-fixing Rust code...$(NC)"
	@cd k8s-data-collector && cargo clippy --fix --all-targets --all-features --allow-dirty --allow-staged
	@cd gpu-detect && cargo clippy --fix --all-targets --all-features --allow-dirty --allow-staged
	@make fmt-rust
	@echo "$(GREEN)✓ Rust auto-fix complete!$(NC)"

fix-python: ## Auto-fix Python linting issues
	@echo "$(CYAN)Auto-fixing Python code...$(NC)"
	@ruff check --fix infra_training/
	@make fmt-python
	@echo "$(GREEN)✓ Python auto-fix complete!$(NC)"

clippy:
	@echo "Running cargo clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

lint-rust: fmt-rust clippy
	@echo "Rust formatting + linting complete ✅"

check-all: lint ## Alias for 'make lint'

clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning build artifacts...$(NC)"
	@cd k8s-data-collector && cargo clean
	@cd gpu-detect && cargo clean
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete!$(NC)"