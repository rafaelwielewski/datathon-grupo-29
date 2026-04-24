.PHONY: install setup train test lint format type-check security pre-commit serve clean

# First project setup - install dependencies and create folder structure
setup: install
	@mkdir -p data/raw data/processed data/golden_set data/processed/artifacts
	@echo "Setup complete! Run 'make train' to train the model."

# Install all dependencies including dev extras
install:
	pip install -e ".[dev]"

# Train the LSTM model and export to ONNX
train:
	@echo "Training LSTM model..."
	python src/models/train.py

# Start FastAPI serving endpoint locally on port 8000
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Run linter
lint:
	@ruff check src/ tests/ evaluation/

# Format code
format:
	@ruff format src/ tests/ evaluation/

# Run static type checking
type-check:
	@mypy src/ --ignore-missing-imports

# Run security scan
security:
	@bandit -r src/ -c pyproject.toml

# Run all pre-commit hooks
pre-commit:
	@pre-commit run --all-files

# Run full test suite with coverage
test:
	@echo "Running tests..."
	@pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run tests fast - stop at first failure
test-fast:
	@pytest tests/ -x -q

# Remove cache and build artifacts
clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
