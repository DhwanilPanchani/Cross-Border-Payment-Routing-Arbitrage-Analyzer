# FX Routing Optimizer Pipeline Makefile

.PHONY: all setup download etl analyze model dashboard test clean

# Run full pipeline
all: setup download etl analyze model dashboard

# Setup environment
setup:
	@echo "Setting up environment..."
	pip install -r requirements.txt
	mkdir -p data/raw data/processed outputs/plots outputs/reports models logs

# Download datasets
download:
	@echo "Downloading datasets from Kaggle..."
	python -c "from kagglehub import dataset_download; dataset_download('ieee-fraud-detection')"

# Run ETL pipeline
etl:
	@echo "Running ETL pipeline..."
	python -c "import notebook_runner; notebook_runner.run_cells([6,7,8,9,10,11,14])"

# Run analysis
analyze:
	@echo "Running analysis..."
	python -c "import notebook_runner; notebook_runner.run_cells([15,16,17])"

# Train model
model:
	@echo "Training models..."
	python -c "import notebook_runner; notebook_runner.run_cells([22,23,24])"

# Launch dashboard
dashboard:
	@echo "Launching Streamlit dashboard..."
	streamlit run streamlit_app.py

# Run data quality tests
test:
	@echo "Running data quality tests..."
	python -c "import notebook_runner; notebook_runner.run_cells([27])"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/* outputs/* models/*.pkl logs/*
	rm -f fintech_analytics.db

# Docker build
docker-build:
	@echo "Building Docker image..."
	docker-compose build

# Docker run
docker-run:
	@echo "Running Docker containers..."
	docker-compose up

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Run full pipeline"
	@echo "  setup        - Setup environment"
	@echo "  download     - Download datasets"
	@echo "  etl          - Run ETL pipeline"
	@echo "  analyze      - Run analysis"
	@echo "  model        - Train models"
	@echo "  dashboard    - Launch dashboard"
	@echo "  test         - Run data quality tests"
	@echo "  clean        - Clean generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker containers"
