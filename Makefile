
# Makefile for Bank Churn Project

#PYTHON := python
.PHONY: install test run clean

install:
	pip install --no-cache-dir -r requirements.txt
#Run tests
test:
	pytest -v

#Run main script
run:
	python bank_churn_analysis.py

# Lint code
lint:
	flake8 .

# Format code
format:
	black .
	isort .

# Remove pycache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	-rm -rf .pytest_cache
