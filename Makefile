
# Makefile for Bank Churn Project

PYTHON := python


install:
	pip install -r requirements.txt

lint:
	flake8 bank_churn_analysis.py

format:
	black bank_churn_analysis.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache

all: install format lint test
