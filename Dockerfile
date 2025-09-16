# Use official Python 3.12 image
FROM python:3.12-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /workspaces/de_bank_churn_analysis

#Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential\
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set default user to vscode (matches devcontainer.json)
ARG USERNAME=vscode

#Default shell
CMD ["bash"]