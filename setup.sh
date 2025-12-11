#!/bin/bash
# setup.sh - Force Python 3.10 and install dependencies

echo "🔧 Setting up Chemistry AI Assistant..."

# Install system dependencies for RDKit
apt-get update && apt-get install -y \
    libxrender1 \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libfontconfig1 \
    libfreetype6

# Install Python 3.10 if not present (Streamlit Cloud usually has it)
python3.10 --version || echo "Python 3.10 not found, continuing..."

# Upgrade pip
python3 -m pip install --upgrade pip==23.3.1

# Install specific compatible versions
python3 -m pip install \
    rdkit-pypi==2023.9.5 \
    streamlit==1.32.0 \
    pandas==2.1.4 \
    numpy==1.24.3 \
    requests==2.31.0 \
    Pillow==10.1.0 \
    markdown-it-py==2.2.0 \
    rich==13.7.0

echo "✅ Setup complete!"
