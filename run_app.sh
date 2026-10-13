#!/usr/bin/env bash
# Run with: bash run_app.sh   (Git Bash / WSL / Linux / macOS)
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python -m streamlit run app.py
