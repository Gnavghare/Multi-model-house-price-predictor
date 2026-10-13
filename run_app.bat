@echo off
REM Double-click this file to install packages and start the app
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
