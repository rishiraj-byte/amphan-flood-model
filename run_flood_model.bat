@echo off
REM Flood model: run_flood_model.bat STACK_FOLDER OUTPUT_FOLDER [--cpu]
set PY=python
if exist "D:\sar_flood\train_kit\.venv\Scripts\python.exe" set PY=D:\sar_flood\train_kit\.venv\Scripts\python.exe
if "%~2"=="" (echo Usage: run_flood_model.bat STACK_FOLDER OUTPUT_FOLDER [--cpu] & exit /b 1)
"%PY%" "%~dp0predict.py" --stack "%~1" --out "%~2" %3
