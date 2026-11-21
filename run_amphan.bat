@echo off
REM Re-make the Amphan 2020 map from kit v4 (about 2 min on the laptop GPU)
call "%~dp0run_flood_model.bat" D:\sar_flood\train_kit_v4\data "%~dp0rerun_amphan_2020" %1
