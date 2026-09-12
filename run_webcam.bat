@echo off
cd /d "%~dp0"
echo ========================================================
echo   Smart Waste Segregation - Real-Time Webcam Detection
echo   (Press 'q' in the camera window to quit)
echo ========================================================
echo.
python detect.py
pause
