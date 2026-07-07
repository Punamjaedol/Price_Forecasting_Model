@echo off
cd /d %~dp0

echo ===== TRAIN START =====

if not exist logs\Train (
    mkdir logs\Train
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set LOG_FILE=logs\Train\train_%%i.log

echo [START] %date% %time% >> "%LOG_FILE%"
python main_(train_model).py >> "%LOG_FILE%" 2>&1
echo [END] %date% %time% >> "%LOG_FILE%"

echo ===== TRAIN DONE =====