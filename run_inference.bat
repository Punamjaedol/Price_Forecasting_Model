@echo off
cd /d %~dp0

echo ===== INFERENCE START =====
if not exist logs\Inference (
    mkdir logs\Inference
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set LOG_FILE=logs\Inference\inference_%%i.log

echo [START] %date% %time% >> "%LOG_FILE%"
python main_(inference).py >> "%LOG_FILE%" 2>&1
echo [END] %date% %time% >> "%LOG_FILE%"

echo ===== INFERENCE DONE =====
