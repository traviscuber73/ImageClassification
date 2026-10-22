@echo off
setlocal
cd /d "%~dp0"
set "IC_WEB_PYTHON=%~dp0.venv-torch-xpu\Scripts\python.exe"
if not exist "%IC_WEB_PYTHON%" (
    echo Intel GPU Python environment not found.
    echo Put this launcher in the ImageClassification project folder.
    pause
    exit /b 1
)
if not exist "%~dp0web_ui\server.py" (
    echo web_ui\server.py was not found.
    pause
    exit /b 1
)
echo Starting Image Lab with your saved PyTorch model...
echo A browser window will open automatically.
echo Keep this window open. Press Ctrl+C here to stop.
"%IC_WEB_PYTHON%" -B -X utf8 "%~dp0web_ui\server.py" --project "%CD%" --open
if errorlevel 1 pause
