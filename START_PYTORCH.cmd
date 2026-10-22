@echo off
setlocal
cd /d "%~dp0"
set "IC_PYTHON=%~dp0.venv-torch-xpu\Scripts\python.exe"
if not exist "%IC_PYTHON%" (
    echo The Intel GPU Python environment was not found.
    echo Place this launcher in the ImageClassification project folder.
    pause
    exit /b 1
)
if not exist "01_PyTorch_Intel_GPU.ipynb" (
    echo The notebook 01_PyTorch_Intel_GPU.ipynb was not found.
    pause
    exit /b 1
)
echo Opening the new PyTorch Intel GPU notebook...
echo Keep this window open while using JupyterLab.
"%IC_PYTHON%" -m jupyterlab "01_PyTorch_Intel_GPU.ipynb"
if errorlevel 1 pause

