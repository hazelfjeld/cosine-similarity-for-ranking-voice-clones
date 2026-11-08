@echo off
setlocal
cd /d "%~dp0"
echo ===============================
echo    VOICE SIMILARITY RANKER
 echo ===============================
set "PYTHONCMD=python"
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHONCMD=py -3.11"
if "%PYTHONCMD%"=="python" (
  py -3.12 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PYTHONCMD=py -3.12"
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment...
  %PYTHONCMD% -m venv ".venv"
  if errorlevel 1 goto :failure
)
call ".venv\Scripts\activate.bat"
echo Installing any missing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :failure
echo.
echo Opening http://127.0.0.1:7860 in your browser...
python app.py
if errorlevel 1 goto :failure
goto :end
:failure
echo.
echo Something failed. Check the message above and README.md.
pause
:end
endlocal
