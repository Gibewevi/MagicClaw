@echo off
setlocal
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\run.ps1" %*
set MAGIC_CLAW_EXIT=%ERRORLEVEL%
if not "%MAGIC_CLAW_NO_PAUSE%"=="1" (
  echo.
  if not "%MAGIC_CLAW_EXIT%"=="0" echo Magic Claw ended with exit code %MAGIC_CLAW_EXIT%.
  echo Press any key to close this window.
  pause >nul
)
exit /b %MAGIC_CLAW_EXIT%
