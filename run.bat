@echo off
REM Liliana - lancement sous Windows.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   Liliana n'est pas encore installee.
    echo   Lancez d'abord install.bat
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\python.exe run.py %*

REM En cas d'erreur, la fenetre reste ouverte pour laisser lire le message.
if errorlevel 1 pause
