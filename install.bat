@echo off
REM Liliana - installation sous Windows.
REM Double-cliquez sur ce fichier, ou lancez-le depuis une invite de commandes.

setlocal
cd /d "%~dp0"

echo.
echo   Liliana - installation
echo   ======================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo   [--] Python introuvable.
    echo        Installez Python 3.10 ou plus recent depuis https://www.python.org/downloads/
    echo        Cochez bien "Add Python to PATH" pendant l'installation.
    echo.
    pause
    exit /b 1
)

echo   [1/4] Creation de l'environnement virtuel...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo   [--] Echec de la creation de l'environnement virtuel.
        pause
        exit /b 1
    )
)

echo   [2/4] Installation des dependances...
call .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo   [--] Echec de l'installation des dependances.
    pause
    exit /b 1
)

echo   [3/3] Preparation de Liliana (configuration, modeles, voix)...
echo.
call .venv\Scripts\python.exe scripts\setup.py --pull-model

echo.
echo   Si un modele de langage manque encore :
echo     1. Installez Ollama : https://ollama.com/download
echo     2. Lancez : ollama serve
echo     3. Relancez install.bat
echo.
echo   Liliana choisit toute seule le modele installe le plus adapte :
echo   il n'y a rien a editer dans .env.
echo.
echo   Pour demarrer : run.bat
echo.
pause
