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

echo   [3/4] Creation du fichier de configuration...
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo        .env cree a partir de .env.example
) else (
    echo        .env existe deja, il est conserve
)

echo   [4/4] Telechargement des voix Piper...
call .venv\Scripts\python.exe scripts\download_voices.py

echo.
echo   Verification de l'environnement :
call .venv\Scripts\python.exe scripts\check_env.py

echo.
echo   Il reste probablement a installer le modele de langage :
echo     1. Installez Ollama : https://ollama.com/download
echo     2. Ouvrez une invite de commandes et lancez : ollama serve
echo     3. Telechargez un modele, par exemple : ollama pull qwen2.5:3b-instruct
echo     4. Renseignez LLM_MODEL=qwen2.5:3b-instruct dans le fichier .env
echo.
echo   Ensuite, lancez Liliana avec run.bat
echo.
pause
