@echo off
setlocal
rem Ir a la carpeta del repo
cd /d "%~dp0"

rem Crear venv solo si NO existe
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
)

call ".venv\Scripts\activate"
python -m pip install -U pip
pip install -r requirements.txt

rem >>> Ajustá el input que quieras usar:
python main.py run --input "ort.MOV" --max_frames 300 --no-triang
