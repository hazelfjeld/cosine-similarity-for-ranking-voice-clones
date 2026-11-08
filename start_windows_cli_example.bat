@echo off
REM EDIT THESE PATHS if you want to use the command-line batch ranking.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m voice_ranker.cli --reference "reference.wav" --candidates "generated_voices" --output "rankings.csv" --json "rankings.json"
pause
