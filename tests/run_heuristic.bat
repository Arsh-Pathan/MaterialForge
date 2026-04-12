@echo off
echo Starting MaterialForge evaluation in HEURISTIC-ONLY mode...
echo No LLM will be used; the agent will rely solely on its structural heuristic brain.
python analytical_benchmark.py --trials 100 --mode heuristic
pause
