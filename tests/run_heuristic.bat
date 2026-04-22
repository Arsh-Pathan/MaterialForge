@echo off
# Launches the MaterialForge evaluation using the Pure Heuristic brain.
# No external LLM or API keys are required for this mode.
echo Starting MaterialForge evaluation in HEURISTIC-ONLY mode...
echo No LLM will be used; the agent will rely solely on its structural heuristic brain.
python ./tests/benchmark.py --trials 100 --mode heuristic
pause
