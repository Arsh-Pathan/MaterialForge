@echo off
set API_BASE_URL=http://localhost:11434/v1
set API_KEY=ollama
set MODEL_NAME=qwen3.5:latest
echo Starting MaterialForge 100-Test Analytical Suite with Ollama...
echo Model: %MODEL_NAME%
echo API: %API_BASE_URL%
python analytical_benchmark.py --trials 100 --mode llm
pause
