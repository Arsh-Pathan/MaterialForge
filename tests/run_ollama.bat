@echo off
# Launches the MaterialForge analytical suite using a local Ollama server.
# Defaults to the Qwen3.5 model for high-performance reasoning.
set API_BASE_URL=http://localhost:11434/v1
set API_KEY=ollama
set MODEL_NAME=qwen3.5:latest
echo Starting MaterialForge 100-Test Analytical Suite with Ollama...
echo Model: %MODEL_NAME%
echo API: %API_BASE_URL%
python ./tests/benchmark.py --trials 100 --mode llm
pause
