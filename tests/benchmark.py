# Analytical Benchmark Suite for MaterialForge.
# This script runs batch evaluations to provide "Evidence of Improvement" for the hackathon.

import asyncio
import os
import sys
import random
import csv
import json
import statistics
import time
import subprocess
import socket
from datetime import datetime
from typing import List, Dict

# Standardize environment variables to prevent quoting errors from terminal shells.
for key in ["API_BASE_URL", "API_KEY", "MODEL_NAME"]:
    val = os.environ.get(key)
    if val:
        # Strip literal quotes that CMD sometimes includes
        os.environ[key] = val.strip('"').strip("'")

# Configuration defaults for local testing with Ollama.
if "API_BASE_URL" not in os.environ:
    os.environ["API_BASE_URL"] = "http://127.0.0.1:11434/v1"
if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = "ollama"
if "MODEL_NAME" not in os.environ:
    os.environ["MODEL_NAME"] = "qwen3.5:latest"


# Ensure the root project directory is in the path for internal imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference import (
    MaterialForgeEnv, 
    run_episode, 
    start_environment_server,
    TASKS as ORIGINAL_TASKS,
    MODEL_NAME, 
    SUCCESS_THRESHOLD
)
from openai import OpenAI

# Connectivity utility: checks if a specific local port is available for the server.
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# Automation: automatically launches the Ollama inference server if not already running.
def start_ollama_server():
    if is_port_open(11434):
        print("[INFO] Ollama server is already running on port 11434.")
        return None
    
    print("[INFO] Ollama server not detected. Starting 'ollama serve'...")
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == 'nt' else 0
        )
        
        # Wait up to 30 seconds for Ollama to initialize.
        start_time = time.time()
        while time.time() - start_time < 30:
            if is_port_open(11434):
                print("[INFO] Ollama server started successfully.")
                time.sleep(2)
                return proc
            time.sleep(1)
            
        print("[WARNING] Ollama server start timed out. It may still be warming up.")
        return proc
    except Exception as e:
        print(f"[ERROR] Failed to start Ollama server: {e}")
        return None

# Main Execution Engine: manages the full benchmarking lifecycle (Startup -> Trials -> Analysis).
async def run_analytical_suite(num_trials: int = 100, mode: str = "llm"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"tests/benchmarks/benchmark_{mode}_{timestamp}.csv"
    summary_file = f"tests/benchmarks/summary_{mode}_{timestamp}.txt"
    
    # Ensure the benchmarks directory exists before saving logs.
    os.makedirs("tests/benchmarks", exist_ok=True)
    
    ollama_proc = None
    if mode == "llm":
        ollama_proc = start_ollama_server()

    print(f"📊 Starting Analytical Benchmark Task ({mode.upper()} mode)")
    print(f"🕒 Timestamp: {timestamp}")
    print(f"📝 Logging results to: {report_file}")
    
    api_url = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("API_KEY", "ollama")
    
    # Client initialization: bypasses LLM if mode is set to 'heuristic'.
    client = OpenAI(base_url=api_url, api_key=api_key) if mode == "llm" else None
    if mode == "heuristic":
        print("🛠 Mode: Pure Heuristic (LLM Bypassed)")
    else:
        print(f"🤖 Model: {os.getenv('MODEL_NAME', MODEL_NAME)}")

    # Environment Setup: launches the FastAPI simulator on port 7860.
    server_proc = start_environment_server(port=7860)
    env = MaterialForgeEnv(base_url="http://127.0.0.1:7860")
    
    try:
        await env.connect()
    except Exception as e:
        print(f"❌ Connection failed: {e}. Is the server running?")
        if server_proc: server_proc.terminate()
        return

    results = []
    successes = 0
    
    print("\n" + "="*50)
    print(f"{'Trial':<8} | {'Seed':<8} | {'Score':<8} | {'Status':<10}")
    print("-" * 50)

    # Trial Loop: iterates through random seeds to evaluate agent stability and generalization.
    with open(report_file, mode='w', newline='') as csvfile:
        fieldnames = ['trial', 'seed', 'score', 'success', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, num_trials + 1):
            seed = random.randint(1000, 9999)
            template = random.choice(ORIGINAL_TASKS)
            task = {
                "name": f"analytical-{i}",
                "difficulty": template["difficulty"],
                "seed": seed
            }

            try:
                # Executes the episode and captures the normalized performance score.
                score = await run_episode(env, client, task)
                
                is_success = score >= SUCCESS_THRESHOLD
                results.append(score)
                if is_success:
                    successes += 1
                    
                status = "✅ PASS" if is_success else "❌ FAIL"
                print(f"{i:<8} | {seed:<8} | {score:<8.3f} | {status:<10}")
                
                writer.writerow({
                    'trial': i, 'seed': seed, 'score': f"{score:.3f}",
                    'success': is_success, 'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Trial {i} Error: {e}")

    # Aggregated Analysis: computes high-level metrics for the final presentation.
    summary_data = [
        "="*50,
        "📈 FINAL ANALYTICAL REPORT",
        "="*50,
        f"Mode:            {mode.upper()}",
        f"Model:           {os.getenv('MODEL_NAME', MODEL_NAME) if mode == 'llm' else 'Heuristic-Only'}",
        f"Total Trials:    {num_trials}",
        f"Success Rate:    {(successes/num_trials)*100:.1f}%",
        f"Average Score:   {statistics.mean(results):.3f}",
        f"Max Score:       {max(results):.3f}",
        f"Min Score:       {min(results):.3f}",
        "="*50
    ]

    report_text = "\n".join(summary_data)
    print("\n" + report_text)
    
    with open(summary_file, 'w') as f:
        f.write(report_text)

    # Resource Cleanup: shuts down the environment server and local inference engine.
    await env.close()
    if server_proc:
        server_proc.terminate()
        server_proc.wait()
    
    if ollama_proc:
        print("[INFO] Terminating local Ollama server...")
        ollama_proc.terminate()
        try:
            ollama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ollama_proc.kill()

# Entry point for the CLI benchmark tool.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--mode", type=str, choices=["llm", "heuristic"], default="llm")
    args = parser.parse_args()

    asyncio.run(run_analytical_suite(args.trials, mode=args.mode))
