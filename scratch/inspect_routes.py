import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from server.app import app

print("\n--- APP ROUTES ---")
for route in app.routes:
    path = getattr(route, 'path', 'N/A')
    name = getattr(route, 'name', 'N/A')
    print(f"Path: {path} | Name: {name} | Type: {type(route).__name__}")
print("------------------\n")
