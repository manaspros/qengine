import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

M08 = Path.home() / "models" / "Qwen3.5-0.8B"
M27 = Path.home() / "models" / "Qwen3.8-27B"
PROMPT = ROOT / "bench" / "prompts" / "prose.txt"
CACHE = ROOT / "build" / "testdata"
