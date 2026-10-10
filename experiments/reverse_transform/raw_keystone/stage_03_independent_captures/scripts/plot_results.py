"""Stage 3 wrapper for the common raw-keystone plot_results.py entry point."""
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "docs/Theory.md").is_file())
COMMON = ROOT / "experiments/reverse_transform/raw_keystone/scripts/plot_results.py"
sys.argv = [str(COMMON), "--stage", "3", *sys.argv[1:]]
runpy.run_path(str(COMMON), run_name="__main__")
