"""Capture 1 P4 inverse-keystone diagnostic with fixed audited P1 scale."""
import importlib.util
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
path=ROOT/'experiments/reverse_transform/stage_03_independent_captures/scripts/run.py'
spec=importlib.util.spec_from_file_location('independent_runner',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

if __name__=='__main__':
    sys.argv.append('--capture1-only')
    runner.main()
