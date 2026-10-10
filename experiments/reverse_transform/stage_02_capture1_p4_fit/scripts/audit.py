"""Audit the capture 1 saved fit using the common reference audit."""
import importlib.util
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
path=ROOT/'experiments/reverse_transform/stage_03_independent_captures/scripts/audit.py'
spec=importlib.util.spec_from_file_location('independent_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

if __name__=='__main__':audit.main()
