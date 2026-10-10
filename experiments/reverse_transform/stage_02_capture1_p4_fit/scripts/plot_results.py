"""Plot capture 1 diagnostics using the common saved-results plotter."""
import importlib.util
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
path=ROOT/'experiments/reverse_transform/stage_03_independent_captures/scripts/plot_results.py'
spec=importlib.util.spec_from_file_location('independent_plots',path)
plots=importlib.util.module_from_spec(spec);spec.loader.exec_module(plots)

if __name__=='__main__':plots.main()
