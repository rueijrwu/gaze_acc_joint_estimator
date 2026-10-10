"""Run permanent contracts for stages 1–4; save outcomes and source hashes."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({'test': test.id(), 'status': 'passed'})
    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append({'test': test.id(), 'status': 'failed', 'detail': self._exc_info_to_string(error, test)})
    def addError(self, test, error):
        super().addError(test, error)
        self.records.append({'test': test.id(), 'status': 'error', 'detail': self._exc_info_to_string(error, test)})
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({'test': test.id(), 'status': 'skipped', 'reason': reason})

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit('Refusing to overwrite test evidence')
started = time.perf_counter()
suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_stage*.py')
result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
sources = sorted(list((ROOT/'distortion_model').glob('*.py'))+list((ROOT/'tests').glob('*.py')))
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'backend': 'NumPy float64 CPU',
          'tolerances': {'stage1': {'rtol': 1e-10, 'atol': 1e-9},
                         'stage2_finite_difference': {'rtol': 3e-6, 'atol': 1e-5},
                         'stage3': {'rtol': 1e-10, 'atol': 1e-9},
                         'stage4': {'rtol': 3e-6, 'atol': 1e-5}},
          'total': result.testsRun,
          'passed': sum(r['status']=='passed' for r in result.records),
          'failed': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
          'tests': result.records, 'elapsed_seconds': time.perf_counter()-started,
          'source_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
sys.exit(not result.wasSuccessful())
