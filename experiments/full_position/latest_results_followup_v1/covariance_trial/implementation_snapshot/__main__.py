"""Reproducible command-line entry points; python -m full_position --help."""
import argparse
from pathlib import Path
from .validate import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-per-fixation", type=int, default=48,
                        help="Evenly spaced original rows; 0 uses all core rows")
    parser.add_argument("--eval-per-fixation", type=int, default=8,
                        help="Evenly spaced original rows including invalid rows; 0 uses all core rows")
    parser.add_argument("--folds", nargs="+", choices=["gaze", "capture", "full"], default=["gaze", "capture"])
    parser.add_argument("--only-fold")
    parser.add_argument("--calibration-starts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--prior", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args.root, args.output, args.train_per_fixation, args.eval_per_fixation, args.folds,
        args.calibration_starts, args.max_nfev, args.prior, args.seed, args.only_fold)


if __name__ == "__main__":
    main()
