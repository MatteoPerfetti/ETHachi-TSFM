"""Context-length sweep: run the rolling backtest of the TSFMs for several context lengths and compare them.

Defaults come from config.py (CONTEXT_SWEEP_DAYS, CONTEXT_SWEEP_MODELS, BACKTEST_ISSUE_EVERY_DAYS);
command-line options override them, e.g. one context per job on a cluster.

    python context_sweep.py                          # config defaults
    python context_sweep.py --days 3 7 --reuse       # skip the models already saved for these settings
    python context_sweep.py --days 30 --device cuda  # one context on a GPU
    python context_sweep.py --days 3 7 --every 7     # quick run: one forecast every 7 days
    python context_sweep.py --days 3 7 14 --table-only   # only rebuild the table from saved runs
"""
import argparse

import config
from src.context_sweep import report, run_sweep, sweep_settings
from src.validate_config import validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, nargs="+", default=config.CONTEXT_SWEEP_DAYS,
                        help="context lengths in days (default: CONTEXT_SWEEP_DAYS)")
    parser.add_argument("--models", nargs="+", default=config.CONTEXT_SWEEP_MODELS, choices=config.TSFM_MODELS,
                        help="TSFM keys (default: CONTEXT_SWEEP_MODELS)")
    parser.add_argument("--every", type=int, default=config.BACKTEST_ISSUE_EVERY_DAYS,
                        help="one forecast every N days of the test period (default: BACKTEST_ISSUE_EVERY_DAYS)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="default: cuda if available")
    parser.add_argument("--reuse", action="store_true", help="skip models already saved with the same settings")
    parser.add_argument("--table-only", action="store_true", help="do not run anything, only build the table")
    args = parser.parse_args()
    if args.every < 1 or min(args.days) < 1:
        parser.error("--days and --every must be >= 1")

    validate_config()
    days = sorted(set(args.days))
    if args.table_only:
        paths = {d: config.BACKTEST_RESULTS_DIR / f"{sweep_settings(d, args.every).run_name}.parquet" for d in days}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            parser.error(f"no saved run for: {missing}")
    else:
        paths = run_sweep(days, args.models, issue_every_days=args.every, device=args.device, reuse=args.reuse)
    report(paths, args.models, tag=f"_every{args.every}d" if args.every != 1 else "")


if __name__ == "__main__":
    main()
