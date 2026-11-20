"""Build RAFT wind covariates from historical ECMWF IFS runs served by Open-Meteo."""
import argparse
from pathlib import Path
import sys

# Support invocation by path as well as python -m scripts.build_raft_weather.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasource import build_raft_weather_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_raft_weather_archive(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
