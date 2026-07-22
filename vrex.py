"""Train V-REx on DrugOOD IC50."""

import argparse

try:
    from .common import add_common_args, train
except ImportError:  # Direct execution: python3 vrex.py
    from common import add_common_args, train


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=1.0,
        help="V-REx risk-variance weight; sweep over 1e-2, 1e-1, 1, 1e1.",
    )
    parser.add_argument(
        "--penalty-anneal-steps",
        type=int,
        default=500,
        help="Optimizer steps before applying the searched penalty weight.",
    )
    train("vrex", parser.parse_args())


if __name__ == "__main__":
    main()
