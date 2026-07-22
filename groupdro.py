"""Train GroupDRO on DrugOOD IC50."""

import argparse

try:
    from .common import add_common_args, train
except ImportError:  # Direct execution: python3 groupdro.py
    from common import add_common_args, train


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Group-weight update step size; sweep over 1.0, 1e-1, 1e-2.",
    )
    train("groupdro", parser.parse_args())


if __name__ == "__main__":
    main()
