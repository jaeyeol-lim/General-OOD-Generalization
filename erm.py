"""Train ERM on DrugOOD IC50."""

import argparse

try:
    from .common import add_common_args, train
except ImportError:  # Direct execution: python3 erm.py
    from common import add_common_args, train


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    train("erm", parser.parse_args())


if __name__ == "__main__":
    main()

