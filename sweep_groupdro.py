"""Search the GroupDRO step size over the prescribed grid."""

try:
    from .sweep_common import run_sweep
except ImportError:  # Direct execution: python3 sweep_groupdro.py
    from sweep_common import run_sweep


if __name__ == "__main__":
    run_sweep("groupdro", "--step-size", (1.0, 1e-1, 1e-2))
