"""Search the IRM penalty weight over the prescribed grid."""

try:
    from .sweep_common import run_sweep
except ImportError:  # Direct execution: python3 sweep_irm.py
    from sweep_common import run_sweep


if __name__ == "__main__":
    run_sweep("irm", "--penalty-weight", (1e-2, 1e-1, 1.0, 1e1))
