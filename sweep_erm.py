"""Run ERM across domains and random seeds (ERM has no method-specific grid)."""

try:
    from .sweep_common import run_sweep
except ImportError:  # Direct execution: python3 sweep_erm.py
    from sweep_common import run_sweep


if __name__ == "__main__":
    run_sweep("erm")

