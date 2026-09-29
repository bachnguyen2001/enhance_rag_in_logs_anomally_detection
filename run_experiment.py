"""Compatibility wrapper for the current experiment CLI.

Prefer running the package directly:

    .venv/bin/python -m experiments.run --help

This wrapper is kept only so older notes that call `run_experiment.py` fail less
surprisingly; it delegates all arguments to `experiments.run`.
"""

from experiments.run import main


if __name__ == "__main__":
    main()
