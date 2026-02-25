"""
Deprecated compatibility wrapper.

Use generate_figures.py as the single authoritative entrypoint.
"""

from __future__ import annotations

import warnings

import sys

from generate_figures import main


if __name__ == "__main__":
    warnings.warn(
        "evaluate.py is deprecated. Use generate_figures.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    # Backward compatibility: old script used --fig; new script uses --figs.
    if "--fig" in sys.argv and "--figs" not in sys.argv:
        idx = sys.argv.index("--fig")
        sys.argv[idx] = "--figs"
    main()
