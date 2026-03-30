"""Add project root to sys.path so viz.* and openpi.* are importable."""
import sys
from pathlib import Path

# Project root = openpi/
_root = Path(__file__).parent.parent.parent.parent
for p in [str(_root), str(_root / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
