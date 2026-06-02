# conftest.py
# -----------
# Pytest configuration — sits at the project root.
#
# Adds the project root to sys.path so all imports work correctly
# when pytest runs on Windows. Without this, pytest can't find any
# project modules because Windows doesn't automatically add the
# project root the way Linux/Mac does.
#
# Picked up automatically by pytest — no extra config needed.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))