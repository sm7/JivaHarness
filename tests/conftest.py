import os
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parent.parent / "src")


@pytest.fixture(autouse=True)
def _src_on_subprocess_path(monkeypatch):
    # pyproject's pythonpath only patches this process; CLI tests spawn `python -m`.
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", SRC + (os.pathsep + existing if existing else ""))
