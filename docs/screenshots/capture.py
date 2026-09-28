"""Regenerate the README screenshots from real CLI runs.

Requires: pip install playwright. Set CHROMIUM_PATH to use a system Chromium.
Run from the repo root: python3 docs/screenshots/capture.py
"""
from __future__ import annotations

import html
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;padding:24px;background:#0b1020;font-family:'DejaVu Sans Mono',monospace}}
.win{{background:#0f172a;border:1px solid #1e293b;border-radius:10px;overflow:hidden;width:max-content;min-width:760px}}
.bar{{background:#1e293b;padding:9px 12px;display:flex;gap:7px;align-items:center}}
.bar i{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.bar span{{color:#94a3b8;font-size:12px;margin-left:10px}}
pre{{margin:0;padding:16px 20px;color:#e2e8f0;font-size:13px;line-height:1.45}}
.p{{color:#22d3ee}}
</style></head><body><div class="win"><div class="bar">
<i style="background:#ef4444"></i><i style="background:#f59e0b"></i><i style="background:#22c55e"></i>
<span>JivaHarness</span></div><pre>{body}</pre></div></body></html>"""


def run(display: str, argv: list[str]) -> str:
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return f'<span class="p">$</span> {html.escape(display)}\n{html.escape(proc.stdout.rstrip())}'


def main() -> None:
    state = Path(tempfile.mkdtemp())
    py = sys.executable
    shots = {
        "loop-demo.png": run(
            "python3 -m jiva_harness.cli loop-demo --state-dir .demo-state",
            [py, "-m", "jiva_harness.cli", "loop-demo", "--state-dir", str(state / "loop")],
        ),
        "demo.png": run(
            "python3 -m jiva_harness.cli demo --state-dir .demo-state",
            [py, "-m", "jiva_harness.cli", "demo", "--state-dir", str(state / "demo")],
        ),
        "tests.png": run("python3 -m pytest", [py, "-m", "pytest", "-p", "no:cacheprovider"]),
    }
    with sync_playwright() as p:
        # CHROMIUM_PATH lets you point at a system Chromium when Playwright's own build is absent.
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        page = browser.new_page(device_scale_factor=2)
        for name, body in shots.items():
            page.set_content(PAGE.format(body=body))
            page.locator(".win").screenshot(path=str(OUT / name))
        page.set_viewport_size({"width": 1500, "height": 900})
        page.goto((ROOT / "docs" / "data-flow-architecture.html").as_uri(), wait_until="networkidle")
        page.screenshot(path=str(OUT / "data-flow-architecture.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
