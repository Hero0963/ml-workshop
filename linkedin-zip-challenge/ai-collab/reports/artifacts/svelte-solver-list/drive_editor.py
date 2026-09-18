# ai-collab/reports/artifacts/svelte-solver-list/drive_editor.py
"""Drives the built Svelte editor in headless Chrome and records what a user would see.

Checks the three things the editor's solver list has to get right: the dropdown is the
registry (read from the page, never from a list here), every solver in it can be run from
the page with a drawn answer that really is a solution, and a solver that gives up is shown
as giving up rather than under "Solution".

Needs the app serving the built editor (`npm run build`, then e.g.
`uv run uvicorn src.app.main:app --port 7452`). Run from the project root:
`uv run python ai-collab/reports/artifacts/svelte-solver-list/drive_editor.py <scratch_dir>`
where `scratch_dir` is outside the repo: screenshots and Chrome's throwaway profile go there.
"""

import ast
import asyncio
import base64
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import websockets
from loguru import logger

from src.core.solvers.verify import is_solution
from src.core.utils import parse_puzzle_layout

APP_URL = "http://127.0.0.1:7452/svelte-ui/"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
CDP_PORT = 9333
OUTPUT_DIR = Path(__file__).parent
VIEWPORT = {"width": 1280, "height": 1100}
CHROME_READY_TIMEOUT_SECONDS = 15
#: The heuristics' budget is 5s; the RL solver samples up to 32 times.
SOLVE_TIMEOUT_MS = 30_000

BOARD_SIZE = 4
#: A snake through a 4x4 board visits 1, 2, 3, 4 in order, so it has a solution.
NUMBERS = {(0, 0): "1", (1, 3): "2", (2, 0): "3", (3, 0): "4"}
#: Walling (3, 3) off from both neighbours leaves it unreachable: no solution exists.
ISOLATING_WALLS = [((3, 2), (3, 3)), ((2, 3), (3, 3))]

#: Mirrors the editor's own coordinates (`cell_size`, `margin` in Index.svelte).
PAGE_HELPERS = r"""
window.__e2e = {
  CELL: 50, MARGIN: 10,
  sleep: ms => new Promise(r => setTimeout(r, ms)),
  async clickCanvas(x, y) {
    const canvas = document.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    canvas.dispatchEvent(new MouseEvent('click', {clientX: rect.left + x, clientY: rect.top + y, bubbles: true}));
    await this.sleep(60);
  },
  async setSize(n) {
    for (const input of document.querySelectorAll('input.control-input')) {
      input.value = String(n);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      await this.sleep(60);
    }
  },
  async setCell(r, c, value) {
    await this.clickCanvas(this.MARGIN + c * this.CELL + this.CELL / 2, this.MARGIN + r * this.CELL + this.CELL / 2);
    const input = document.querySelector('.cell-input');
    input.value = value;
    input.dispatchEvent(new Event('blur'));
    await this.sleep(60);
  },
  async toggleWall(a, b) {
    const [r1, c1] = a, [r2, c2] = b;
    const edge = 3;
    if (r1 === r2) {
      // Side by side: click just inside the right cell's left border.
      await this.clickCanvas(this.MARGIN + c2 * this.CELL + edge, this.MARGIN + r2 * this.CELL + this.CELL / 2);
    } else {
      // Stacked: click just inside the lower cell's top border.
      await this.clickCanvas(this.MARGIN + c2 * this.CELL + this.CELL / 2, this.MARGIN + r2 * this.CELL + edge);
    }
  },
  dropdown() {
    const select = document.querySelector('select');
    return {
      disabled: select.disabled,
      selected: select.value,
      groups: [...select.querySelectorAll('optgroup')].map(g => ({
        label: g.label, options: [...g.querySelectorAll('option')].map(o => o.value)})),
      notes: [...document.querySelectorAll('.control-group .solver-note')].map(p => p.textContent),
    };
  },
  async choose(name) {
    const select = document.querySelector('select');
    select.value = name;
    select.dispatchEvent(new Event('change', {bubbles: true}));
    await this.sleep(60);
  },
  async solve(timeoutMs) {
    const button = [...document.querySelectorAll('button')].find(b => /Solv/.test(b.textContent));
    const started = performance.now();
    button.click();
    await this.sleep(150);
    while (button.disabled && performance.now() - started < timeoutMs) await this.sleep(100);
    return {
      ms: Math.round(performance.now() - started),
      boxes: [...document.querySelectorAll('.solution-box')].map(b => ({
        heading: b.querySelector('h3') ? b.querySelector('h3').textContent : null,
        text: b.innerText,
        images: [...b.querySelectorAll('img')].map(i => (i.getAttribute('src') || '').slice(0, 22)),
      })),
    };
  },
};
"""


class Page:
    """The few CDP calls this needs, over one websocket."""

    def __init__(self, socket) -> None:
        self._socket = socket
        self._ids = itertools.count(1)

    async def send(self, method: str, params: dict | None = None) -> dict:
        message_id = next(self._ids)
        await self._socket.send(
            json.dumps({"id": message_id, "method": method, "params": params or {}})
        )
        while True:
            reply = json.loads(await self._socket.recv())
            if reply.get("id") == message_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply["result"]

    async def js(self, expression: str):
        result = await self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    async def screenshot(self, path: Path) -> None:
        shot = await self.send("Page.captureScreenshot", {"format": "png"})
        path.write_bytes(base64.b64decode(shot["data"]))


def _layout() -> list[list[str]]:
    """The board as the editor sends it: two-character cells, '  ' for empty."""
    return [
        [NUMBERS.get((r, c), "  ") for c in range(BOARD_SIZE)]
        for r in range(BOARD_SIZE)
    ]


def _verified(result: dict, walls: set) -> bool | None:
    """Parses the drawn path back from the page and checks it; None when nothing was drawn."""
    solved = [box for box in result["boxes"] if box["heading"] == "Solution"]
    if not solved:
        return None
    path_text = solved[0]["text"].split("Path:", 1)[1].split("\n", 1)[0].strip()
    path = [ast.literal_eval(step.strip()) for step in path_text.split("->")]
    puzzle = parse_puzzle_layout(_layout())
    puzzle["walls"] = walls
    return is_solution(puzzle, path)


def _row(name: str, result: dict, walls: set) -> dict:
    return {
        "solver": name,
        "seconds": round(result["ms"] / 1000, 2),
        "headings": [box["heading"] for box in result["boxes"]],
        "drew_gif": any(
            image.startswith("data:image/gif")
            for box in result["boxes"]
            for image in box["images"]
        ),
        "verified": _verified(result, walls),
        "text": [box["text"] for box in result["boxes"]],
    }


async def _solve_each(page: Page, names: list[str], walls: set) -> list[dict]:
    rows = []
    for name in names:
        await page.js(f"window.__e2e.choose({json.dumps(name)})")
        result = await page.js(f"window.__e2e.solve({SOLVE_TIMEOUT_MS})")
        rows.append(_row(name, result, walls))
        logger.info(f"{name}: {rows[-1]['headings']} in {result['ms']} ms")
    return rows


async def _drive(websocket_url: str, screenshot_dir: Path) -> dict:
    async with websockets.connect(websocket_url, max_size=None) as socket:
        page = Page(socket)
        await page.send("Page.enable")
        await page.send(
            "Emulation.setDeviceMetricsOverride",
            {**VIEWPORT, "deviceScaleFactor": 1, "mobile": False},
        )
        await page.send("Page.navigate", {"url": APP_URL})

        deadline = time.monotonic() + CHROME_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            loaded = await page.js(
                "document.readyState === 'complete' && "
                "!!document.querySelector('select option, .error-text')"
            )
            if loaded:
                break
            await asyncio.sleep(0.2)
        await page.js(PAGE_HELPERS)

        dropdown = await page.js("window.__e2e.dropdown()")
        await page.screenshot(screenshot_dir / "01-dropdown.png")

        await page.js(f"window.__e2e.setSize({BOARD_SIZE})")
        for (r, c), value in NUMBERS.items():
            await page.js(f"window.__e2e.setCell({r}, {c}, '{value}')")

        every_solver = [
            name for group in dropdown["groups"] for name in group["options"]
        ]
        solvable = await _solve_each(page, every_solver, set())
        await page.screenshot(screenshot_dir / "02-solvable-last-solver.png")

        for a, b in ISOLATING_WALLS:
            await page.js(f"window.__e2e.toggleWall({list(a)}, {list(b)})")
        first_of_each_kind = [group["options"][0] for group in dropdown["groups"]]
        unsolvable = []
        for name in first_of_each_kind:
            unsolvable += await _solve_each(page, [name], set(ISOLATING_WALLS))
            slug = "".join(ch for ch in name.lower() if ch.isalnum())
            await page.screenshot(screenshot_dir / f"03-unsolvable-{slug}.png")

    return {
        "dropdown": dropdown,
        "solvable_board": solvable,
        "unsolvable_board": unsolvable,
    }


def _wait_for_page_target() -> str:
    deadline = time.monotonic() + CHROME_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            targets = requests.get(
                f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=1
            ).json()
            pages = [target for target in targets if target["type"] == "page"]
            if pages:
                return pages[0]["webSocketDebuggerUrl"]
        except requests.RequestException:
            pass
        time.sleep(0.3)
    raise RuntimeError("Chrome did not expose a page over CDP")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    screenshot_dir = Path(sys.argv[1])
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    profile = screenshot_dir / "chrome-profile"

    chrome = subprocess.Popen(
        [
            str(CHROME),
            "--headless=new",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "about:blank",
        ]
    )
    try:
        record = asyncio.run(_drive(_wait_for_page_target(), screenshot_dir))
    finally:
        chrome.terminate()
        chrome.wait(timeout=10)

    record = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "app_url": APP_URL,
        "board": {"layout": _layout(), "isolating_walls": ISOLATING_WALLS},
        **record,
    }
    output = OUTPUT_DIR / "editor-e2e.json"
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
