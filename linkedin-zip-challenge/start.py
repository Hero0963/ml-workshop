# start.py
"""One command from a fresh clone to a running stack.

    python start.py            # production stack (code baked into the image)
    python start.py --dev      # development stack (hot reload + Svelte dev server)
    python start.py --down     # stop everything
    python start.py --status   # what is running, and what it can and cannot do

It does the four things a newcomer would otherwise have to know about: create `.env` from
the template, bring the compose stack up, wait until the API actually answers, and say
plainly which optional pieces are missing and what that costs. Nothing here is clever --
the point is that `docker compose up` alone does not tell you whether the service works,
and this does.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_TEMPLATE = PROJECT_ROOT / ".env.example"
PROD_COMPOSE = "docker-compose.yml"
DEV_COMPOSE = "docker-compose.dev.yml"

DEFAULT_APP_PORT = "7440"
DEFAULT_OLLAMA_PORT = "11435"
DEFAULT_SVELTE_PORT = "5173"

HEALTH_TIMEOUT_SECONDS = 180
HEALTH_POLL_SECONDS = 3
OLLAMA_TIMEOUT_SECONDS = 60

# Where the learned solver looks for weights. Absent on a fresh clone: `models/` is far
# too large for version control, so the RL solver answers 503 until something trains.
RL_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "rl_a2"
    / "bc_multi_456"
    / "checkpoints"
    / "model_final.zip"
)


def say(message: str) -> None:
    print(message, flush=True)


def read_env() -> dict[str, str]:
    """Reads `.env` well enough to know which ports to poll. Not a dotenv parser."""
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def ensure_docker() -> None:
    if shutil.which("docker") is None:
        sys.exit(
            "Docker is not on PATH. Install Docker Desktop and start it, then retry."
        )
    probe = subprocess.run(
        ["docker", "compose", "version"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        sys.exit(
            "`docker compose` is unavailable (Docker daemon not running, or Compose V2 "
            f"missing).\n{probe.stderr.strip()}"
        )


def ensure_env_file() -> None:
    if ENV_FILE.exists():
        return
    if not ENV_TEMPLATE.exists():
        sys.exit(
            f"Neither {ENV_FILE.name} nor {ENV_TEMPLATE.name} exists; cannot continue."
        )
    shutil.copyfile(ENV_TEMPLATE, ENV_FILE)
    say(
        f"Created {ENV_FILE.name} from {ENV_TEMPLATE.name}. Edit it if you need other ports."
    )


def compose(args: list[str], compose_file: str, check: bool = True) -> int:
    command = ["docker", "compose", "-f", compose_file, *args]
    say(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if check and result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}.")
    return result.returncode


def wait_for_health(port: str) -> bool:
    url = f"http://127.0.0.1:{port}/api/echo/health"
    say(f"Waiting for {url}")
    deadline = time.time() + HEALTH_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    say("\nAPI is up.")
                    return True
        except Exception:
            print(".", end="", flush=True)
        time.sleep(HEALTH_POLL_SECONDS)
    say("\nThe API did not become healthy in time.")
    say("Look at the logs:  docker compose logs -f zip-challenge-app")
    return False


def report_ollama(port: str) -> None:
    """Reports the vision model's availability. Never fatal: only /api/vision needs it."""
    url = f"http://127.0.0.1:{port}/api/tags"
    deadline = time.time() + OLLAMA_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                names = [
                    m["name"] for m in json.loads(response.read()).get("models", [])
                ]
                say(f"Ollama is up. Models: {names or '(none pulled yet)'}")
                return
        except Exception:
            time.sleep(HEALTH_POLL_SECONDS)
    say(
        "Ollama did not answer. `Solve from Screenshot` will be unavailable; everything "
        "else works. Logs:  docker compose logs -f ollama"
    )


def report_rl_weights() -> None:
    if RL_CHECKPOINT.exists():
        say(f"RL solver weights found ({RL_CHECKPOINT.relative_to(PROJECT_ROOT)}).")
    else:
        say(
            "No RL checkpoint under models/ -- the `RL (behaviour cloning)` solver will "
            "answer 503. Every other solver works. See ai-collab/deployment-guide.md."
        )


def print_urls(env: dict[str, str], dev: bool) -> None:
    app_port = env.get("APP_PORT", DEFAULT_APP_PORT)
    say("")
    say("  Gradio console   http://127.0.0.1:%s/ui" % app_port)
    # The dev stack mounts ./src over the image, and the host has no built `dist/`, so
    # the app's /svelte-ui is a 404 there; the editor is served by vite instead.
    if dev:
        say(
            "  Svelte editor    http://127.0.0.1:%s/svelte-ui/  (vite, hot reload)"
            % env.get("SVELTE_PORT", DEFAULT_SVELTE_PORT)
        )
    else:
        say("  Svelte editor    http://127.0.0.1:%s/svelte-ui/" % app_port)
    say("  API docs         http://127.0.0.1:%s/docs" % app_port)
    say("")
    say("  Stop with:  python start.py --down%s" % (" --dev" if dev else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="Hot-reloading dev stack.")
    parser.add_argument(
        "--down", action="store_true", help="Stop and remove containers."
    )
    parser.add_argument(
        "--status", action="store_true", help="Report without starting."
    )
    parser.add_argument("--no-build", action="store_true", help="Skip the image build.")
    args = parser.parse_args()

    ensure_docker()
    compose_file = DEV_COMPOSE if args.dev else PROD_COMPOSE
    env = read_env()

    if args.down:
        compose(["down", "--remove-orphans"], compose_file)
        say("Stopped.")
        return

    if args.status:
        compose(["ps"], compose_file, check=False)
        report_rl_weights()
        return

    ensure_env_file()
    env = read_env()

    say("Building and starting. The first build downloads a few GB and takes a while.")
    up = ["up", "-d"] if args.no_build else ["up", "-d", "--build"]
    compose(up, compose_file)

    healthy = wait_for_health(env.get("APP_PORT", DEFAULT_APP_PORT))
    report_ollama(env.get("OLLAMA_HOST_PORT", DEFAULT_OLLAMA_PORT))
    report_rl_weights()
    if not healthy:
        sys.exit(1)
    print_urls(env, args.dev)


if __name__ == "__main__":
    main()
