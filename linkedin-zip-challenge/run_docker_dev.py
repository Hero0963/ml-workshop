# run_docker_dev.py
import subprocess
import time
import urllib.request
import os

# --- Configuration ---
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = os.getenv("APP_PORT", "7440")
SVELTE_PORT = os.getenv("SVELTE_PORT", "5173")  # Restored for dev environment
HEALTHCHECK_URL = f"http://{APP_HOST}:{APP_PORT}/api/echo/health"
HEALTHCHECK_TIMEOUT = 30  # seconds
HEALTHCHECK_INTERVAL = 3  # seconds
# ---


def run_command(command: str, description: str):
    """Runs a command and prints its description, streaming output in real-time."""
    # Prepend the docker-compose command with the dev file flag
    if command.startswith("docker compose"):
        command = command.replace(
            "docker compose", "docker compose -f docker-compose.dev.yml"
        )

    print(f"--- {description} ---")
    try:
        # Using shell=True to handle complex commands. Output is streamed directly.
        subprocess.run(command, shell=True, check=True, text=True)
        print(f"SUCCESS: {description}\n")
    except subprocess.CalledProcessError as e:
        # The output has already been streamed to the console.
        print(
            f"ERROR: Failed to {description}. Command returned non-zero exit status {e.returncode}."
        )
        exit(1)


def check_service_health():
    """Polls the healthcheck endpoint until the service is ready."""
    print(f"--- Waiting for service to be available at {HEALTHCHECK_URL} ---")
    start_time = time.time()
    while time.time() - start_time < HEALTHCHECK_TIMEOUT:
        try:
            # Use a short timeout for the request itself
            with urllib.request.urlopen(HEALTHCHECK_URL, timeout=2) as response:
                if response.status == 200:
                    print("\nSUCCESS: Service is healthy and responding.\n")
                    return
        except Exception:
            # Silently ignore connection errors, timeouts, etc. and retry
            print(".", end="", flush=True)
        time.sleep(HEALTHCHECK_INTERVAL)

    print(
        f"\nERROR: Service failed to become healthy within {HEALTHCHECK_TIMEOUT} seconds."
    )
    print(
        "Please check the container logs with: docker compose logs -f zip-challenge-app"
    )
    exit(1)


def main():
    """Main function to orchestrate the deployment."""
    # Step 1: Clean up previous run
    run_command(
        "docker compose down --remove-orphans",
        "Stopping and removing existing containers",
    )

    # Step 2: Build and start all services in detached mode
    run_command(
        "docker compose up --build -d",
        "Building and starting all services in the background",
    )

    # Step 3: Start the main application process inside the container.
    # This command runs the server in the background, similar to the manual steps.
    app_start_command = (
        '''bash -c "source .venv/bin/activate && python -m src.app.main"'''
    )
    run_command(
        f"docker compose exec -d zip-challenge-app {app_start_command}",
        "Starting FastAPI server in the background",
    )

    # Step 4: Wait for the backend service to become healthy
    check_service_health()

    print("\n🚀 Deployment script finished successfully! 🚀")
    print("The application is now running in the background.")
    print(f"- Gradio UI (main app): http://{APP_HOST}:{APP_PORT}/ui")
    print(f"- Svelte UI (hot-reload): http://{APP_HOST}:{SVELTE_PORT}")
    print(f"- Svelte UI (integrated): http://{APP_HOST}:{APP_PORT}/svelte-ui")
    print("\nTo monitor logs, run:")
    print("docker compose -f docker-compose.dev.yml logs -f zip-challenge-app")
    print("docker compose -f docker-compose.dev.yml logs -f svelte_frontend_dev")


if __name__ == "__main__":
    main()
