# run_docker_dev.py
import subprocess
import sys
from loguru import logger


def run_command(command: list[str], step_name: str):
    """Runs a command, logs the process, and exits on failure."""
    logger.info(f"--- Running Step: {step_name} ---")
    logger.info(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Step '{step_name}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(
            f"Error: Command '{command[0]}' not found. Is Docker installed and in the PATH?"
        )
        sys.exit(1)


def main():
    """
    Builds and starts the development Docker container.
    Accepts an optional '--no-cache' argument to force a rebuild without cache.
    """
    args = sys.argv[1:]
    compose_file = "docker-compose.dev.yml"

    if "--no-cache" in args:
        # Two-step process for no-cache builds
        build_cmd = ["docker", "compose", "-f", compose_file, "build", "--no-cache"]
        run_command(build_cmd, "Build (no cache)")

        up_cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
        run_command(up_cmd, "Up")
    else:
        # Default single-step process for building and starting
        up_cmd = ["docker", "compose", "-f", compose_file, "up", "--build", "-d"]
        # Pass through any other arguments the user might have provided
        up_cmd.extend(args)
        run_command(up_cmd, "Build and Up")

    logger.success("\nDocker container operation completed.")
    logger.info(
        f"To attach to the container, run: docker compose -f {compose_file} exec app bash"
    )


if __name__ == "__main__":
    main()
