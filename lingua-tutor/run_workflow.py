# run_workflow.py
import subprocess
import sys
from loguru import logger


def run_command(command: list[str], step_name: str):
    """
    Runs a command, logs the process, and exits on failure.

    Args:
        command: The command to run as a list of strings.
        step_name: A descriptive name for the step being run.
    """
    logger.info(f"--- Running Step: {step_name} ---")
    try:
        # Use Popen to stream output in real-time, which is better for user experience
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        # Read and print output line by line as it comes
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                # Print the output from the subprocess directly
                print(line, end="")

        # Wait for the process to complete
        process.wait()

        # Check if the process exited with an error
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        logger.error(f"Step '{step_name}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(
            f"Error: Command '{command[0]}' not found. Is Python in the system's PATH?"
        )
        sys.exit(1)
    logger.success(f"--- Finished Step: {step_name} ---")


def main():
    """
    Runs the full GPU check, STT, and evaluation workflow.
    """
    logger.info(">>> Starting Full Workflow <<<")

    run_command(command=["python", "-m", "src.stt.check_gpu"], step="GPU Health Check")

    run_command(
        command=["python", "-m", "src.stt.whisper_stt"], step="STT (Speech-to-Text)"
    )

    run_command(
        command=["python", "-m", "src.evaluation.text_evaluator"], step="Evaluation"
    )

    logger.success(">>> Workflow Finished Successfully <<<")


if __name__ == "__main__":
    main()
