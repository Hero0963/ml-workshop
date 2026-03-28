#!/bin/bash

# Start FastAPI in the background with hot-reload
echo "Starting FastAPI on port 8000..."
uv run uvicorn board_game_rl.api.main:app --host 0.0.0.0 --port 8000 --reload &

# Start Gradio in the background with auto-reload enabled via env var
echo "Starting Gradio on port 7860..."
GRADIO_WATCH_DIRS=src uv run python -m board_game_rl.ui.gradio_app &


# Keep the script running to keep the container alive
wait -n

exit $?
