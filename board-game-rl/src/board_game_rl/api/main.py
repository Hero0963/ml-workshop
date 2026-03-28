"""
FastAPI application for Board Game RL API.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from board_game_rl.api.inference import get_optimal_move

app = FastAPI(
    title="Board Game RL API",
    description="A simple API to serve a reinforcement learning model for Tic-Tac-Toe.",
    version="1.0.0",
)


class GameStateRequest(BaseModel):
    board: list[list[int]]
    current_player: int
    agent_type: str = "Alpha-Beta Pruning (完美大師)"


class MoveResponse(BaseModel):
    action: int


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Board Game RL Model API! Visit /docs for API documentation."
    }


@app.post("/predict", response_model=MoveResponse)
async def predict(request: GameStateRequest):
    """
    Endpoint to get predictions (optimal moves) from the RL model.
    """
    action = get_optimal_move(request.board, request.current_player, request.agent_type)
    return MoveResponse(action=action)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
