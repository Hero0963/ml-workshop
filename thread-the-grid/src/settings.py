# src/settings.py
from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Defines the application settings.

    It automatically reads environment variables or from a .env file.
    """

    # pydantic-settings will automatically look for a .env file and load it.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    # Define settings variables with type hints and default values.
    app_port: int = 7440
    app_host: str = "127.0.0.1"
    svelte_port: int = 5173

    ollama_model_name: str = ""
    ollama_provider_url: str = ""

    # Which prompt the vision endpoint pairs with `ollama_model_name`. It belongs next
    # to the model tag rather than being guessed from it: the two must match what the
    # checkpoint was trained on, and a tag can be renamed freely.
    #   finetune -> the short instruction the P4c run trained against
    #   sized    -> the best measured setting for an un-finetuned model
    vision_prompt_variant: str = "finetune"

    # Where /api/vision/solve keeps a copy of each image and what the model said, in the
    # same shape as a training dataset so a hand-written `label` makes a line scoreable.
    # Empty disables logging entirely. The path is relative to the project root.
    vision_log_dir: str = "logs/vision"


@cache
def get_settings() -> Settings:
    """Returns a cached instance of the Settings object.

    The @cache decorator ensures that the Settings are loaded only once.
    """
    return Settings()
