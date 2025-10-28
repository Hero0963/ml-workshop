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

    ollama_model_name: str
    ollama_provider_url: str


@cache
def get_settings() -> Settings:
    """Returns a cached instance of the Settings object.

    The @cache decorator ensures that the Settings are loaded only once.
    """
    return Settings()
