"""
Application configuration using pydantic-settings.
All values can be overridden via environment variables or .env file.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MQTTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MQTT_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str = "robot-command-controller"
    keepalive: int = 60
    qos: int = 1


class MQTT2Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MQTT2_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str = "robot-command-controller-2"
    keepalive: int = 60
    qos: int = 1


class MongoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONGO_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    uri: str = "mongodb://127.0.0.1:27017"
    database: str = "xtend_robotics"
    # Maps command slot_type → MongoDB collection name.
    # Add new entity types here without touching any other code.
    # Override via env: MONGO_COLLECTIONS='{"users":"users","locations":"locations","rooms":"rooms"}'
    collections: dict[str, str] = Field(
        default={"users": "users", "locations": "locations"}
    )


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    _BACKEND_DEFAULTS: ClassVar[dict] = {
        "mlc": {
            "base_url": "http://0.0.0.0:9000",
            "model": "Qwen2.5-7B-Instruct-q4f16_1-MLC",
        },
        "llama": {
            "base_url": "http://0.0.0.0:8080",
            "model": "qwen2.5-7b-instruct-q8_0",
        },
    }

    backend: Literal["mlc", "llama"] = "mlc"
    base_url: str = ""
    model: str = ""
    timeout: float = 30.0
    max_tokens: int = 1024
    temperature: float = 0.1
    grammar_path: str = "grammars/commands.gbnf"
    system_prompt_path: str = "data/system_prompt.txt"
    max_history_pairs: int = 10  # max conversational turns to keep in context

    @model_validator(mode="after")
    def apply_backend_defaults(self) -> "LLMSettings":
        defaults = self._BACKEND_DEFAULTS[self.backend]
        if not self.base_url:
            self.base_url = defaults["base_url"]
        if not self.model:
            self.model = defaults["model"]
        return self


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    robot_id: str = "robot-01"
    log_level: str = "INFO"
    log_format: str = "json"  # "json" for production, "console" for dev
    commands_file: str = "data/commands.json"
    entity_cache_maxsize: int = 500
    entity_cache_ttl: int = 300
    queue_maxsize: int = 50
    max_concurrent_llm: int = 1

    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
