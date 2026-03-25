"""
Application configuration using pydantic-settings.
All values can be overridden via environment variables or .env file.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MQTTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MQTT_")

    host: str = "127.0.0.1"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    client_id: str = "robot-command-controller"
    keepalive: int = 60
    qos: int = 1


class MongoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONGO_")

    uri: str = "mongodb://127.0.0.1:27017"
    database: str = "xtend_robotics"
    # Maps command slot_type → MongoDB collection name.
    # Add new entity types here without touching any other code.
    # Override via env: MONGO_COLLECTIONS='{"users":"users","locations":"locations","rooms":"rooms"}'
    collections: dict[str, str] = Field(
        default={"users": "users", "locations": "locations"}
    )


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    base_url: str = "http://127.0.0.1:8080"
    model: str = "qwen2.5-7b-instruct"
    timeout: float = 30.0
    max_tokens: int = 1024
    temperature: float = 0.1
    grammar_path: str = "grammars/commands.gbnf"
    system_prompt_path: str = "data/system_prompt.txt"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
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
