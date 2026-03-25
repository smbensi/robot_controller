"""
MQTT Adapter: handles all MQTT communication using aiomqtt.
Subscribes to input topics, publishes parsed commands and chat responses.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, Callable, Awaitable

import aiomqtt

from src.config import AppSettings
from src.models import CommandResponse, ParsedMessage
from src.utils import get_logger

logger = get_logger(__name__)


class MQTTAdapter:
    """Async MQTT client for robot command communication."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings.mqtt
        self._robot_id = settings.robot_id
        self._client: aiomqtt.Client | None = None

    @property
    def input_topic(self) -> str:
        return f"robot/{self._robot_id}/input/text"

    @property
    def commands_topic(self) -> str:
        return f"robot/{self._robot_id}/commands/parsed"

    @property
    def chat_topic(self) -> str:
        return f"robot/{self._robot_id}/chat/response"

    @property
    def status_topic(self) -> str:
        return f"robot/{self._robot_id}/status"

    def _build_client(self) -> aiomqtt.Client:
        """Create a new aiomqtt Client instance."""
        kwargs: dict = {
            "hostname": self._settings.host,
            "port": self._settings.port,
            "identifier": self._settings.client_id,
            "keepalive": self._settings.keepalive,
        }
        if self._settings.username:
            kwargs["username"] = self._settings.username
        if self._settings.password:
            kwargs["password"] = self._settings.password

        # Last Will and Testament: publish offline status if disconnected
        kwargs["will"] = aiomqtt.Will(
            topic=self.status_topic,
            payload=json.dumps({"status": "offline", "robot_id": self._robot_id}),
            qos=1,
            retain=True,
        )

        return aiomqtt.Client(**kwargs)

    async def listen(
        self,
        on_message: Callable[[ParsedMessage], Awaitable[None]],
    ) -> None:
        """
        Connect, subscribe, and dispatch incoming messages.
        Reconnects automatically on disconnection.
        """
        while True:
            try:
                async with self._build_client() as client:
                    self._client = client

                    # Publish online status (retained)
                    await client.publish(
                        self.status_topic,
                        json.dumps({
                            "status": "online",
                            "robot_id": self._robot_id,
                            "timestamp": time.time(),
                        }),
                        qos=1,
                        retain=True,
                    )

                    await client.subscribe(self.input_topic, qos=self._settings.qos)
                    logger.info("mqtt_subscribed", topic=self.input_topic)

                    async for message in client.messages:
                        try:
                            payload = message.payload.decode("utf-8")
                            data = json.loads(payload)
                            text = data.get("text", payload)
                            tid = data.get("transaction_id", str(uuid.uuid4()))

                            parsed = ParsedMessage(
                                transaction_id=tid,
                                robot_id=self._robot_id,
                                text=text,
                                timestamp=time.time(),
                            )
                            await on_message(parsed)

                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            logger.warning("mqtt_invalid_message", error=str(exc))

            except aiomqtt.MqttError as exc:
                logger.error("mqtt_connection_lost", error=str(exc))
                import asyncio
                await asyncio.sleep(5)  # Reconnect backoff

    async def publish_commands(self, response: CommandResponse) -> None:
        """Publish parsed commands to the commands topic."""
        if self._client is None:
            logger.error("mqtt_not_connected")
            return

        topic = f"{self.commands_topic}/{response.transaction_id}"
        payload = response.model_dump_json()

        await self._client.publish(topic, payload, qos=self._settings.qos)
        logger.info(
            "mqtt_commands_published",
            tid=response.transaction_id,
            n_commands=len(response.commands),
        )

    async def publish_chat(self, tid: str, text: str) -> None:
        """Publish a conversational response."""
        if self._client is None:
            logger.error("mqtt_not_connected")
            return

        payload = json.dumps({
            "transaction_id": tid,
            "robot_id": self._robot_id,
            "response": text,
            "timestamp": time.time(),
        })

        await self._client.publish(
            f"{self.chat_topic}/{tid}",
            payload,
            qos=self._settings.qos,
        )
        logger.info("mqtt_chat_published", tid=tid)
