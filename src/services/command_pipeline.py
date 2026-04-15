"""
Command Pipeline Service: orchestrates the full flow from text input to parsed commands.
Text → LLM extraction → Entity resolution → MQTT publish
"""

from __future__ import annotations

import asyncio
from typing import List

from src.adapters import AmbiguousEntityError, EntityNotFoundError, EntityResolver, LLMAdapter, MQTTAdapter
from src.config import AppSettings
from src.models import (
    CommandBatch,
    CommandDefinition,
    CommandResponse,
    ConditionalCommand,
    ParsedMessage,
    SimpleCommand,
)
from src.utils import get_logger

logger = get_logger(__name__)


class CommandPipeline:
    """
    Main service: receives text, extracts commands via LLM,
    resolves entities via MongoDB, and publishes results via MQTT.
    """

    def __init__(
        self,
        settings: AppSettings,
        llm: LLMAdapter,
        entities: EntityResolver,
        mqtt: MQTTAdapter,
        command_defs: List[CommandDefinition],
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._entities = entities
        self._mqtt = mqtt
        self._command_defs = command_defs

        # Build lookup: command_name -> CommandDefinition
        self._cmd_lookup = {d.name: d for d in command_defs}

        # Queue for backpressure
        self._queue: asyncio.Queue[ParsedMessage] = asyncio.Queue(
            maxsize=settings.queue_maxsize
        )
        # Semaphore to serialize LLM calls (GPU is the bottleneck)
        self._llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm)

    async def enqueue(self, message: ParsedMessage) -> None:
        """Add a message to the processing queue."""
        try:
            self._queue.put_nowait(message)
            logger.info("message_enqueued", tid=message.transaction_id, qsize=self._queue.qsize())
        except asyncio.QueueFull:
            logger.warning("queue_full", tid=message.transaction_id)
            await self._mqtt.publish_chat(
                message.transaction_id,
                "I'm currently busy processing other requests. Please try again shortly.",
            )

    async def run(self) -> None:
        """Main processing loop — drains the queue."""
        logger.info("pipeline_started")
        while True:
            message = await self._queue.get()
            try:
                await self._process(message)
            except Exception as exc:
                logger.error(
                    "pipeline_error",
                    tid=message.transaction_id,
                    error=str(exc),
                    exc_info=True,
                )
                await self._mqtt.publish_chat(
                    message.transaction_id,
                    "An error occurred while processing your request.",
                )
            finally:
                self._queue.task_done()

    async def _process(self, message: ParsedMessage) -> None:
        """Process a single message through the full pipeline."""
        logger.info("processing_start", tid=message.transaction_id, text=message.text[:100])

        # 1. LLM inference (serialized via semaphore)
        # Timeout is handled per-request by the httpx client inside LLMAdapter.
        # wait_for is intentionally omitted here: conversational turns now make
        # two LLM calls (command detection + history-aware reply), so a single
        # fixed timeout would fire prematurely on slow hardware (e.g. Jetson).
        async with self._llm_semaphore:
            batch = await self._llm.extract_commands(message.text)

        # 2. If conversational, publish the response
        if batch.is_conversation:
            if batch.tool_response and batch.conversation_response:
                # Response was built from a data tool result (news, weather, time…).
                # Use it directly — calling stream_reply would make a fresh LLM call
                # without the tool context and produce a wrong/empty answer.
                await self._mqtt.publish_chat(message.transaction_id, batch.conversation_response)
                logger.info("tool_response_published", tid=message.transaction_id)
            else:
                # Pure conversation — use streaming history-aware reply
                async for chunk in self._llm.stream_reply(message.text):
                    await self._mqtt.publish_chat_chunk(message.transaction_id, chunk)
                await self._mqtt.publish_chat_chunk(message.transaction_id, "", done=True)
                logger.info("conversation_streamed", tid=message.transaction_id)
            return

        # 3. Entity resolution — abort with a user-facing message on any resolution failure
        try:
            for cmd in batch.commands:
                await self._resolve_entities(cmd)
        except AmbiguousEntityError as exc:
            names = ", ".join(exc.matches)
            msg = (
                f"I found multiple {exc.slot_type} matching '{exc.name}': {names}. "
                "Please use the full name to be more specific."
            )
            logger.warning("entity_ambiguous", slot_type=exc.slot_type, name=exc.name, matches=exc.matches)
            await self._mqtt.publish_chat(message.transaction_id, msg)
            return
        except EntityNotFoundError as exc:
            if exc.unknown_type:
                msg = f"I don't know how to look up '{exc.slot_type}'."
            else:
                msg = f"I couldn't find '{exc.name}'."
            logger.warning("entity_not_found_response", slot_type=exc.slot_type, name=exc.name)
            await self._mqtt.publish_chat(message.transaction_id, msg)
            return

        # 4. Build and publish response
        commands_dicts = []
        for cmd in batch.commands:
            if isinstance(cmd, ConditionalCommand):
                commands_dicts.append({
                    "command": "conditional",
                    "condition": cmd.condition,
                    "then_commands": [self._cmd_to_dict(c) for c in cmd.then_commands],
                    "else_commands": (
                        [self._cmd_to_dict(c) for c in cmd.else_commands]
                        if cmd.else_commands
                        else None
                    ),
                    "status": cmd.status.value,
                })
            else:
                commands_dicts.append(self._cmd_to_dict(cmd))

        response = CommandResponse(
            transaction_id=message.transaction_id,
            robot_id=message.robot_id,
            commands=commands_dicts,
        )

        await self._mqtt.publish_commands(response)
        logger.info(
            "processing_complete",
            tid=message.transaction_id,
            n_commands=len(batch.commands),
        )

    async def _resolve_entities(self, cmd: SimpleCommand | ConditionalCommand) -> None:
        """Resolve entity references in a command using MongoDB."""
        if isinstance(cmd, ConditionalCommand):
            for sub_cmd in cmd.then_commands:
                await self._resolve_entities(sub_cmd)
            if cmd.else_commands:
                for sub_cmd in cmd.else_commands:
                    await self._resolve_entities(sub_cmd)
            return

        # Look up the command definition to know the slot types
        cmd_def = self._cmd_lookup.get(cmd.command)
        if not cmd_def or cmd_def.n_slots == 0 or not cmd.data:
            return

        if len(cmd_def.slots_type) == 1:
            # Single collection — must match or raise
            slot_type = cmd_def.slots_type[0]
            resolved_id = await self._entities.resolve_by_type(slot_type, cmd.data)
            cmd.resolved_ids[slot_type] = resolved_id
        else:
            # Multiple collections — search each in order, use the first hit.
            # AmbiguousEntityError propagates immediately (found many, need specificity).
            # EntityNotFoundError on a type is skipped; if ALL types miss, re-raise the last one.
            last_error: EntityNotFoundError | None = None
            for slot_type in cmd_def.slots_type:
                try:
                    resolved_id = await self._entities.resolve_by_type(slot_type, cmd.data)
                    cmd.resolved_ids[slot_type] = resolved_id
                    logger.info(
                        "entity_resolved_via_multi_slot",
                        slot_type=slot_type,
                        name=cmd.data,
                    )
                    return  # first match wins — stop searching
                except AmbiguousEntityError:
                    raise  # ambiguity is always a hard stop
                except EntityNotFoundError as exc:
                    last_error = exc  # try next collection
            raise last_error  # exhausted all collections

    @staticmethod
    def _cmd_to_dict(cmd: SimpleCommand) -> dict:
        """Convert a SimpleCommand to the output dict format."""
        result = {
            "command": cmd.command,
            "data_type": cmd.data_type,
            "data": cmd.data,
            "status": cmd.status.value,
        }
        # Add resolved IDs generically: slot_type "users" → key "user_id", etc.
        for slot_type, entity_id in cmd.resolved_ids.items():
            result[f"{slot_type.rstrip('s')}_id"] = entity_id
        return result
