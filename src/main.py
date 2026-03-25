"""
Main entry point for the Robot Command Controller.
Orchestrates startup, shutdown, and the async event loop.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path

from src.adapters import EntityResolver, LLMAdapter, MQTTAdapter
from src.config import AppSettings
from src.models import CommandDefinition
from src.services import CommandPipeline
from src.utils import get_logger, setup_logging


async def main() -> None:
    # 1. Load configuration
    settings = AppSettings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)
    logger = get_logger("main")
    logger.info("starting", robot_id=settings.robot_id)

    # 2. Load command definitions
    commands_path = Path(settings.commands_file)
    if not commands_path.exists():
        logger.error("commands_file_not_found", path=str(commands_path))
        sys.exit(1)

    with open(commands_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support both {"commands": [...]} and bare [...]
    cmd_list = raw if isinstance(raw, list) else raw.get("commands", [])
    command_defs = [CommandDefinition(**c) for c in cmd_list]
    logger.info("commands_loaded", count=len(command_defs))

    # 3. Initialize adapters
    entity_resolver = EntityResolver(settings)
    llm = LLMAdapter(settings, command_defs)
    mqtt = MQTTAdapter(settings)

    # 4. Connect to MongoDB (optional — only needed for entity resolution)
    try:
        await entity_resolver.connect()
    except Exception as exc:
        logger.warning("mongodb_unavailable_continuing", error=str(exc))
        logger.warning("entity_resolution_disabled_call_command_will_lack_user_id")

    # 5. Check LLM health
    if not await llm.health_check():
        logger.error("llm_not_available", url=settings.llm.base_url)
        logger.warning("llm_continuing_without_health_check")

    # 6. Build pipeline
    pipeline = CommandPipeline(
        settings=settings,
        llm=llm,
        entities=entity_resolver,
        mqtt=mqtt,
        command_defs=command_defs,
    )

    # 7. Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int, _frame) -> None:
        logger.info("shutdown_signal_received", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # 8. Run pipeline consumer + MQTT listener concurrently
    pipeline_task = asyncio.create_task(pipeline.run())
    mqtt_task = asyncio.create_task(mqtt.listen(on_message=pipeline.enqueue))

    logger.info("controller_ready", robot_id=settings.robot_id)

    # Wait for shutdown signal
    await shutdown_event.wait()

    # 9. Graceful shutdown
    logger.info("shutting_down")
    pipeline_task.cancel()
    mqtt_task.cancel()

    try:
        await asyncio.gather(pipeline_task, mqtt_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    await llm.close()
    await entity_resolver.close()
    logger.info("shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
