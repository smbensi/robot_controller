"""
Interactive test for the LLM command extraction pipeline.
No MQTT or MongoDB required — talks directly to llama-server.

Usage:
    uv run python scripts/chat.py
    uv run python scripts/chat.py --url http://192.168.1.10:8080
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm_adapter import LLMAdapter
from src.config import AppSettings
from src.models import CommandDefinition, ConditionalCommand


async def main(base_url: str | None) -> None:
    # Load settings (can be overridden via .env or env vars)
    settings = AppSettings()
    if base_url:
        settings.llm.base_url = base_url

    # Load command definitions
    commands_path = Path(settings.commands_file)
    raw = json.loads(commands_path.read_text())
    cmd_list = raw if isinstance(raw, list) else raw.get("commands", [])
    command_defs = [CommandDefinition(**c) for c in cmd_list]

    llm = LLMAdapter(settings, command_defs)

    # Health check
    healthy = await llm.health_check()
    status = "ok" if healthy else "unreachable (will still try)"
    print(f"llama-server @ {settings.llm.base_url}  [{status}]")
    print(f"Commands loaded: {', '.join(c.name for c in command_defs)}")
    print("Type free text and press Enter. Ctrl+C or 'quit' to exit.\n")

    loop = asyncio.get_event_loop()

    async def read_line() -> str:
        return await loop.run_in_executor(None, lambda: input("> "))

    try:
        while True:
            try:
                text = (await read_line()).strip()
            except EOFError:
                break
            if not text or text.lower() in {"quit", "exit", "q"}:
                break

            batch = await llm.extract_commands(text)

            if batch.is_conversation:
                print(f"[conversation] {batch.conversation_response}\n")
            else:
                for cmd in batch.commands:
                    if isinstance(cmd, ConditionalCommand):
                        print(f"[conditional]  if: {cmd.condition}")
                        for c in cmd.then_commands:
                            print(f"               then: {c.command}"
                                  + (f" | {c.data_type}={c.data}" if c.data else ""))
                        if cmd.else_commands:
                            for c in cmd.else_commands:
                                print(f"               else: {c.command}"
                                      + (f" | {c.data_type}={c.data}" if c.data else ""))
                    else:
                        slot = f" | {cmd.data_type}={cmd.data}" if cmd.data else ""
                        print(f"[command]      {cmd.command}{slot}")
                print()
    finally:
        await llm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive command extraction test")
    parser.add_argument("--url", help="llama-server base URL (overrides LLM_BASE_URL)")
    args = parser.parse_args()
    asyncio.run(main(args.url))
