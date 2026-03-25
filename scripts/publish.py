"""
Interactive MQTT publisher — send free text to the robot controller and
see the parsed command response in real time.

Requires: MQTT broker + robot controller running (src/main.py)

Usage:
    uv run python scripts/publish.py
    uv run python scripts/publish.py --broker 192.168.1.5
    uv run python scripts/publish.py --robot-id robot-02
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AppSettings

import aiomqtt


async def main(broker: str, robot_id: str, timeout: int) -> None:
    input_topic = f"robot/{robot_id}/input/text"
    commands_topic = f"robot/{robot_id}/commands/parsed"
    chat_topic = f"robot/{robot_id}/chat/response"

    print(f"Broker:   {broker}:1883")
    print(f"Robot:    {robot_id}")
    print(f"Topic:    {input_topic}")
    print("Type text and press Enter. Ctrl+C or 'quit' to exit.\n")

    async with aiomqtt.Client(hostname=broker) as client:
        # Subscribe to all response topics for this robot
        await client.subscribe(f"{commands_topic}/#")
        await client.subscribe(f"{chat_topic}/#")

        async def _recv(tid: str) -> None:
            async for message in client.messages:
                topic = str(message.topic)
                if tid not in topic:
                    continue
                data = json.loads(message.payload.decode())
                if "commands" in topic:
                    cmds = data.get("commands", [])
                    if not cmds:
                        print("[no commands]")
                    for cmd in cmds:
                        if cmd.get("command") == "conditional":
                            print(f"[conditional]  if: {cmd['condition']}")
                            for c in cmd.get("then_commands", []):
                                slot = f" | {c['data_type']}={c['data']}" if c.get("data") else ""
                                print(f"               then → {c['command']}{slot}")
                            for c in cmd.get("else_commands") or []:
                                slot = f" | {c['data_type']}={c['data']}" if c.get("data") else ""
                                print(f"               else → {c['command']}{slot}")
                        else:
                            slot = f" | {cmd['data_type']}={cmd['data']}" if cmd.get("data") else ""
                            uid = f" (user_id={cmd['user_id']})" if cmd.get("user_id") else ""
                            print(f"[command]      {cmd['command']}{slot}{uid}")
                else:
                    print(f"[conversation] {data.get('response', '')}")
                return

        async def listen_for_response(tid: str) -> None:
            try:
                await asyncio.wait_for(_recv(tid), timeout=timeout)
            except asyncio.TimeoutError:
                print(f"[timeout]      no response after {timeout}s — is the controller running?")

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

                tid = str(uuid.uuid4())[:8]
                payload = json.dumps({
                    "transaction_id": tid,
                    "text": text,
                    "timestamp": time.time(),
                })
                await client.publish(input_topic, payload, qos=1)
                await listen_for_response(tid)
                print()

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    settings = AppSettings()

    parser = argparse.ArgumentParser(description="Publish free text to the robot controller via MQTT")
    parser.add_argument("--broker", default="127.0.0.1", help="MQTT broker hostname")
    parser.add_argument("--robot-id", default=settings.robot_id, help="Robot ID")
    parser.add_argument("--timeout", type=int, default=35, help="Seconds to wait for a response")
    args = parser.parse_args()

    asyncio.run(main(args.broker, args.robot_id, args.timeout))
