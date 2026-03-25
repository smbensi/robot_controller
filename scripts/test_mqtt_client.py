#!/usr/bin/env python3
"""
MQTT test client — send text commands and listen for parsed results.
Usage: python3 scripts/test_mqtt_client.py "call John"
"""

import asyncio
import json
import sys
import time
import uuid

import aiomqtt

ROBOT_ID = "robot-01"
BROKER = "localhost"


async def main(text: str) -> None:
    tid = str(uuid.uuid4())[:8]
    print(f"Sending: '{text}' (tid={tid})")

    async with aiomqtt.Client(hostname=BROKER) as client:
        # Subscribe to responses
        await client.subscribe(f"robot/{ROBOT_ID}/commands/parsed/{tid}")
        await client.subscribe(f"robot/{ROBOT_ID}/chat/response/{tid}")

        # Publish input
        payload = json.dumps({
            "transaction_id": tid,
            "text": text,
            "timestamp": time.time(),
        })
        await client.publish(f"robot/{ROBOT_ID}/input/text", payload, qos=1)
        print(f"Published to robot/{ROBOT_ID}/input/text")

        # Wait for response (timeout 35s)
        async def _recv() -> None:
            async for message in client.messages:
                topic = str(message.topic)
                data = json.loads(message.payload.decode())
                print(f"\n{'='*60}")
                print(f"Response on: {topic}")
                print(json.dumps(data, indent=2))
                print(f"{'='*60}")
                return

        try:
            await asyncio.wait_for(_recv(), timeout=35)
        except asyncio.TimeoutError:
            print("Timeout — no response received within 35 seconds.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_mqtt_client.py <text>")
        print('  Example: python3 test_mqtt_client.py "call John"')
        print('  Example: python3 test_mqtt_client.py "take a photo and then take vitals"')
        print('  Example: python3 test_mqtt_client.py "if someone is here, take a photo"')
        print('  Example: python3 test_mqtt_client.py "what time is it?"')
        sys.exit(1)

    asyncio.run(main(" ".join(sys.argv[1:])))
