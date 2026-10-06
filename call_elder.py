"""Dial a specific elder by name, looked up from elders.json.

Usage:  python call_elder.py <name>
Example: python call_elder.py Raja

Needs the agent worker running separately:  python agent.py dev
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys

from dotenv import load_dotenv
from livekit import api

load_dotenv(dotenv_path=".env.local")

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
AGENT_NAME = os.getenv("DISPATCH_AGENT_NAME", "outbound-caller")
ELDERS_FILE = os.getenv("ELDERS_FILE", "elders.json")


def find_elder_number(name: str) -> str | None:
    with open(ELDERS_FILE, "r", encoding="utf-8") as f:
        elders = json.load(f)
    name_lower = name.strip().lower()
    for phone_number, profile in elders.items():
        if profile.get("name", "").strip().lower() == name_lower:
            return phone_number
    return None


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python call_elder.py <name>")
        sys.exit(1)

    name = " ".join(sys.argv[1:])
    phone_number = find_elder_number(name)
    if phone_number is None:
        print(f"No elder named '{name}' found in {ELDERS_FILE}")
        sys.exit(1)

    room = f"call-{secrets.token_hex(4)}"
    metadata = json.dumps({"phone_number": phone_number})

    async with api.LiveKitAPI(
        url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET
    ) as lk:
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME, room=room, metadata=metadata
            )
        )

    print(f"Dialing {name} ({phone_number}) in room {room}")


if __name__ == "__main__":
    asyncio.run(main())
