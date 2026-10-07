"""Tiny web frontend to trigger outbound calls.

Serves a one-page UI with a call button. Clicking it creates a LiveKit agent
dispatch, which makes the `outbound-caller` worker place the call. The LiveKit
API key/secret stay on the server and are never sent to the browser.

Run:  python server.py          (then open http://localhost:8080)
Needs the agent worker running too:  python agent.py dev
"""

from __future__ import annotations

import base64
import json
import os
import secrets

from aiohttp import web
from dotenv import load_dotenv
from livekit import api

load_dotenv(dotenv_path=".env.local")

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

AGENT_NAME = os.getenv("DISPATCH_AGENT_NAME", "outbound-caller")
DEFAULT_NUMBER = os.getenv("OUTBOUND_PHONE_NUMBER", "0740525967")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PORT = int(os.getenv("PORT", "8080"))

# HTTP basic-auth password for the whole console (any username). Without it
# anyone who can reach the page can make the SIP trunk dial any number, so set
# it on every hosted deployment. Unset = no auth, for local use only.
CONSOLE_PASSWORD = os.getenv("CONSOLE_PASSWORD", "")


def _lkapi() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )


async def index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"))


async def config(request: web.Request) -> web.Response:
    return web.json_response({"default_number": DEFAULT_NUMBER})


async def start_call(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    number = str(body.get("number") or DEFAULT_NUMBER).strip()
    if not number:
        return web.json_response({"error": "no number"}, status=400)

    room = f"web-call-{secrets.token_hex(4)}"
    metadata = json.dumps({"phone_number": number})

    try:
        async with _lkapi() as lk:
            await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME, room=room, metadata=metadata
                )
            )
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)

    return web.json_response({"room": room, "number": number})


async def call_status(request: web.Request) -> web.Response:
    room = request.query.get("room", "")
    if not room:
        return web.json_response({"state": "unknown"})

    try:
        async with _lkapi() as lk:
            participants = await lk.room.list_participants(
                api.ListParticipantsRequest(room=room)
            )
    except Exception:
        # room no longer exists -> call is over
        return web.json_response({"state": "ended"})

    sip_status = None
    for p in participants.participants:
        attrs = dict(p.attributes)
        if "sip.callStatus" in attrs:
            sip_status = attrs["sip.callStatus"]

    if sip_status == "active":
        state = "active"
    elif sip_status in ("ringing", "dialing", "automation"):
        state = "ringing"
    elif participants.participants:
        state = "connecting"
    else:
        state = "ended"

    return web.json_response({"state": state, "sip": sip_status})


async def hangup(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    room = str(body.get("room") or "").strip()
    if not room:
        return web.json_response({"error": "no room"}, status=400)
    try:
        async with _lkapi() as lk:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room))
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)
    return web.json_response({"ok": True})


async def healthz(request: web.Request) -> web.Response:
    return web.Response(text="ok")


@web.middleware
async def require_password(request: web.Request, handler):
    if not CONSOLE_PASSWORD or request.path == "/healthz":
        return await handler(request)
    header = request.headers.get("Authorization", "")
    if header.startswith("Basic "):
        try:
            _, _, supplied = base64.b64decode(header[6:]).decode().partition(":")
        except Exception:
            supplied = ""
        if secrets.compare_digest(supplied, CONSOLE_PASSWORD):
            return await handler(request)
    return web.Response(
        status=401, headers={"WWW-Authenticate": 'Basic realm="call console"'}
    )


def build_app() -> web.Application:
    app = web.Application(middlewares=[require_password])
    app.add_routes(
        [
            web.get("/healthz", healthz),
            web.get("/", index),
            web.get("/api/config", config),
            web.post("/api/call", start_call),
            web.get("/api/status", call_status),
            web.post("/api/hangup", hangup),
            web.static("/web", WEB_DIR),
        ]
    )
    return app


if __name__ == "__main__":
    print(f"call console:  http://localhost:{PORT}")
    print(f"dispatching agent '{AGENT_NAME}', default number {DEFAULT_NUMBER}")
    if not CONSOLE_PASSWORD:
        print("WARNING: CONSOLE_PASSWORD is not set - the console is open to anyone who can reach it")
    web.run_app(build_app(), host="0.0.0.0", port=PORT)
