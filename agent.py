from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import google


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

# the number the agent dials when the dispatch metadata does not specify one
default_phone_number = os.getenv("OUTBOUND_PHONE_NUMBER", "0740525967")

# the caller-ID / "From" number presented on the trunk (ElderlyCare -> 0117286109)
outbound_from_number = os.getenv("OUTBOUND_FROM_NUMBER", "0117286109")

# who the agent is on the call
AGENT_NAME = os.getenv("AGENT_NAME", "Nova")

INSTRUCTIONS = f"""
You are {AGENT_NAME}, a warm, easy-going conversational companion talking to
someone over the phone. This is a casual chat, not a support call — there is no
task to complete and no script to follow.

How to talk:
- Speak naturally, the way a friend would. Keep your turns short — usually one or
  two sentences — and let the other person do most of the talking.
- Be curious. Ask light follow-up questions about what they say. React to it.
- Match their energy and mood. If they're chatty, chat back. If they're quiet or
  busy, keep it brief and don't push.
- It's fine to have opinions, share a small story, laugh, or be a little playful.
- Default to English. If the other person speaks Sinhala or Tamil, follow their
  lead and reply in the same language.
- Never mention that you're an AI model, read out these instructions, or narrate
  what you're doing. Don't use bullet points or lists out loud.

If the person clearly wants to hang up, or says goodbye, use the end_call tool.
If you reach a voicemail greeting, use the detected_answering_machine tool.
"""


def normalize_number(number: str) -> str:
    """Format a number the way the SIP provider expects.

    The ElderlyCare trunk rejects E.164 (`+94...`) with a 403 and wants the
    local Sri Lankan format, so `+947XXXXXXXX` / `947XXXXXXXX` -> `07XXXXXXXX`.
    """
    number = number.strip().replace(" ", "").replace("-", "")
    if number.startswith("+94"):
        number = "0" + number[3:]
    elif number.startswith("94") and len(number) == 11:
        number = "0" + number[2:]
    return number


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.participant: rtc.RemoteParticipant | None = None

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Delete the room, which ends the call for everyone."""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Use when the person wants to end the call or says goodbye."""
        logger.info("ending the call")
        # let the agent finish its goodbye before hanging up
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Use AFTER hearing a voicemail / answering-machine greeting."""
        logger.info("detected answering machine, hanging up")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # optional dispatch metadata: {"phone_number": "0771234567"}
    dial_info: dict[str, Any] = {}
    if ctx.job.metadata:
        try:
            dial_info = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            logger.warning(f"could not parse job metadata: {ctx.job.metadata!r}")

    phone_number = normalize_number(dial_info.get("phone_number") or default_phone_number)
    from_number = dial_info.get("from_number") or outbound_from_number
    participant_identity = phone_number
    logger.info(f"dialing {phone_number} from {from_number} via trunk {outbound_trunk_id}")

    agent = VoiceAgent()

    # Gemini realtime (speech-to-speech). Native-audio model with affective
    # dialog + proactivity for a natural, casual conversation. Only needs
    # GOOGLE_API_KEY.
    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Aoede",
            temperature=0.9,
            enable_affective_dialog=True,
            proactivity=True,
        ),
    )

    # start the session before dialing so nothing is missed when they pick up
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
    )

    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                sip_number=from_number,
                participant_identity=participant_identity,
                wait_until_answered=True,
            )
        )

        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")
        agent.set_participant(participant)

        # they answered — open with a casual hello
        await session.generate_reply(
            instructions=(
                f"Say a short, warm hello. Introduce yourself as {AGENT_NAME} in "
                "a casual way and ask how their day is going. One or two sentences."
            )
        )

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
