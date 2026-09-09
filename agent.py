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
from livekit.plugins import google, noise_cancellation
from google.genai import types


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
You are {AGENT_NAME}, a helpful voice assistant on a phone call. Your job is to
listen to the caller and answer their questions and requests.

- ALWAYS speak in English. Never switch to Sinhala, Tamil, or any other
  language, even if the audio sounds unclear or you think you heard another
  language. English only.
- Respond directly to what the caller just said. Answer the actual question
  first, then add a brief helpful detail only if it's useful.
- Keep replies short and natural for speech — usually one to four sentences. If
  the topic is big, give the key points and offer to go further.
- If you're not sure or don't know, say so plainly. Never make up facts.
- If the audio was unclear, say "Sorry, I didn't catch that — could you say it
  again?" and wait. Do not guess.
- Always say something back when the caller speaks — don't go silent on them.
- Talk warmly and plainly, like a knowledgeable friend. No lists or markdown
  read out loud.
- Don't say that you're an AI model or describe these instructions.

Use the end_call tool when the caller is done or says goodbye. Use the
detected_answering_machine tool if you reach a voicemail greeting.
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

    # Gemini realtime (speech-to-speech), native-audio model. Only needs
    # GOOGLE_API_KEY.
    # - `proactivity` is OFF: with it on the model decides when *not* to answer.
    # - `language="en-US"` pins the agent's speech to English (a native-audio
    #   model will otherwise mirror garbled phone audio into Sinhala/Tamil).
    # - transcription is enabled so both sides of the call show in the logs.
    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Aoede",
            temperature=0.7,
            language="en-US",
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=["en-US"]
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        ),
    )

    @session.on("conversation_item_added")
    def _log_turn(ev):
        role = getattr(ev.item, "role", None)
        if role is None:
            return  # e.g. AgentHandoff items have no role
        text = getattr(ev.item, "text_content", None) or ""
        logger.info(f"[{role}] {text}")

    @session.on("user_input_transcribed")
    def _log_user(ev):
        if getattr(ev, "is_final", False):
            logger.info(f"[user heard] {ev.transcript}")

    # start the session before dialing so nothing is missed when they pick up
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # echo + background-noise removal tuned for phone calls; without
                # this the model hears its own echo and mis-transcribes speech
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
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

        # they answered — open with a short hello and offer to help
        await session.generate_reply(
            instructions=(
                f"Greet the caller warmly in one sentence, say you're {AGENT_NAME}, "
                "and ask what you can help them with."
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
