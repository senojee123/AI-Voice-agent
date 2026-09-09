<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Python Outbound Call Agent

<p>
  <a href="https://docs.livekit.io/agents/overview/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>

An AI **voice assistant** that places an outbound phone call and answers the
questions and requests of whoever picks up. It uses LiveKit SIP and the Python
[Agents Framework](https://github.com/livekit/agents).

The voice is Google's **Gemini realtime** native-audio model
(`livekit-plugins-google`), speech-to-speech. It only needs a `GOOGLE_API_KEY`.
Calls go out over the `ElderlyCare` SIP trunk (`ST_CTjL7C7PrnZk`) to
`+94740525967` by default.

This builds on the [Outbound Calls](https://docs.livekit.io/agents/start/telephony/#outbound-calls)
docs. A SIP outbound trunk must already be configured.

## Features

- Places outbound calls and opens by asking how it can help
- Answers the caller's questions directly (Gemini native audio, `proactivity`
  off so it always responds)
- Logs both sides of the call as `[user]` / `[assistant]` lines
- Follows the caller's language (English / Sinhala / Tamil)
- Hangs up when the caller says goodbye (`end_call`) or on voicemail
  (`detected_answering_machine`)
- Persona/name configurable via the `AGENT_NAME` env var; behaviour in the
  `INSTRUCTIONS` string in `agent.py`

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```shell
git clone https://github.com/livekit-examples/outbound-caller-python.git
cd outbound-caller-python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python agent.py download-files
```

Set up the environment by copying `.env.example` to `.env.local` and filling in the required values:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `GOOGLE_API_KEY` - Gemini API key from https://aistudio.google.com/apikey (paste it exactly - a stray trailing `.` causes a 401)
- `SIP_OUTBOUND_TRUNK_ID`
- `OUTBOUND_PHONE_NUMBER` - default number to call. The ElderlyCare trunk rejects
  `+E.164` with a 403, so use local format (`0740525967`); `+94...` is auto-converted.
- `OUTBOUND_FROM_NUMBER` - caller-ID number on the trunk (`0117286109`)
- `AGENT_NAME` - optional, what the agent calls itself on the call (default `Nova`)

To change the agent's personality, edit `INSTRUCTIONS` near the top of `agent.py`.
To change the voice, edit `voice=` in `agent.py` (Gemini voices: `Aoede`, `Puck`,
`Charon`, `Kore`, `Fenrir`, `Leda`, `Orus`, `Zephyr`, ...).

### Placing a call from your machine (Windows / PowerShell)

**1. Start the worker** and leave it running (one window):

```powershell
.\start-agent.ps1        # or:  python agent.py dev
```

It prints `registered worker` once connected. Keep this window open — it must
stay running to handle calls.

**2. Place a call** from a second window:

```powershell
.\call.ps1                      # dials OUTBOUND_PHONE_NUMBER (+94740525967)
.\call.ps1 0740525967           # dials a specific number
.\call.ps1 +94740525967         # +E.164 is auto-converted to local format
.\call.ps1 0740525967 -TransferTo 0112345678
```

`call.ps1` reads the LiveKit credentials from `.env.local` and dispatches the
job; the worker window then logs `dialing ... via trunk ...` and the target
phone rings. The agent greets first when you answer.

### Making a call with the raw CLI

`call.ps1` just wraps this:

```shell
lk dispatch create --new-room --agent-name outbound-caller
# or target a number / set a transfer destination:
lk dispatch create --new-room --agent-name outbound-caller \
  --metadata '{"phone_number": "0740525967", "transfer_to": "0112345678"}'
```

(the `lk` CLI needs `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in the environment)
