# One image for both processes; the command decides which one runs:
#   worker:  python agent.py start
#   console: python server.py
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent.py server.py call_elder.py elders.json ./
COPY web ./web

# fetch the turn-detector / VAD model files now so pods start without downloading
RUN python agent.py download-files

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
CMD ["python", "agent.py", "start"]
