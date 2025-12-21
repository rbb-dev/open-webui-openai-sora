# OpenAI Sora Video Generation Pipe (Open WebUI)

An Open WebUI **Pipe** that turns natural-language prompts (optionally with a reference image) into OpenAI Sora video jobs, streams progress updates, downloads the result, and uploads the final video into Open WebUI’s file store.

## Features

- Generate videos via `sora-2` and `sora-2-pro`
- Task-model analysis (Structured Outputs / JSON Schema) to infer:
  - intent (`generate` vs `remix`)
  - duration (snaps **up** to **4 / 8 / 12** seconds)
  - resolution bucket
  - whether to reuse a previous reference image
  - optional variations (multiple takes)
- Reference image support:
  - Extracts the latest user image (or reuses a previous turn’s image when requested)
  - Automatically crops/resizes to a valid Sora resolution bucket
- Streaming UX:
  - Status events + progress percentage during job polling
  - HTML video embed + download link on completion
- Safety/robustness:
  - SSRF guard for remote media URLs
  - Sanitises invalid resolution valve entries (won’t send non-Sora sizes)
  - Defensive parsing and clear user-facing errors

## Requirements

- Open WebUI (backend running and able to load “Functions / Pipes”)
- Python 3.12
- An OpenAI-compatible endpoint + API key:
  - Default: `https://api.openai.com/v1`
  - Also works with other OpenAI-compatible providers if they support:
    - `POST /v1/chat/completions` for the task model
    - `POST /v1/videos` (and related retrieve/download endpoints) for Sora

## Installation (Open WebUI)

1. Open **Admin Panel → Functions** (or wherever your Open WebUI build manages Pipes).
2. Create a new function/pipe and paste the contents of `open-webui-openai-sora.py` (or upload it, depending on your UI).
3. Save and enable the pipe.
4. Configure **Valves** (see below).

## Configuration (Valves)

The pipe exposes configuration via a `Valves` model in `open-webui-openai-sora.py`.

Core valves:

- `API_KEY` (required): your provider key.
- `API_BASE_URL`: default `https://api.openai.com/v1`.
- `MODEL`: `sora-2` or `sora-2-pro`.
- `TASK_MODEL_TO_USE`: model used for parameter extraction (Structured Outputs).

Defaults and limits:

- `DEFAULT_DURATION`: `4`, `8`, or `12` (seconds).
- `DEFAULT_RESOLUTION`: one of `720x1280`, `1280x720`, `1024x1792`, `1792x1024`.
- `ALLOWED_RESOLUTIONS`: comma-separated list used to *restrict base (`sora-2`) resolutions*.
  - Invalid values are ignored (e.g. `800x600`).
  - Pro-only sizes are controlled by using `MODEL=sora-2-pro`.
- `ENABLE_VARIATIONS` / `MAX_VARIATIONS`: enable multiple takes in one request.
- `MAX_REMOTE_MEDIA_BYTES`: maximum remote media size (downloaded or decoded).

Operational:

- `REQUEST_TIMEOUT`, `POLL_INTERVAL_SECONDS`, `MAX_POLL_TIME_SECONDS`
- `STATUS_POLL_MAX_ERRORS`
- `IMMEDIATE_DELETE_JOBS`
- `ENABLE_LOGGING`: when enabled, logs include a preview of the task model’s raw response.

## How it behaves

### Duration snapping (important)

Sora accepts only **4**, **8**, or **12** seconds. If the user asks for anything else, the pipe **rounds up** to the next supported value (capped at 12) and uses that value both in the API request and in status messages.

Examples:

- “1 second” → 4s
- “7 seconds” → 8s
- “20 seconds” → 12s

### Resolutions

Supported buckets:

- Base (`sora-2`): `720x1280`, `1280x720`
- Pro-only (`sora-2-pro`): `1024x1792`, `1792x1024`

If a user asks for an unsupported size/aspect ratio, the pipe picks the closest supported bucket (respecting orientation hints like “vertical”, “portrait”, “16:9”, etc.).

### Reference images

If the user uploads an image with their prompt, the pipe:

1. Extracts the image from the Open WebUI message
2. Detects its dimensions
3. Chooses a valid Sora size bucket
4. Crops/resizes the image to match that bucket
5. Sends it as the Sora reference input for generation

If no image is attached in the current message, the task model can request reusing a previous-turn image (only when the user explicitly asks to reuse it).

### Remixing

The pipe stores hidden metadata for completed jobs (video id, size, seconds, etc.) and can use it to locate the most recent video for remix flows.

## Troubleshooting

- **Task model “Model not found”**
  - Your `TASK_MODEL_TO_USE` doesn’t exist for `API_BASE_URL` or your key lacks access.
  - Confirm you can call `POST /v1/chat/completions` with that model id on the same provider.

- **Task model “finish_reason=length”**
  - The model didn’t finish emitting the structured JSON in time.
  - This repo sets a generous `max_completion_tokens`; if you still see truncation, increase it in the task-model call.

- **Video API errors**
  - Double-check `MODEL` is set to `sora-2` or `sora-2-pro` and that your org/project has access.

## Development

See `AGENTS.md` for the repo workflow. Typical local loop:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install open_webui aiohttp cryptography fastapi httpx lz4 pydantic pydantic_core sqlalchemy tenacity pytest pytest-asyncio pillow
PYTHONPATH=. .venv/bin/python -m pytest -q
```

## License

MIT. See `LICENSE`.
