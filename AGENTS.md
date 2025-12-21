# Repository Guidelines

## Build, Test & Development Commands
Python 3.12 + repo venv loop:
example:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
(.venv) pip install -e . open_webui aiohttp cryptography fastapi httpx lz4 pydantic pydantic_core sqlalchemy tenacity pytest pytest-asyncio
PYTHONPATH=. .venv/bin/pytest tests -q
```

Editable install loads the pytest bootstrap plugin; always prefix commands with `PYTHONPATH=.` (pytest, coverage, `python -m build`) so Open WebUI shims resolve.

## Coding Style & Naming Conventions
Stick to the current Python style: 4-space indent, typed signatures, `snake_case` helpers, `PascalCase` models, uppercase constants near the manifest docstring. Keep import grouping (stdlib → third-party → Open WebUI). Comment only when logic is non-obvious. Lint with `ruff` or `flake8`; avoid formatters that would rewrite the manifest header or valve tables.

## Testing Guidelines
Pytest plus `pytest_asyncio` power the suite. Name files `tests/test_<feature>.py` and extend the subsystem you touched (`test_streaming_queues.py`, `test_pipe_guards.py`, etc.). Reuse `tests/conftest.py` fixtures (fake Redis, ULIDs, FastAPI requests). Run the touched file first, then `PYTHONPATH=. .venv/bin/pytest tests -q`; update coverage via `coverage run -m pytest` before touching `coverage_annotate/`.

## Commit & Pull Request Guidelines
History follows lightweight Conventional Commits (`fix:`, `feat:`, `chore:`). Write imperative subjects under ~72 characters and keep each commit scoped. PRs should explain the behavior change, list valves/docs touched, and paste the exact test commands or Open WebUI flows exercised; update the relevant docs in the same PR.

## Backup Discipline
Before editing, overwriting, renaming, or deleting any file, create a fresh snapshot in `backups/` named `backups/<relative-path>-YYYY-MM-DD-HH-MM-SS` (24-hour clock). Always capture a new backup for each edit session so no local work is lost.

## External References
`.external/` lives at the repo root and is read-only reference material. Browse Open-WebUI sources via `.external/open-webui`, consult OpenRouter docs in `.external/openrouter_docs`, inspect the latest `/models` dump at `.external/models-openrouter-*.json`, and reach the live model catalog at `https://openrouter.ai/api/frontend/models` (JSON response).

## Security & Configuration Tips
Never commit real secrets (`OPENROUTER_API_KEY`, `WEBUI_SECRET_KEY`, `ARTIFACT_ENCRYPTION_KEY`). Changes to SSRF guards, artifact persistence, breakers, or new valves must include tests (`test_security_methods.py`, `test_artifact_helpers.py`) plus a short operator note. Capture new valve defaults and migrations in the valve atlas before shipping.
