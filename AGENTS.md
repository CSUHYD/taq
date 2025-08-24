# Repository Guidelines

## Project Structure & Module Organization
- `web_app.py`: Flask + Socket.IO web server and routes.
- `robot_service.py`: Business logic, Pydantic models, experiment logging.
- `vlmCall_ollama.py`: VLM API client and prompt config loader.
- `config/`: English prompt config (`prompt_config_en.json`).
- `templates/`: Frontend HTML (`index.html`).
- `captured_frames/`: Saved snapshots from the video stream.
- `logs/experiments/`: Session logs and images created at runtime.
- Tests: top‑level scripts `test_*.py`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run web app (camera): `python web_app.py --camera-id 0`
- Run web app (ZMQ): `python web_app.py --use-zmq --zmq-server 192.168.123.164 --zmq-port 5555`
- Image client (optional): `python image_client.py`
- Tests (offline-safe): `python test_timer_system.py`
- Tests (may hit network): `python test_architecture.py`

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4‑space indentation.
- Files/modules: `snake_case.py`; classes: `PascalCase`; functions/vars: `lower_snake_case`.
- Keep functions small and focused; add docstrings where behavior isn’t obvious.
- No repo‑wide formatter configured; keep lines ≤ 100 chars and avoid unused imports.

## Testing Guidelines
- Tests are executable scripts under the repo root: `python test_*.py`.
- Prefer adding new tests next to similar ones; name `test_feature_name.py`.
- Some tests call external services (VLM API). Stub or skip in CI if network is unavailable.

## Commit & Pull Request Guidelines
- Commit style: use short, prefixed imperatives seen in history, e.g. `add: …`, `fix: …`, `update: …`.
  - Example: `fix: handle camera open failure`.
- PRs should include: clear description, linked issue, test plan (commands run + results), and screenshots/GIFs for UI changes (`templates/index.html`).
- Ensure new runtime files are generated under `logs/experiments` and not committed.

## Security & Configuration Tips
- API endpoint is set in `vlmCall_ollama.VLMAPI.api_url`. Update for your environment if needed.
- Prompts live in `config/prompt_config_en.json`; keep schema keys consistent.
- Do not commit sensitive data or large captured images; `captured_frames/` and `logs/` are ephemeral.

## Architecture Overview
- Web UI (`web_app.py` + `templates/`) calls business services (`robot_service.py`) which wrap the VLM client (`vlmCall_ollama.py`). Keep this separation when adding features.
