import asyncio
import base64
import importlib.util
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

OPEN_WEBUI_BACKEND = Path("/home/boris/src/open-webui/backend")
if OPEN_WEBUI_BACKEND.exists():
    sys.path.insert(0, str(OPEN_WEBUI_BACKEND))

MODULE_PATH = Path(__file__).resolve().parents[1] / "open-webui-openai-sora.py"
SPEC = importlib.util.spec_from_file_location("sora_pipe", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("sora_pipe", MODULE)
SPEC.loader.exec_module(MODULE)

Pipe = MODULE.Pipe
ResolutionPolicy = MODULE.ResolutionPolicy


@pytest.fixture()
def pipe():
    return Pipe()


def test_allowed_resolutions_sanitizes_to_canonical_base_set():
    sanitized = Pipe._sanitize_allowed_resolutions("800x600, 720x1280, 1024x1792, nonsense")
    assert sanitized == {"720x1280"}
    assert sanitized.issubset(Pipe.BASE_RESOLUTIONS)


def test_resolution_policy_respects_orientation_hint():
    policy = ResolutionPolicy(
        default_resolution="720x1280",
        allowed_resolutions={"720x1280", "1280x720"},
        base_resolutions={"720x1280", "1280x720"},
        pro_resolutions={"1024x1792", "1792x1024"},
    )
    decision = policy.choose_size("1900x3000", prompt="vertical reels vibe", video_model="sora-2")
    assert decision.resolved == "720x1280"
    assert decision.substituted is True


def test_select_latest_artifact_prefers_newest(pipe):
    artifacts = [
        {"video_id": "video-old", "created_at": 100},
        {"video_id": "video-new", "created_at": 200},
    ]
    result = pipe._select_latest_artifact(artifacts)
    assert result["video_id"] == "video-new"

def test_task_model_response_format_is_json_schema(pipe):
    response_format = pipe._task_model_response_format(max_variations=pipe.valves.MAX_VARIATIONS)
    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert "intent" in schema["properties"]
    assert "seconds" in schema["properties"]
    assert "size" in schema["properties"]
    assert "video_model" in schema["properties"]
    assert "reuse_previous_image" in schema["properties"]


@pytest.mark.asyncio()
async def test_inline_media_respects_byte_limit(pipe):
    pipe.valves.MAX_REMOTE_MEDIA_BYTES = 8
    big_payload = base64.b64encode(b"A" * 32).decode("utf-8")
    small_payload = base64.b64encode(b"abcd").decode("utf-8")

    big_message = {
        "role": "user",
        "content": f"![img](data:image/png;base64,{big_payload})",
    }
    small_message = {
        "role": "user",
        "content": f"![img](data:image/png;base64,{small_payload})",
    }

    assert await pipe._extract_media_from_message(big_message) == []
    media = await pipe._extract_media_from_message(small_message)
    assert len(media) == 1
    assert media[0]["mimeType"] == "image/png"


@pytest.mark.asyncio()
async def test_file_media_respects_byte_limit(tmp_path):
    pipe = Pipe()
    pipe.valves.MAX_REMOTE_MEDIA_BYTES = 32

    small_file = tmp_path / "small.bin"
    small_file.write_bytes(b"x" * 16)
    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"y" * 128)

    class DummyFile:
        def __init__(self, path):
            self.path = path
            self.meta = {"content_type": "image/png"}

    async def get_small(_: str):
        return DummyFile(str(small_file))

    async def get_large(_: str):
        return DummyFile(str(large_file))

    markdown = "![ref](/files/abc123)"
    message = {"role": "user", "content": markdown}

    pipe._get_file_by_id = get_small  # type: ignore[assignment]
    media = await pipe._extract_media_from_message(message)
    assert len(media) == 1

    pipe._get_file_by_id = get_large  # type: ignore[assignment]
    assert await pipe._extract_media_from_message(message) == []


def test_resize_image_to_matches_target(pipe):
    img = Image.new("RGB", (1920, 1080), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    data = buffer.getvalue()

    resized = pipe._resize_image_to(data, "720x1280")
    resized_img = Image.open(io.BytesIO(resized))
    assert resized_img.size == (720, 1280)
