"""
title: OpenAI Sora Video Generation Pipe
description: OpenAI Sora video job integration with task model analysis and progress streaming
id: sora-video
author: rbb-dev
author_url: https://github.com/rbb-dev/
version: 0.10.0
requirements: pillow
features:
  - Video generation via OpenAI Sora with async job polling and cleanup.
  - Task model analysis to infer intent, duration, and resolution from natural language.
  - Supports reference image extraction from Open WebUI messages (current or previous turn) for guided generation.
  - Streams OpenAI-compatible responses and emits detailed progress updates.
  - Uploads generated videos to the WebUI file store and returns Markdown download links.
  - Configurable via valves (API key, base URL, model, defaults, polling, logging).
  - Defensive error handling, logging controls, and Open WebUI-native status emission.
"""
import asyncio
import base64
import contextlib
import html
import io
import json
import logging
import os
import re
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
import ipaddress
import urllib.parse
from typing import List, Dict, Any, Optional, Callable, Awaitable, Iterable, Literal, Tuple, Set, cast
import httpx
from PIL import Image
from fastapi import Request, UploadFile, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI, OpenAIError
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema
from open_webui.routers.files import upload_file
from open_webui.models.users import UserModel, Users
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse, HTMLResponse, PlainTextResponse, Response
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass
class ResolutionDecision:
    """Represents the resolved video size along with context about the selection."""
    requested: Optional[str]
    resolved: str
    supported: Set[str]
    allowed_text: str
    requested_display: Optional[str] = None
    substituted: bool = False
    message: Optional[str] = None
    orientation_hint: Optional[str] = None
    requires_model_upgrade: bool = False


@dataclass
class ReferenceAssessment:
    """Details whether a reference image can be reused as-is, resized, or rejected."""
    blocked: bool
    should_resize: bool
    message: Optional[str]
    image_size: Optional[str]
    target_size: str


class ResolutionPolicy:
    """Encapsulate all resolution parsing, selection, and reference-image checks."""

    # Scoring weights for resolution similarity matching
    RATIO_WEIGHT = 5.0
    ORIENTATION_MISMATCH_PENALTY = 0.5
    HINT_MISMATCH_PENALTY = 0.25

    def __init__(
        self,
        *,
        default_resolution: str,
        allowed_resolutions: Set[str],
        base_resolutions: Set[str],
        pro_resolutions: Set[str],
    ):
        """Initialize resolution limits and derive the default bucket."""
        self.base_resolutions = self._normalize_set(base_resolutions)
        self.pro_resolutions = self._normalize_set(pro_resolutions)
        configured = self._normalize_set(allowed_resolutions)
        if not configured:
            configured = self.base_resolutions or {"720x1280", "1280x720"}
        self.standard_resolutions = configured
        self.default_resolution = (
            self.normalize(default_resolution)
            or next(iter(sorted(self.standard_resolutions)), "720x1280")
        )
        if self.default_resolution not in self.standard_resolutions:
            # Ensure the default is always part of the standard set for downstream logic.
            self.standard_resolutions = set(self.standard_resolutions)
            self.standard_resolutions.add(self.default_resolution)

    @staticmethod
    def _normalize_set(values: Set[str]) -> Set[str]:
        """Return a sanitized set of WIDTHxHEIGHT strings."""
        normalized: Set[str] = set()
        for value in values or set():
            norm = ResolutionPolicy.normalize(value)
            if norm:
                normalized.add(norm)
        return normalized

    @staticmethod
    def normalize(value: Any) -> Optional[str]:
        """Return a 'WIDTHxHEIGHT' string or None if the input cannot be parsed."""
        if value is None:
            return None
        text = str(value).strip().lower().replace(" ", "")
        if not text or "x" not in text:
            return None
        width_str, height_str = text.split("x", 1)
        try:
            width = int(width_str)
            height = int(height_str)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return f"{width}x{height}"

    @staticmethod
    def _parse_resolution_dims(value: str) -> Optional[Tuple[int, int]]:
        """Convert a WIDTHxHEIGHT string into integer dimensions."""
        normalized = ResolutionPolicy.normalize(value)
        if not normalized:
            return None
        width_str, height_str = normalized.split("x", 1)
        try:
            return int(width_str), int(height_str)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _orientation_label(value: str) -> str:
        """Describe the orientation (horizontal/vertical/square) of a resolution."""
        dims = ResolutionPolicy._parse_resolution_dims(value)
        if not dims:
            return "unknown"
        width, height = dims
        if width == height:
            return "square"
        return "horizontal" if width > height else "vertical"

    @staticmethod
    def detect_orientation_hint(prompt: str) -> Optional[str]:
        """Infer an orientation hint from conversational prompt text."""
        prompt_lower = (prompt or "").lower()
        if not prompt_lower:
            return None
        vertical_terms = ["vertical", "portrait", "9:16", "tiktok", "reels", "story", "phone"]
        horizontal_terms = ["16:9", "horizontal", "landscape", "youtube", "widescreen"]
        square_terms = ["square", "1:1"]
        if any(term in prompt_lower for term in vertical_terms):
            return "vertical"
        if any(term in prompt_lower for term in horizontal_terms):
            return "horizontal"
        if any(term in prompt_lower for term in square_terms):
            return "square"
        return None

    def _pick_by_orientation(self, candidates: Set[str], orientation: Optional[str]) -> Optional[str]:
        """Prefer a candidate that matches the requested orientation."""
        if not candidates:
            return None
        orientation = orientation or ""
        preference_map = {
            "vertical": ["720x1280", "1024x1792"],
            "horizontal": ["1280x720", "1792x1024"],
        }
        preferred = preference_map.get(orientation, [])
        for candidate in preferred:
            normalized = self.normalize(candidate)
            if normalized and normalized in candidates:
                return normalized
        filtered = [
            candidate
            for candidate in candidates
            if self._orientation_label(candidate) == orientation
        ]
        if not filtered and orientation == "square":
            filtered = sorted(
                candidates,
                key=lambda value: abs(
                    (dims := self._parse_resolution_dims(value) or (0, 0))[0] - dims[1]
                ),
            )
        if filtered:
            return sorted(
                filtered,
                key=lambda value: (dims := self._parse_resolution_dims(value) or (0, 0))[0] * dims[1],
            )[-1]
        return None

    def _pick_closest_supported(
        self,
        target_norm: Optional[str],
        candidates: Set[str],
        orientation_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Return the supported resolution whose ratio/area most closely matches target_norm."""
        if not target_norm or not candidates:
            return None
        target_dims = self._parse_resolution_dims(target_norm)
        if not target_dims:
            return None
        target_ratio = target_dims[0] / target_dims[1]
        target_area = target_dims[0] * target_dims[1]
        target_orientation = self._orientation_label(target_norm)
        best_choice: Optional[str] = None
        best_score: Optional[float] = None
        for candidate in candidates:
            dims = self._parse_resolution_dims(candidate)
            if not dims:
                continue
            ratio = dims[0] / dims[1]
            area = dims[0] * dims[1]
            ratio_diff = abs(ratio - target_ratio)
            area_diff = abs(area - target_area) / max(target_area, 1)
            score = ratio_diff * self.RATIO_WEIGHT + area_diff
            cand_orientation = self._orientation_label(candidate)
            if cand_orientation != target_orientation:
                score += self.ORIENTATION_MISMATCH_PENALTY
            if orientation_hint and cand_orientation != orientation_hint:
                score += self.HINT_MISMATCH_PENALTY
            if best_score is None or score < best_score:
                best_score = score
                best_choice = candidate
        return best_choice

    def get_supported(self, video_model: str) -> Set[str]:
        """Return the set of resolutions enabled for the requested model tier."""
        supported = set(self.standard_resolutions)
        if not supported:
            supported = set(self.base_resolutions)
        if video_model == "sora-2-pro":
            supported |= set(self.pro_resolutions)
        if not supported:
            supported = {"720x1280", "1280x720"}
        return supported

    def choose_size(self, requested: Any, *, prompt: str, video_model: str) -> ResolutionDecision:
        """Resolve the final size to use, optionally substituting the closest supported option."""
        requested_display = str(requested).strip() if requested is not None else ""
        requested_norm = self.normalize(requested)
        orientation_hint = self.detect_orientation_hint(prompt)
        supported = self.get_supported(video_model)
        requires_pro = bool(
            requested_norm and requested_norm in self.pro_resolutions and video_model != "sora-2-pro"
        )
        if requires_pro:
            supported = self.get_supported("sora-2-pro")
        resolved = requested_norm if requested_norm in supported else None
        substituted = False
        message = None
        if not resolved:
            if requested_norm:
                resolved = self._pick_closest_supported(requested_norm, supported, orientation_hint)
                substituted = bool(resolved)
                requested_label = requested_display or requested_norm
                if resolved:
                    message = (
                        f"Requested video size {requested_label} isn't supported; using {resolved} instead."
                    )
            if not resolved:
                resolved = self._pick_by_orientation(supported, orientation_hint)
            if not resolved:
                if self.default_resolution in supported:
                    resolved = self.default_resolution
                else:
                    resolved = next(iter(sorted(supported)), self.default_resolution)
        allowed_text = ", ".join(sorted(supported))
        full_message = f"{message} Supported sizes: {allowed_text}." if message else None
        return ResolutionDecision(
            requested=requested_norm,
            resolved=resolved,
            supported=set(supported),
            allowed_text=allowed_text,
            requested_display=requested_display or requested_norm,
            substituted=substituted,
            message=full_message,
            orientation_hint=orientation_hint,
            requires_model_upgrade=requires_pro,
        )

    def evaluate_reference(
        self,
        *,
        image_size: Optional[str],
        target_size: str,
        supported: Set[str],
        allowed_text: str,
    ) -> ReferenceAssessment:
        """Explain whether a reference image must be resized or rejected for the target size."""
        target_norm = self.normalize(target_size) or target_size
        if not image_size:
            return ReferenceAssessment(False, False, None, None, target_norm)
        image_norm = self.normalize(image_size)
        if not image_norm:
            message = (
                "Unable to determine your reference image size. Please re-upload it so we can match the required resolution."
            )
            message = f"{message} Supported sizes: {allowed_text}."
            return ReferenceAssessment(True, False, message, None, target_norm)
        if image_norm == target_norm and image_norm in supported:
            return ReferenceAssessment(False, False, None, image_norm, target_norm)
        resize_message = (
            f"Resized your reference image from {image_norm} to {target_norm} to match Sora's requirements."
        )
        return ReferenceAssessment(
            False,
            True,
            resize_message,
            image_norm,
            target_norm,
        )

class Pipe:
    """Implements the Open WebUI pipe that orchestrates OpenAI Sora video jobs."""
    PIPE_ID = "sora-video"
    PIPE_FULL_ID = f"{__name__}.{PIPE_ID}"
    ZW_ARTIFACT_PREFIX = "\u2063\u2060"
    ZW_ARTIFACT_SUFFIX = "\u2063\u2061"
    ZW_BIT_ZERO = "\u200B"
    ZW_BIT_ONE = "\u200C"
    DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024
    BASE_RESOLUTIONS = {"720x1280", "1280x720"}
    PRO_ONLY_RESOLUTIONS = {"1024x1792", "1792x1024"}

    @classmethod
    def _sanitize_allowed_resolutions(cls, raw: str) -> Set[str]:
        """
        Sanitize the ALLOWED_RESOLUTIONS valve value to the canonical Sora resolution buckets.

        Only Sora's fixed sizes are valid. We allow operators to restrict the base (sora-2) set,
        but we never treat pro-only buckets as valid for sora-2.
        """
        canonical = set(cls.BASE_RESOLUTIONS) | set(cls.PRO_ONLY_RESOLUTIONS)
        enabled_base: Set[str] = set()
        invalid_tokens: List[str] = []
        for token in (raw or "").split(","):
            candidate = token.strip()
            if not candidate:
                continue
            normalized = ResolutionPolicy.normalize(candidate)
            if not normalized or normalized not in canonical:
                invalid_tokens.append(candidate)
                continue
            if normalized in cls.BASE_RESOLUTIONS:
                enabled_base.add(normalized)
        if invalid_tokens:
            logger.error(
                "Ignoring invalid ALLOWED_RESOLUTIONS entries: %s. Allowed values: %s",
                ", ".join(invalid_tokens),
                ", ".join(sorted(canonical)),
            )
        return enabled_base or set(cls.BASE_RESOLUTIONS)
    class Valves(BaseModel):
        """Configuration surface exposed to Open WebUI for the Sora pipe."""
        # Auth and endpoint
        API_KEY: str = Field(default="", description="OpenAI API key (Bearer)")
        API_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
        # Logging
        ENABLE_LOGGING: bool = Field(default=False, description="Enable info/debug logs for this plugin. When False, only errors are logged.")
        # Model and basic generation options
        MODEL: str = Field(
            default="sora-2",
            description="OpenAI Sora video model id (choices: sora-2, sora-2-pro)",
            json_schema_extra={"choices": ["sora-2", "sora-2-pro"]},
        )
        # Task model settings
        TASK_MODEL_TO_USE: str = Field(
            default="gpt-5-mini",
            description="Model id used for the internal parameter-extraction task (called directly via OpenAI-compatible API).",
        )
        # Video defaults
        DEFAULT_DURATION: Literal[4, 8, 12] = Field(
            default=4,
            description="Default video duration in seconds (Sora buckets to 4, 8, 12).",
        )
        DEFAULT_RESOLUTION: Literal["720x1280", "1280x720", "1024x1792", "1792x1024"] = Field(
            default="720x1280",
            description="Default video resolution as 'WIDTHxHEIGHT' (Sora-supported).",
        )
        ALLOWED_RESOLUTIONS: str = Field(
            default="720x1280,1280x720",
            description="Comma-separated Sora-valid resolutions. Use the longer list when enabling sora-2-pro.",
            json_schema_extra={
                "examples": [
                    "720x1280,1280x720",
                    "720x1280,1280x720,1024x1792,1792x1024",
                ]
            },
        )
        ENABLE_VARIATIONS: bool = Field(
            default=False,
            description="If true, allow generating multiple variations per job.",
        )
        MAX_VARIATIONS: int = Field(
            default=4,
            description="Maximum number of variations (versions) per job when variations are enabled.",
        )
        MAX_REMOTE_MEDIA_BYTES: int = Field(
            default=50 * 1024 * 1024,
            description="Maximum size for remote media downloads in bytes.",
        )
        # HTTP client / polling
        REQUEST_TIMEOUT: int = Field(default=600, description="Request timeout in seconds")
        POLL_INTERVAL_SECONDS: int = Field(default=3, description="Polling interval in seconds for video job status")
        MAX_POLL_TIME_SECONDS: int = Field(default=600, description="Maximum total wait time for a single video job in seconds")
        STATUS_POLL_MAX_ERRORS: int = Field(
            default=5,
            ge=1,
            le=10,
            description="Maximum consecutive OpenAI errors allowed while polling a video job before aborting.",
        )
        IMMEDIATE_DELETE_JOBS: bool = Field(
            default=False,
            description="If true, delete Sora video jobs immediately after download. When false, jobs remain on OpenAI for potential reuse.",
        )
    def __init__(self):
        """Instantiate the pipe with sanitized valve values and computed policies."""
        self.valves = self.Valves()
        # Parse allowed resolutions once
        raw = getattr(self.valves, "ALLOWED_RESOLUTIONS", "") or ""
        self.allowed_resolutions = set(self._sanitize_allowed_resolutions(raw))
        self.resolution_policy = ResolutionPolicy(
            default_resolution=self.valves.DEFAULT_RESOLUTION,
            allowed_resolutions=set(self.allowed_resolutions),
            base_resolutions=self.BASE_RESOLUTIONS,
            pro_resolutions=self.PRO_ONLY_RESOLUTIONS,
        )
        self.allowed_resolutions = set(self.resolution_policy.standard_resolutions)
        self._apply_logging_valve()
    def _apply_logging_valve(self) -> None:
        """Set logger level based on ENABLE_LOGGING valve for this plugin."""
        enabled = bool(getattr(self.valves, "ENABLE_LOGGING", False))
        logger.setLevel(logging.INFO if enabled else logging.ERROR)
        logger.propagate = True

    def _get_supported_resolutions(self, video_model: str) -> Set[str]:
        """
        Return the resolution set that should be considered valid for the
        requested Sora video model. sora-2-pro unlocks the taller/wider sizes.
        """
        return self.resolution_policy.get_supported(video_model)
    def _build_openai_client(self) -> OpenAI:
        """Construct an OpenAI SDK client that respects the configured base URL."""
        base_url = (getattr(self.valves, "API_BASE_URL", "") or "").strip()
        if base_url:
            return OpenAI(api_key=self.valves.API_KEY, base_url=base_url)
        return OpenAI(api_key=self.valves.API_KEY)
    @contextlib.asynccontextmanager
    async def _openai_client(self):
        """Yield an OpenAI client and ensure it is closed on exit."""
        client = self._build_openai_client()
        try:
            yield client
        finally:
            try:
                await asyncio.to_thread(client.close)
            except Exception as exc:
                logger.debug(f"Error closing OpenAI client: {exc}")

    async def emit_status(
        self,
        message: str,
        done: bool = False,
        show_in_chat: bool = False,
        emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        """Emit status updates to the client."""
        logger.info(
            "emit_status called: message=%s done=%s show_in_chat=%s emitter_present=%s",
            message,
            done,
            show_in_chat,
            bool(emitter),
        )
        if emitter:
            await emitter({"type": "status", "data": {"description": message, "done": done}})
        if show_in_chat:
            return f"**✅ {message}**\n\n" if done else f"**⏳ {message}**\n\n"
        return ""
    @staticmethod
    def _normalise_model_content(value: Any) -> str:
        """Flatten chat completion content payloads into a single string."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(str(item.get("content", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if isinstance(value, dict):
            for key in ("text", "content"):
                if key in value and value[key]:
                    return str(value[key])
        return str(value) if value is not None else ""
    def _encode_hidden_artifact(self, payload: Dict[str, Any]) -> str:
        """Wrap metadata in zero-width characters so it can ride back in chat output."""
        try:
            raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
            return self._encode_zero_width_artifact(encoded)
        except Exception as exc:
            logger.error("Failed to encode hidden artifact: %s", exc)
            return ""
    def _decode_hidden_artifacts(self, text: str) -> List[Dict[str, Any]]:
        """Extract any zero-width artifacts from assistant text."""
        if not text:
            return []
        artifacts: List[Dict[str, Any]] = []
        artifacts.extend(self._decode_zero_width_artifacts(text))
        return artifacts
    def _append_artifact_payload(self, artifacts: List[Dict[str, Any]], payload: Any) -> None:
        """Add a decoded payload to the running list if it is dict-like."""
        if isinstance(payload, dict):
            artifacts.append(payload)
        elif isinstance(payload, list):
            artifacts.extend(item for item in payload if isinstance(item, dict))
    def _encode_zero_width_artifact(self, encoded: str) -> str:
        """Convert a base64 blob to zero-width characters for invisibility."""
        bits = "".join(f"{ord(ch):08b}" for ch in encoded)
        body = "".join(self.ZW_BIT_ONE if bit == "1" else self.ZW_BIT_ZERO for bit in bits)
        return f"{self.ZW_ARTIFACT_PREFIX}{body}{self.ZW_ARTIFACT_SUFFIX}"
    def _decode_zero_width_artifacts(self, text: str) -> List[Dict[str, Any]]:
        """Decode any zero-width markers embedded in the supplied text."""
        artifacts: List[Dict[str, Any]] = []
        prefix = self.ZW_ARTIFACT_PREFIX
        suffix = self.ZW_ARTIFACT_SUFFIX
        start = 0
        while True:
            start_idx = text.find(prefix, start)
            if start_idx == -1:
                break
            end_idx = text.find(suffix, start_idx + len(prefix))
            if end_idx == -1:
                break
            body = text[start_idx + len(prefix) : end_idx]
            payload = self._decode_zero_width_payload(body)
            if payload:
                self._append_artifact_payload(artifacts, payload)
            start = end_idx + len(suffix)
        return artifacts
    def _decode_zero_width_payload(self, body: str) -> Optional[Any]:
        """Translate a zero-width body back into its JSON payload."""
        if not body:
            return None
        bits: List[str] = []
        for char in body:
            if char == self.ZW_BIT_ZERO:
                bits.append("0")
            elif char == self.ZW_BIT_ONE:
                bits.append("1")
            else:
                # Unknown character within zero-width payload; abort decode.
                return None
        bit_string = "".join(bits)
        if len(bit_string) % 8 != 0:
            return None
        try:
            encoded = "".join(
                chr(int(bit_string[i : i + 8], 2)) for i in range(0, len(bit_string), 8)
            )
            decoded = base64.urlsafe_b64decode(encoded.encode("ascii"))
            return json.loads(decoded.decode("utf-8"))
        except Exception as exc:
            logger.error("Failed to decode zero-width hidden artifact: %s", exc)
            return None
    def _select_latest_artifact(self, artifacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the newest artifact that still carries a video_id reference."""
        for artifact in reversed(artifacts or []):
            video_id = artifact.get("video_id") if isinstance(artifact, dict) else None
            if isinstance(video_id, str) and video_id:
                return artifact
        return None
    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """Return the first JSON object substring embedded in text, if any."""
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            start = text.find("{", start + 1)
        return None
    async def _extract_prompt_and_media(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[
        str,
        List[Dict[str, str]],
        List[Dict[str, Any]],
        List[Dict[str, str]],
        str,
    ]:
        """
        Extract the latest user prompt, any media uploaded with that prompt,
        any hidden artifacts, plus the previous user turn that supplied media.
        """

        def _text_segments(source: Any) -> List[str]:
            """Return plain-text segments from OpenAI-style content blocks."""
            if isinstance(source, str):
                return [source]
            if isinstance(source, list):
                segments: List[str] = []
                for item in source:
                    if isinstance(item, dict) and item.get("type") == "text":
                        segments.append(item.get("text", ""))
                return segments
            return []

        def _collapse_text(content: Any) -> str:
            """Normalize different message content shapes into a single prompt string."""
            segments = _text_segments(content)
            if segments:
                return " ".join(segment.strip() for segment in segments if segment).strip()
            if isinstance(content, str):
                return content.strip()
            return ""

        prompt = ""
        current_media: List[Dict[str, str]] = []
        previous_media: List[Dict[str, str]] = []
        previous_media_prompt = ""
        artifacts: List[Dict[str, Any]] = []

        # Collect hidden artifacts from assistant messages throughout the thread.
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            for segment in _text_segments(content):
                artifacts.extend(self._decode_hidden_artifacts(segment))

        latest_user_idx: Optional[int] = None
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            role = message.get("role")
            if role != "user":
                continue
            content = message.get("content", "")
            if latest_user_idx is None:
                prompt = _collapse_text(content)
                latest_user_idx = idx
            media_entries = await self._extract_media_from_message(message)
            if not media_entries:
                continue
            if idx == latest_user_idx and not current_media:
                current_media = media_entries
            elif idx < latest_user_idx and not previous_media:
                previous_media = media_entries
                previous_media_prompt = _collapse_text(content)
            if current_media and previous_media:
                break

        prompt = (prompt or "").strip()
        if current_media and isinstance(prompt, str):
            prompt = re.sub(r"!\[[^\]]*\]\(data:[^;]+;base64,[^)]+\)", "", prompt).strip()
            prompt = re.sub(r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)", "", prompt).strip()

        logger.info(
            "Extracted prompt: '%s...', current_media=%s, previous_media=%s",
            (prompt[:100] if prompt else ""),
            len(current_media),
            len(previous_media),
        )
        return prompt, current_media, artifacts, previous_media, previous_media_prompt

    async def _extract_media_from_message(self, message: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return media entries (images/videos) embedded in a single user message."""
        media: List[Dict[str, str]] = []
        if message.get("role") != "user":
            return media
        content = message.get("content", "")
        max_media_bytes = getattr(self.valves, "MAX_REMOTE_MEDIA_BYTES", 50 * 1024 * 1024)

        def _estimate_base64_size(data: str) -> int:
            """Estimate decoded byte size from a base64 data URI payload."""
            payload = (data or "").strip()
            if not payload:
                return 0
            padding = payload[-2:].count("=")
            return max(0, (len(payload) * 3) // 4 - padding)

        def _append_base64_media(mime_type: str, data: str):
            """Append inline base64 media when it respects the byte ceiling."""
            estimated_size = _estimate_base64_size(data)
            if max_media_bytes and estimated_size > max_media_bytes:
                size_mb = max_media_bytes / (1024 * 1024)
                logger.error(
                    "Inline media (%s) exceeds %.0fMB limit; skipping.",
                    mime_type or "unknown",
                    size_mb,
                )
                return
            media.append({"mimeType": (mime_type or "application/octet-stream").lower(), "data": data})

        async def _append_media_from_url(url: str):
            """Resolve inline, file, or remote URLs into normalized media entries."""
            if not url:
                return
            url = url.strip()
            if not url:
                return
            if url.startswith("data:"):
                parts = url.split(";base64,", 1)
                if len(parts) == 2:
                    mime_type = parts[0].replace("data:", "", 1).lower()
                    data = parts[1]
                    _append_base64_media(mime_type, data)
                return
            if "/api/v1/files/" in url or "/files/" in url:
                file_id = (
                    url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
                    if "/api/v1/files/" in url
                    else url.split("/files/")[-1].split("/")[0].split("?")[0]
                )
                file_item = await self._get_file_by_id(file_id)
                if file_item and file_item.path:
                    try:
                        file_size = os.path.getsize(file_item.path)
                    except Exception as exc:
                        logger.error(f"Failed to stat file {file_id}: {exc}")
                        return
                    if max_media_bytes and file_size > max_media_bytes:
                        size_mb = max_media_bytes / (1024 * 1024)
                        logger.error(
                            "File %s exceeds %.0fMB limit; skipping.",
                            file_id,
                            size_mb,
                        )
                        return
                    try:
                        with open(file_item.path, "rb") as f:
                            file_data = f.read()
                    except Exception as exc:
                        logger.error(f"Failed to read file {file_id} from disk: {exc}")
                        return
                    data = base64.b64encode(file_data).decode("utf-8")
                    meta = (file_item.meta or {})
                    mime_type = meta.get("content_type", "application/octet-stream")
                    _append_base64_media(mime_type, data)
                else:
                    logger.error(f"Failed to fetch file {file_id}: not found")
                return
            remote_media = await self._fetch_remote_media(url)
            if remote_media:
                media.append(remote_media)

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    await _append_media_from_url(item.get("image_url", {}).get("url", ""))
        elif isinstance(content, str):
            for mime_type, data in re.findall(r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)", content):
                _append_base64_media(mime_type, data)
            for file_url in re.findall(r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)", content):
                await _append_media_from_url(file_url)
            for remote_url in re.findall(r"!\[[^\]]*\]\((https?://[^)]+)\)", content):
                await _append_media_from_url(remote_url)
        return media

    def _decode_image_media_entry(
        self, media_entry: Dict[str, str]
    ) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Decode a base64 image payload into bytes plus size/orientation metadata."""
        image_bytes: Optional[bytes] = None
        size_str: Optional[str] = None
        orientation: Optional[str] = None
        data_b64 = (media_entry or {}).get("data") or ""
        if not data_b64:
            return image_bytes, size_str, orientation
        try:
            decoded_bytes = base64.b64decode(data_b64)
            image_bytes = decoded_bytes
            with Image.open(io.BytesIO(decoded_bytes)) as img:
                width, height = img.size
            size_str = f"{width}x{height}"
            if width > height:
                orientation = "horizontal"
            elif width < height:
                orientation = "vertical"
            else:
                orientation = "square"
        except Exception as exc:
            logger.error(f"Failed to inspect reference image dimensions: {exc}")
        return image_bytes, size_str, orientation
    async def _fetch_remote_media(self, url: str) -> Optional[Dict[str, str]]:
        """Download remote media URLs with basic validation and size limits."""
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return None
        try:
            parts = urllib.parse.urlsplit(url)
        except Exception:
            return None
        if parts.scheme not in ("http", "https"):
            return None
        if parts.username or parts.password:
            logger.error("Remote media URL contains userinfo; skipping.")
            return None
        hostname = parts.hostname
        if not hostname:
            return None
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if not await self._is_safe_remote_host(hostname, port):
            logger.error("Remote media host '%s' is not allowed; skipping.", hostname)
            return None
        try:
            async with httpx.AsyncClient(
                timeout=min(self.valves.REQUEST_TIMEOUT, 60),
                follow_redirects=False,
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    mime_type = response.headers.get("content-type", "").split(";")[0].lower()
                    if not (mime_type.startswith("image/") or mime_type.startswith("video/")):
                        logger.error("Unsupported remote media type '%s' from %s. Skipping.", mime_type, url)
                        return None
                    max_bytes = getattr(self.valves, "MAX_REMOTE_MEDIA_BYTES", 50 * 1024 * 1024)
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > max_bytes:
                                size_mb = max_bytes / (1024 * 1024)
                                logger.error(
                                    "Remote media %s exceeds %.0fMB decoded size (Content-Length). Skipping.",
                                    url,
                                    size_mb,
                                )
                                return None
                        except Exception:
                            pass
                    buffer = bytearray()
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        buffer.extend(chunk)
                        if max_bytes and len(buffer) > max_bytes:
                            size_mb = max_bytes / (1024 * 1024)
                            logger.error(
                                "Remote media %s exceeds %.0fMB decoded size. Skipping.",
                                url,
                                size_mb,
                            )
                            return None
                    data = base64.b64encode(bytes(buffer)).decode("utf-8")
                    return {"mimeType": mime_type, "data": data}
        except Exception as e:
            logger.error(f"Failed to fetch remote media {url}: {e}")
            return None

    @staticmethod
    def _is_global_ip(ip: Any) -> bool:
        """Return True only for globally routable IP addresses (no private/loopback/link-local/etc)."""
        try:
            return bool(getattr(ip, "is_global", False))
        except Exception:
            return False

    async def _is_safe_remote_host(self, hostname: str, port: int) -> bool:
        """Best-effort SSRF guard: only allow hosts that resolve to global IPs."""
        host = (hostname or "").strip().strip("[]").lower()
        if not host:
            return False
        if host in {"localhost"} or host.endswith(".localhost"):
            return False
        if host.endswith((".local", ".internal", ".lan", ".home")):
            return False
        try:
            ip = ipaddress.ip_address(host)
            return self._is_global_ip(ip)
        except ValueError:
            pass
        try:
            addr_infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
        except Exception as exc:
            logger.error("Failed to resolve remote media host '%s': %s", host, exc)
            return False
        if not addr_infos:
            return False
        for info in addr_infos:
            sockaddr = info[4]
            if not sockaddr:
                return False
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return False
            if not self._is_global_ip(ip):
                return False
        return True
    def _classify_media(self, media: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Split media into images and videos based on mimeType."""
        images: List[Dict[str, str]] = []
        videos: List[Dict[str, str]] = []
        for item in media:
            mime = (item.get("mimeType") or "").lower()
            if mime.startswith("image/"):
                images.append(item)
            elif mime.startswith("video/"):
                videos.append(item)
        return {"images": images, "videos": videos}
    async def _upload_video(
        self,
        __request__: Request,
        user: UserModel,
        file_path: str,
        mime_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload the finished video into Open WebUI's file store."""
        file_metadata = {"mime_type": mime_type}
        if metadata:
            file_metadata.update(metadata)
        filename = f"generated-video-{uuid.uuid4().hex}.mp4"
        with open(file_path, "rb") as file_obj:
            file_item = await run_in_threadpool(
                upload_file,
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=file_obj,
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,
                user=user,
                metadata=file_metadata,
            )
        file_id = getattr(file_item, "id", None)
        if file_id is None and isinstance(file_item, dict):
            file_id = file_item.get("id")
        if not file_id:
            raise RuntimeError("Upload succeeded but did not return a file id.")
        return __request__.app.url_path_for("get_file_content_by_id", id=file_id)
    def _write_response_body_to_tempfile(self, response: Any) -> Tuple[str, int]:
        """Stream an OpenAI download response into a temp file and return its size."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        total_bytes = 0
        try:
            for chunk in self._iter_response_body(response):
                if not chunk:
                    continue
                temp_file.write(chunk)
                total_bytes += len(chunk)
            temp_file.flush()
            return temp_path, total_bytes
        except Exception:
            with contextlib.suppress(Exception):
                os.unlink(temp_path)
            raise
        finally:
            with contextlib.suppress(Exception):
                temp_file.close()

    def _iter_response_body(self, response: Any) -> Iterable[bytes]:
        """Yield bytes from a variety of httpx/openai body types."""
        iter_bytes = getattr(response, "iter_bytes", None)
        if callable(iter_bytes):
            iter_bytes_fn = cast(Callable[[int], Iterable[bytes]], iter_bytes)
            for chunk in iter_bytes_fn(self.DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                yield bytes(chunk)
            return

        read = getattr(response, "read", None)
        if callable(read):
            read_fn = cast(Callable[[int], bytes], read)
            while True:
                chunk = read_fn(self.DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                yield bytes(chunk)
            return

        yield bytes(response or b"")

    def _format_data(
        self,
        *,
        is_stream: bool,
        model: str = "",
        content: Optional[str] = "",
        usage: Optional[dict] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        """OpenAI-compatible response wrapper for streaming/non-streaming output."""
        data: Dict[str, Any] = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if is_stream:
            is_stop_chunk = finish_reason == "stop" and content is None
            delta: Dict[str, Any] = {}
            if not is_stop_chunk:
                delta["role"] = "assistant"
                if content is not None:
                    delta["content"] = content
            data["choices"] = [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "delta": delta,
                }
            ]
        else:
            message_content = content or ""
            data["choices"] = [
                {
                    "finish_reason": finish_reason or "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                    },
                }
            ]
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n" if is_stream else json.dumps(data)
    def _build_help_markdown(self) -> str:
        """Generate the help/advice markdown rendered when the user types 'help'."""
        lines: List[str] = []
        valves = self.valves
        default_duration = valves.DEFAULT_DURATION
        allowed = sorted(self.allowed_resolutions) or sorted(self.BASE_RESOLUTIONS)
        portrait_primary = "720x1280" if "720x1280" in allowed else allowed[0]
        landscape_primary = "1280x720" if "1280x720" in allowed else allowed[-1]
        pro_sizes = sorted(self.PRO_ONLY_RESOLUTIONS)
        lines.append("# Sora video assistant tips")
        lines.append("")
        lines.append("Describe the short clip you want. We’ll turn that idea — plus any optional reference image — into a new video or a remix of your last one.")
        lines.append("Type `help` anytime to reopen this guide.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## How long can clips be?")
        lines.append("")
        lines.append("Sora only renders three clip lengths: **4 seconds**, **8 seconds**, or **12 seconds**. Ask for any other number and we round UP to the next supported length (cap at 12).")
        lines.append("- Shortest possible output: **4 seconds** (even if you type 1 second).")
        lines.append(f"- Default if you don't specify: **{default_duration} seconds**.")
        lines.append("- To change it, write something like `make it 8 seconds` or `about 12s please`.")
        lines.append("")
        lines.append("## Picking the orientation")
        lines.append("")
        lines.append("Say it however feels natural — we map it to the nearest supported resolution:")
        lines.append(f"- `vertical`, `portrait`, `phone`, `TikTok`, `Reels` → **{portrait_primary}**.")
        lines.append(f"- `horizontal`, `landscape`, `YouTube`, `widescreen` → **{landscape_primary}**.")
        if pro_sizes:
            lines.append(f"- Ask for `Sora Pro` when you need the taller/wider sizes: {', '.join(pro_sizes)}.")
        lines.append("- Saying `square` or `1:1` picks whichever of the above is closest right now (true square isn’t available yet).")
        lines.append("")
        lines.append("## Using reference images")
        lines.append("")
        lines.append("Upload one image to guide the vibe. We'll automatically crop/resize it to match the selected video resolution when needed.")
        lines.append("If you prefer a specific framing, mention the exact orientation or size in your prompt.")
        lines.append("")
        lines.append("## Variations (extra takes)")
        lines.append("")
        if getattr(valves, "ENABLE_VARIATIONS", False):
            lines.append(f"Ask for up to {valves.MAX_VARIATIONS} takes at once: `give me 3 variations of this 8s idea`. Each take shows up as its own playable link.")
        else:
            lines.append("Variations are disabled in this space right now.")
        lines.append("")
        lines.append("## Remixing your last video")
        lines.append("")
        lines.append("After a video finishes, describe the tweak: `Change the color of the monster to orange`, `Make the last clip louder`, etc. We automatically target the most recent Sora video.")
        lines.append("Sora download links expire after about an hour, so we always store your finished clip in WebUI immediately.")
        lines.append("If that earlier Sora video is no longer available from OpenAI, we’ll let you know so you can regenerate it first.")
        lines.append("")
        lines.append("## What happens while it's working?")
        lines.append("")
        lines.append("- The chat posts live status updates (queued → rendering → downloading).")
        lines.append("- You’ll see a progress percentage so you know roughly how long is left.")
        lines.append("- Finished clips appear right in the chat with a download link you can reuse.")
        lines.append("")
        lines.append("## Sample prompts")
        lines.append("")
        lines.append("- `Make an 8 second vertical TikTok of latte art forming in slow motion.`")
        lines.append("- `Give me a 4 second 1280x720 loop where our logo assembles from floating glass.`")
        lines.append("- `Change the last video so the monster is orange and the lighting feels spooky.`")
        return "\n".join(lines)
    def _build_video_viewer_html(
        self,
        *,
        video_url: str,
        prompt: str,
        model_used: str,
        seconds_bucket: str,
        size_used: str,
        file_size_bytes: Optional[int] = None,
        variation_index: Optional[int] = None,
        variation_total: Optional[int] = None,
    ) -> str:
        """Construct the HTML payload that Open WebUI will render inside an iframe."""
        logger.info(
            "Building video viewer HTML (url=%s model=%s seconds=%s size=%s)",
            video_url,
            model_used,
            seconds_bucket,
            size_used,
        )
        prompt_text = (prompt or "").strip()
        title = "Your Sora video"
        if variation_total and variation_total > 1 and variation_index:
            title = f"Your Sora video (take {variation_index}/{variation_total})"
        prompt_snippet = prompt_text[:200]
        if len(prompt_text) > 200:
            prompt_snippet += "..."
        prompt_snippet = html.escape(prompt_snippet or "(no prompt provided)", quote=True)
        model_label = html.escape(model_used or "sora-2", quote=True)
        seconds_label = html.escape(str(seconds_bucket or "4"), quote=True)
        size_label = html.escape(size_used or "720x1280", quote=True)
        video_src = html.escape(video_url, quote=True)
        file_size_line = ""
        if file_size_bytes is not None:
            if file_size_bytes >= 1024 * 1024:
                human_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
            elif file_size_bytes >= 1024:
                human_size = f"{file_size_bytes / 1024:.1f} KB"
            else:
                human_size = f"{file_size_bytes} B"
            file_size_line = f'<p><strong>File size:</strong> {human_size}</p>'
        download_link = f'<a href="{video_src}" target="_blank" rel="noopener noreferrer">Download the video</a>'
        return (
            f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      :root {{
        color-scheme: light dark;
      }}
      body {{
        margin: 0;
        background-color: transparent;
      }}
      div {{
        color: #111827;
      }}
      @media (prefers-color-scheme: dark) {{
        div {{
          color: #e5e7eb;
        }}
      }}
    </style>
  </head>
  <body>
    <div>
      <h3>{title}</h3>
      <p><strong>Prompt:</strong> <code>{prompt_snippet}</code></p>
      <p><strong>Model:</strong> {model_label} · <strong>Length:</strong> {seconds_label}s · <strong>Resolution:</strong> {size_label}</p>
      {file_size_line}
      <video controls preload="metadata" style="max-width: 100%; height: auto;">
        <source src="{video_src}" type="video/mp4" />
        Your browser cannot play this video. {download_link}
      </video>
      <p>{download_link}</p>
    </div>
  </body>
</html>"""
        )
    async def _emit_video_embed(
        self,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        html_content: str,
    ) -> None:
        """Send an embed payload to the chat event emitter."""
        if not emitter or not html_content:
            return
        await emitter({"type": "embeds", "data": {"embeds": [html_content]}})
    async def _emit_video_progress(
        self,
        status_data: Dict[str, Any],
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        *,
        variation_index: Optional[int] = None,
        variation_total: Optional[int] = None,
    ):
        """Emit a status update with progress percentage based on job status."""
        if not emitter:
            logger.info(
                "Skipping progress emit (no emitter). status_data=%s",
                status_data,
            )
            return
        progress = status_data.get("progress")
        status = (status_data.get("status") or "").lower()
        if progress is None:
            if status in ("queued", "pending"):
                progress = 5
            elif status in ("in_progress", "processing", "running"):
                progress = 50
            elif status in ("finalising", "finalizing", "postprocessing"):
                progress = 80
            elif status in ("completed", "succeeded"):
                progress = 100
            elif status in ("failed", "error", "cancelled"):
                progress = 100
            else:
                progress = 0
        logger.info(
            "Emitting video progress: status=%s progress=%s emitter_present=%s",
            status,
            progress,
            bool(emitter),
        )
        variation_label = ""
        if variation_total and variation_total > 1:
            current = variation_index if variation_index and variation_index > 0 else 1
            variation_label = f"[take {current}/{variation_total}] "
        display_status = (status or "unknown").replace("_", " ")
        description = f"{variation_label}Video generation progress: {progress}% complete (status: {display_status})"
        done_statuses = {"completed", "succeeded", "failed", "error", "cancelled"}
        await emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": status in done_statuses,
                    "progress": progress,
                },
            }
        )

    @staticmethod
    def _safe_json(value: Any) -> str:
        """Best-effort serialization helper for logging payloads."""
        try:
            return json.dumps(value, separators=(",", ":"), default=str)
        except Exception:
            return str(value)

    def _summarize_video_failure(self, status_data: Dict[str, Any]) -> str:
        """Return a concise description for a failed Sora job."""

        def _extract(detail: Any) -> Optional[str]:
            """Traverse nested structures to surface the first textual error detail."""
            if not detail:
                return None
            if isinstance(detail, str):
                text = detail.strip()
                return text or None
            if isinstance(detail, dict):
                code = detail.get("code") or detail.get("error_code") or detail.get("type")
                for key in ("message", "detail", "description", "reason", "summary", "error"):
                    value = detail.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        if code and isinstance(code, str) and code.strip():
                            return f"{code.strip()}: {text}"
                        return text
                for key in ("errors", "details", "data"):
                    nested = detail.get(key)
                    text = _extract(nested)
                    if text:
                        return text
                return None
            if isinstance(detail, list):
                for item in detail:
                    text = _extract(item)
                    if text:
                        return text
            return None

        if not isinstance(status_data, dict):
            return ""

        for candidate_key in (
            "status_details",
            "status_detail",
            "error",
            "last_error",
            "failure_reason",
        ):
            text = _extract(status_data.get(candidate_key))
            if text:
                return text
        output = status_data.get("output")
        if isinstance(output, dict):
            for key in ("errors", "error", "status_details"):
                text = _extract(output.get(key))
                if text:
                    return text
        text = _extract(status_data.get("status_message"))
        if text:
            return text
        return ""

    @staticmethod
    def _task_model_response_format(*, max_variations: int) -> ResponseFormatJSONSchema:
        """Return a strict JSON Schema response_format for task-model parameter extraction."""
        max_variations = int(max_variations) if max_variations and int(max_variations) > 0 else 1
        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": ["generate", "remix"]},
                "seconds": {"type": "integer", "enum": [4, 8, 12]},
                "size": {
                    "type": "string",
                    "description": "Video resolution bucket as WIDTHxHEIGHT (e.g., 720x1280).",
                },
                "video_model": {"type": "string", "enum": ["sora-2", "sora-2-pro"]},
                "reuse_previous_image": {"type": "boolean"},
                "variations": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": max_variations},
                        {"type": "null"},
                    ],
                    "description": "Number of takes to generate (use 1 or null when not requested).",
                },
            },
            # Structured Outputs strict schemas require every property key to appear in 'required'.
            "required": [
                "intent",
                "seconds",
                "size",
                "video_model",
                "reuse_previous_image",
                "variations",
            ],
        }
        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": "sora_task_params",
                "description": "Extract parameters for a Sora video request.",
                "schema": schema,
                "strict": True,
            },
        }
        return response_format

    async def _analyse_prompt_with_task_model(
        self,
        prompt: str,
        has_media: bool,
        __user__: dict,
        body: dict,
        user_obj: Optional[UserModel],
        __request__: Request,
        emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
        has_image: bool = False,
        image_size: Optional[str] = None,
        image_orientation: Optional[str] = None,
        current_image_attached: bool = False,
        previous_image_available: bool = False,
        previous_image_prompt: Optional[str] = None,
        previous_image_size: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the task model (when enabled) to extract structured video parameters."""
        if not prompt:
            raise ValueError("No prompt supplied for task model analysis.")
        task_model = (getattr(self.valves, "TASK_MODEL_TO_USE", "") or "").strip()
        if not task_model:
            raise ValueError("TASK_MODEL_TO_USE is not set in valves.")
        standard_sizes = set(self.resolution_policy.get_supported("sora-2"))
        pro_sizes = set(self.resolution_policy.get_supported("sora-2-pro"))
        pro_only_sizes = sorted(pro_sizes - standard_sizes)
        standard_sizes_text = ", ".join(sorted(standard_sizes))
        pro_only_text = ", ".join(pro_only_sizes)
        pro_request_hint = pro_only_text or "1024x1792 or 1792x1024"
        previous_prompt_context = (previous_image_prompt or "none")
        previous_prompt_context = previous_prompt_context.replace("\r", " ").replace("\n", " ")
        previous_prompt_context = previous_prompt_context.replace('"', "'")
        additional_context = f"""Context:
- Current message uploaded an image: {current_image_attached}; size="{image_size or 'none'}"; orientation="{image_orientation or 'none'}"
- Previous image available: {previous_image_available}; size="{previous_image_size or 'unknown'}"; prompt="{previous_prompt_context}"
- Resolution policy: sora-2 -> {standard_sizes_text}. sora-2-pro adds -> {pro_only_text or 'no extra sizes'}.
- Server snaps unsupported ratios to the closest allowed size (vertical/horizontal hints are fine). Reference images are automatically resized/cropped to fit the chosen bucket when needed.
- Upgrade to sora-2-pro only when the user explicitly mentions pro mode or requests {pro_request_hint}.
- Set `reuse_previous_image` true only when no new image is attached and the user clearly asks to keep or reuse the earlier one."""
        analysis_prompt = f"""You are a parameter extraction assistant for a video generation API (Sora).
Extract the most appropriate parameters from this user request.
User prompt: "{prompt}"
Has reference media (image or video): {has_media}
{additional_context}
Return ONLY one JSON object with this shape:
{{
  "intent": "generate" or "remix",
  "seconds": <int>,
  "size": "WIDTHxHEIGHT",
  "video_model": "sora-2" or "sora-2-pro",
  "reuse_previous_image": true or false,
  "variations": <int or null>
}}
Guidelines:
1. intent: "generate" for new clips (even with reference images) and "remix" when the user clearly wants to change an existing clip or uploaded a video to modify it.
2. seconds: Sora only accepts 4, 8, or 12 seconds. Round UP to the next supported bucket (cap at 12). If nothing is specified, use {self.valves.DEFAULT_DURATION}.
3. size: Pick from the allowed list for the chosen model (sora-2 -> {standard_sizes_text}; sora-2-pro also includes {pro_only_text or 'no additional sizes'}). When the user gives an unsupported resolution, choose the closest allowed bucket that matches their orientation hint (vertical/horizontal/square). The server will handle the snapping.
4. video_model: Default to "sora-2". Switch to "sora-2-pro" only if the user explicitly mentions pro mode or requests {pro_request_hint}.
5. reuse_previous_image: true only when no new image is attached and the text clearly asks to keep/reuse/edit the earlier image. Otherwise false.
6. variations: if the user asks for multiple takes, output an integer 1..{self.valves.MAX_VARIATIONS}; otherwise output 1 (or null)."""
        try:
            await self.emit_status("Analysing prompt...", emitter=emitter)

            async with self._openai_client() as client:
                response_format = self._task_model_response_format(max_variations=self.valves.MAX_VARIATIONS)
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=task_model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    max_completion_tokens=1024,
                    response_format=response_format,
                )
            content_value = ""
            choice0 = None
            message = None
            try:
                choice0 = response.choices[0]
                message = getattr(choice0, "message", None)
            except Exception:
                choice0 = None
                message = None

            refusal = getattr(message, "refusal", None) if message is not None else None
            if isinstance(refusal, str) and refusal.strip():
                await self.emit_status(
                    f"Task model refused: {refusal.strip()}",
                    done=True,
                    emitter=emitter,
                )
                raise RuntimeError(f"Task model refused: {refusal.strip()}")

            finish_reason = getattr(choice0, "finish_reason", None) if choice0 is not None else None
            if message is not None:
                content_value = getattr(message, "content", "") or ""
            else:
                content_value = getattr(response, "text", "") or ""

            content = self._normalise_model_content(content_value)
            if getattr(self.valves, "ENABLE_LOGGING", False):
                logger.info(
                    "Task model raw response. model=%s finish_reason=%s content_preview=%s",
                    task_model,
                    finish_reason,
                    (content or "").strip().replace("\n", " ")[:4000],
                )
            try:
                params = json.loads((content or "").strip())
            except Exception:
                json_blob = self._extract_first_json_object(content or "")
                params = json.loads(json_blob) if json_blob else None
            if isinstance(params, dict):
                if isinstance(finish_reason, str) and finish_reason and finish_reason != "stop":
                    logger.warning(
                        "Task model finish_reason=%s but JSON parsed successfully; continuing.",
                        finish_reason,
                    )
                validated = self._validate_task_model_params(params, prompt)
                logger.info(f"Task model parameters: {validated}")
                return validated
            if isinstance(finish_reason, str) and finish_reason and finish_reason != "stop":
                raise RuntimeError(
                    f"Task model did not complete cleanly (finish_reason={finish_reason})."
                )
            logger.error(
                "Task model returned non-JSON content. model=%s finish_reason=%s content_preview=%s",
                task_model,
                finish_reason,
                (content or "").strip().replace("\n", " ")[:2000],
            )
            raise ValueError("Task model response did not contain valid JSON")
        except Exception as e:
            logger.error(f"Task model call failed: {e}")
            import traceback
            logger.info(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Task model call failed: {e}") from e
    def _validate_task_model_params(
        self, params: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Normalize the raw task-model output and guarantee required keys."""
        validated: Dict[str, Any] = {}
        # intent
        intent = params.get("intent", "generate")
        if intent not in ("generate", "remix"):
            intent = "generate"
        validated["intent"] = intent
        # seconds / duration_seconds
        raw_seconds = params.get("seconds", params.get("duration_seconds", self.valves.DEFAULT_DURATION))
        duration = self.valves.DEFAULT_DURATION
        try:
            candidate = int(raw_seconds)
            if candidate > 0:
                duration = candidate
        except Exception as exc:
            logger.debug(f"Failed to parse duration seconds '{raw_seconds}': {exc}")
            pass
        validated["duration_seconds"] = int(self._map_duration_to_sora_seconds(duration))
        # video model selection
        video_model = params.get("video_model")
        if video_model not in {"sora-2", "sora-2-pro"}:
            video_model = self.valves.MODEL
        validated["video_model"] = video_model
        # size / resolution
        raw_size = params.get("size") or params.get("resolution") or self.valves.DEFAULT_RESOLUTION
        validated["size"] = raw_size
        # optional variations
        variations = params.get("variations")
        if variations is not None and getattr(self.valves, "ENABLE_VARIATIONS", False):
            try:
                v = int(variations)
                if v < 1:
                    v = 1
                if v > self.valves.MAX_VARIATIONS:
                    v = self.valves.MAX_VARIATIONS
                validated["variations"] = v
            except Exception as exc:
                logger.debug(f"Failed to parse variations value '{variations}': {exc}")
                pass
        # reuse flag
        reuse_previous_image = bool(params.get("reuse_previous_image", False))
        validated["reuse_previous_image"] = reuse_previous_image
        return validated
    @staticmethod
    def _map_duration_to_sora_seconds(duration_seconds: int) -> str:
        """Convert arbitrary user-specified seconds to Sora's discrete duration buckets (ceil to 4/8/12)."""
        try:
            value = int(duration_seconds)
        except Exception:
            value = 4
        if value <= 0:
            value = 4
        if value <= 4:
            return "4"
        if value <= 8:
            return "8"
        return "12"

    def _resize_image_to(self, image_bytes: bytes, target_size: str) -> bytes:
        """Crop and scale an image to match Sora's required aspect and resolution."""
        if not image_bytes:
            raise ValueError("No image data provided for resizing.")
        if not target_size or "x" not in target_size:
            raise ValueError("Invalid target size string.")
        width_str, height_str = target_size.lower().split("x", 1)
        try:
            target_width = int(width_str)
            target_height = int(height_str)
        except Exception as exc:
            raise ValueError(f"Invalid target size '{target_size}'.") from exc
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target dimensions must be positive.")
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            source_width, source_height = img.size
            if source_width <= 0 or source_height <= 0:
                raise ValueError("Source image has invalid dimensions.")
            target_ratio = target_width / target_height
            source_ratio = source_width / source_height
            working = img
            if abs(source_ratio - target_ratio) > 0.01:
                if source_ratio > target_ratio:
                    new_width = int(source_height * target_ratio)
                    new_width = max(1, min(new_width, source_width))
                    left = max((source_width - new_width) // 2, 0)
                    working = working.crop((left, 0, left + new_width, source_height))
                else:
                    new_height = int(source_width / target_ratio)
                    new_height = max(1, min(new_height, source_height))
                    top = max((source_height - new_height) // 2, 0)
                    working = working.crop((0, top, source_width, top + new_height))
            resized = working.resize((target_width, target_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=95)
            return buffer.getvalue()
    async def _get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Fetch a user record using the synchronous Users helper in a worker thread."""
        try:
            return await run_in_threadpool(Users.get_user_by_id, user_id)
        except Exception as exc:
            logger.error(f"Failed to load user {user_id}: {exc}")
            return None
    async def _get_file_by_id(self, file_id: str) -> Optional[Any]:
        """Fetch a file record by id so that stored media can be re-read."""
        try:
            from open_webui.models.files import Files
            return await run_in_threadpool(Files.get_file_by_id, file_id)
        except Exception as exc:
            logger.error(f"Failed to load file {file_id}: {exc}")
            return None
    async def pipes(self) -> List[dict]:
        """Return a list of pipes for Open WebUI discovery APIs."""
        return [{"id": self.PIPE_ID, "name": "OpenAI: Sora Video"}]
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Response:
        """Entry point invoked by Open WebUI per request (streaming or standard)."""
        self._apply_logging_valve()
        logger.info(
            "pipe() invoked (user_id=%s, body_keys=%s, emitter_present=%s)",
            __user__.get("id"),
            list(body.keys()),
            bool(__event_emitter__),
        )
        user = await self._get_user_by_id(__user__["id"])
        async def handle_request() -> Dict[str, Any]:
            """Core coroutine that performs validation, dispatch, and response building."""
            try:
                logger.info("handle_request start")
                model = self.valves.MODEL
                messages = body.get("messages", [])
                if not user:
                    return {"ok": False, "error": "Error: Unable to load user context.", "model": model}
                user_obj: UserModel = user
                (
                    prompt,
                    current_media,
                    artifacts,
                    previous_media,
                    previous_media_prompt,
                ) = await self._extract_prompt_and_media(messages)
                logger.info(
                    "Prompt/media extracted (prompt_len=%s, current_media=%s, previous_media=%s)",
                    len(prompt or ""),
                    len(current_media),
                    len(previous_media),
                )
                current_media_classified = self._classify_media(current_media)
                current_image_media = current_media_classified["images"]
                current_reference_image = current_image_media[0] if current_image_media else None
                current_image_bytes: Optional[bytes] = None
                current_image_size_str: Optional[str] = None
                current_image_orientation: Optional[str] = None
                if current_reference_image:
                    (
                        current_image_bytes,
                        current_image_size_str,
                        current_image_orientation,
                    ) = self._decode_image_media_entry(current_reference_image)

                previous_media_classified = self._classify_media(previous_media)
                previous_image_media = previous_media_classified["images"]
                previous_reference_image = previous_image_media[0] if previous_image_media else None
                previous_image_bytes: Optional[bytes] = None
                previous_image_size_str: Optional[str] = None
                previous_image_orientation: Optional[str] = None
                if not current_reference_image and previous_reference_image:
                    (
                        previous_image_bytes,
                        previous_image_size_str,
                        previous_image_orientation,
                    ) = self._decode_image_media_entry(previous_reference_image)

                previous_prompt_for_model = (previous_media_prompt or "").strip()
                previous_prompt_for_model = (
                    previous_prompt_for_model.replace("\r", " ").replace("\n", " ")
                )
                if len(previous_prompt_for_model) > 400:
                    previous_prompt_for_model = previous_prompt_for_model[:400]
                if prompt:
                    p = prompt.strip().lower()
                    if p in ("help", "sora help", "video help") or p.startswith("help "):
                        help_markdown = self._build_help_markdown()
                        return {
                            "ok": True,
                            "help": True,
                            "help_markdown": help_markdown,
                            "model": model,
                        }
                if not prompt:
                    return {"ok": False, "error": "Error: No prompt text detected in request.", "model": model}
                if not self.valves.API_KEY:
                    return {"ok": False, "error": "Error: API_KEY not set in valves.", "model": model}
                dynamic_params = await self._analyse_prompt_with_task_model(
                    prompt,
                    bool(current_media),
                    __user__,
                    body,
                    user,
                    __request__,
                    emitter=__event_emitter__,
                    has_image=bool(current_reference_image),
                    image_size=current_image_size_str,
                    image_orientation=current_image_orientation,
                    current_image_attached=bool(current_reference_image),
                    previous_image_available=bool(previous_reference_image),
                    previous_image_prompt=previous_prompt_for_model,
                    previous_image_size=previous_image_size_str,
                )
                model = dynamic_params.get("video_model") or self.valves.MODEL
                if model != "sora-2-pro" and "sora pro" in (prompt or "").lower():
                    logger.info("Detected 'sora pro' in prompt; upgrading model to sora-2-pro.")
                    model = "sora-2-pro"
                    dynamic_params["video_model"] = model
                logger.info("Dynamic params resolved: %s", dynamic_params)
                reuse_previous_image = bool(dynamic_params.get("reuse_previous_image"))
                selected_media = current_media if current_media else []
                media_source = "current" if current_media else "none"
                if not selected_media:
                    if reuse_previous_image and previous_media:
                        selected_media = previous_media
                        media_source = "previous"
                        if previous_reference_image and previous_image_bytes is None:
                            (
                                previous_image_bytes,
                                previous_image_size_str,
                                previous_image_orientation,
                            ) = self._decode_image_media_entry(previous_reference_image)
                    elif reuse_previous_image:
                        logger.info("reuse_previous_image requested but no previous media available; ignoring flag.")
                elif reuse_previous_image:
                    logger.info(
                        "Ignoring reuse_previous_image flag because a new image was attached in this turn."
                    )

                media = selected_media
                classified_media = self._classify_media(media)
                image_media = classified_media["images"]
                video_media = classified_media["videos"]
                has_media = bool(media)
                has_images = bool(image_media)
                has_videos = bool(video_media)
                reference_image = image_media[0] if image_media else None
                image_bytes: Optional[bytes] = None
                image_size_str: Optional[str] = None
                image_orientation: Optional[str] = None
                image_mime_override: Optional[str] = None
                if media_source == "current":
                    image_bytes = current_image_bytes
                    image_size_str = current_image_size_str
                    image_orientation = current_image_orientation
                elif media_source == "previous":
                    image_bytes = previous_image_bytes
                    image_size_str = previous_image_size_str
                    image_orientation = previous_image_orientation
                if reference_image and image_bytes is None:
                    image_bytes, image_size_str, image_orientation = self._decode_image_media_entry(reference_image)
                duration_value = dynamic_params.get("duration_seconds")
                requested_duration_display = duration_value
                default_duration = int(self.valves.DEFAULT_DURATION)
                validated_duration: Optional[int] = None
                if duration_value is not None:
                    try:
                        validated_duration = int(duration_value)
                    except Exception:
                        validated_duration = None
                if validated_duration is None or validated_duration <= 0:
                    await self.emit_status(
                        f"Invalid clip length requested, using default {default_duration} seconds instead.",
                        emitter=__event_emitter__,
                    )
                    validated_duration = default_duration
                seconds_str = self._map_duration_to_sora_seconds(validated_duration)
                snapped_seconds = int(seconds_str)
                if snapped_seconds != validated_duration:
                    requested_label = (
                        str(requested_duration_display)
                        if requested_duration_display is not None
                        else str(validated_duration)
                    )
                    await self.emit_status(
                        f"Requested clip length {requested_label}s isn't supported; using {snapped_seconds}s instead (supported: 4, 8, 12).",
                        emitter=__event_emitter__,
                    )
                dynamic_params["duration_seconds"] = snapped_seconds
                snapped_duration_seconds = snapped_seconds
                intent = (dynamic_params.get("intent") or "generate").lower()
                if intent not in ("generate", "remix"):
                    intent = "generate"
                dynamic_params["intent"] = intent
                request_label = (
                    requested_duration_display
                    if requested_duration_display is not None
                    else dynamic_params["duration_seconds"]
                )
                logger.info(
                    "User requested %ss, sending Sora seconds='%s'",
                    request_label,
                    seconds_str,
                )
                if intent == "remix":
                    media = []
                    image_media = []
                    video_media = []
                    has_media = False
                    has_images = False
                    has_videos = False
                    reference_image = None
                    image_bytes = None
                    image_size_str = None
                    image_orientation = None
                    image_mime_override = None
                requested_size_value = dynamic_params.get("size") or self.valves.DEFAULT_RESOLUTION
                size_decision = self.resolution_policy.choose_size(
                    requested_size_value,
                    prompt=prompt,
                    video_model=model,
                )
                if size_decision.requires_model_upgrade and model != "sora-2-pro":
                    logger.info(
                        "Upgrading video model to sora-2-pro to honor requested resolution %s",
                        size_decision.requested or requested_size_value,
                    )
                    model = "sora-2-pro"
                    dynamic_params["video_model"] = model
                    size_decision = self.resolution_policy.choose_size(
                        requested_size_value,
                        prompt=prompt,
                        video_model=model,
                    )
                dynamic_params["size"] = size_decision.resolved
                requested_size_norm = dynamic_params["size"]
                supported_resolutions = size_decision.supported
                allowed_sizes_text = size_decision.allowed_text
                if size_decision.substituted and size_decision.message:
                    await self.emit_status(size_decision.message, emitter=__event_emitter__)
                if intent == "generate" and has_images:
                    reference_assessment = self.resolution_policy.evaluate_reference(
                        image_size=image_size_str,
                        target_size=requested_size_norm,
                        supported=supported_resolutions,
                        allowed_text=allowed_sizes_text,
                    )
                    if reference_assessment.blocked:
                        message = reference_assessment.message or (
                            "Unable to use the provided reference image with the requested resolution."
                        )
                        await self.emit_status(message, True, True, emitter=__event_emitter__)
                        return {"ok": False, "error": message, "model": model}
                    if reference_assessment.should_resize:
                        if not image_bytes:
                            message = (
                                "Reference image bytes were unavailable, so resizing cannot be performed. "
                                "Please re-upload the image or provide one with the desired resolution."
                            )
                            message = f"{message} Supported sizes: {allowed_sizes_text}."
                            await self.emit_status(message, True, True, emitter=__event_emitter__)
                            return {"ok": False, "error": message, "model": model}
                        original_image_size = image_size_str
                        try:
                            image_bytes = self._resize_image_to(image_bytes, requested_size_norm)
                            image_size_str = requested_size_norm
                            image_mime_override = "image/jpeg"
                        except Exception as exc:
                            logger.error(f"Failed to resize reference image: {exc}")
                            message = (
                                "We could not safely resize your reference image. "
                                "Please upload an image at the target size or try again later."
                            )
                            message = f"{message} Supported sizes: {allowed_sizes_text}."
                            await self.emit_status(message, True, True, emitter=__event_emitter__)
                            return {"ok": False, "error": message, "model": model}
                        if reference_assessment.message:
                            await self.emit_status(reference_assessment.message, emitter=__event_emitter__)
                        elif original_image_size:
                            await self.emit_status(
                                f"Resized your reference image from {original_image_size} to {requested_size_norm} to match Sora's requirements.",
                                emitter=__event_emitter__,
                            )
                final_size_value = str(dynamic_params["size"])
                if final_size_value not in supported_resolutions:
                    message = (
                        f"Requested video size {final_size_value} is not supported. "
                        f"Allowed Sora sizes: {allowed_sizes_text}. Type `help` if you'd like to see the supported resolutions and options."
                    )
                    await self.emit_status(message, True, True, emitter=__event_emitter__)
                    return {"ok": False, "error": message, "model": model}
                size_value = final_size_value
                extra_body: Dict[str, Any] = {}
                reference_filename = ""
                reference_mime_type = ""
                if intent == "generate" and has_images:
                    reference_image = reference_image or image_media[0]
                    if image_bytes is None:
                        logger.error("Reference image bytes unavailable during generation.")
                        error_status = await self.emit_status(
                            "Invalid reference image provided.",
                            True,
                            True,
                            emitter=__event_emitter__,
                        )
                        return {
                            "ok": False,
                            "error": f"{error_status}Reference image could not be decoded.",
                            "model": model,
                        }
                    mime_type = (image_mime_override or reference_image.get("mimeType") or "image/png").lower()
                    extension = "png"
                    if "/" in mime_type:
                        suffix = mime_type.split("/", 1)[1].split("+", 1)[0]
                        if suffix:
                            extension = suffix
                    reference_filename = f"reference-image.{extension}"
                    reference_mime_type = mime_type
                    logger.info(
                        "Using reference image for generation (bytes=%s, mime=%s)",
                        len(image_bytes),
                        mime_type,
                    )
                    if len(image_media) > 1:
                        logger.info("Multiple reference images supplied; only the first will be used.")
                elif has_images:
                    logger.info(
                        "Reference images supplied but intent='%s'; skipping image references until remix flow is defined.",
                        intent,
                    )
                if has_media and has_videos:
                    logger.info(
                        "Video media provided but direct video uploads are not supported yet. Ignoring uploaded video blobs.",
                    )
                remix_metadata: Optional[Dict[str, Any]] = None
                remix_video_id: Optional[str] = None
                if intent == "remix":
                    remix_metadata = self._select_latest_artifact(artifacts)
                    if remix_metadata:
                        candidate = remix_metadata.get("video_id")
                        if isinstance(candidate, str) and candidate:
                            remix_video_id = candidate
                    if not remix_video_id:
                        message = (
                            "Unable to find a previous Sora video to remix. Please rerun the original generation request first."
                        )
                        await self.emit_status(message, True, True, emitter=__event_emitter__)
                        return {"ok": False, "error": message, "model": model}
                variation_total = 1
                if getattr(self.valves, "ENABLE_VARIATIONS", False):
                    try:
                        requested_variations = int(dynamic_params.get("variations") or 1)
                    except Exception:
                        requested_variations = 1
                    variation_total = max(1, min(self.valves.MAX_VARIATIONS, requested_variations))

                def build_reference_payload() -> Optional[Tuple[str, io.BytesIO, str]]:
                    """Return a tuple suitable for OpenAI uploads when a reference image exists."""
                    if intent != "generate" or not has_images or image_bytes is None:
                        return None
                    buffer = io.BytesIO(image_bytes)
                    return (reference_filename or "reference-image.png", buffer, reference_mime_type or "image/png")

                async with self._openai_client() as client:
                    job_results: List[Dict[str, Any]] = []
                    last_completed_data: Optional[Dict[str, Any]] = None

                    async def call_videos(method: Callable[..., Any], *args, **kwargs):
                        """Run potentially blocking SDK calls in a worker thread."""
                        return await asyncio.to_thread(method, *args, **kwargs)

                    if remix_video_id:
                        try:
                            await call_videos(
                                client.videos.retrieve,
                                remix_video_id,
                                timeout=self.valves.REQUEST_TIMEOUT,
                            )
                        except OpenAIError as exc:
                            logger.warning("Remix source %s unavailable: %s", remix_video_id, exc)
                            friendly = (
                                "That earlier Sora video is no longer available from OpenAI. "
                                "Please create a fresh clip first, then ask for the remix again."
                            )
                            await self.emit_status(friendly, True, True, emitter=__event_emitter__)
                            return {"ok": False, "error": friendly, "model": model}

                    async def run_single_job(variation_index: int) -> Dict[str, Any]:
                        """Submit/poll/download a single variation and return its metadata."""
                        variation_label = f"[take {variation_index}/{variation_total}] "
                        await self.emit_status(
                            f"{variation_label}Job accepted. Working on your video...",
                            emitter=__event_emitter__,
                        )
                        action_word = "remix" if remix_video_id else "generate"
                        await self.emit_status(
                            f"{variation_label}Submitting {action_word} video job ({snapped_duration_seconds}s @ {dynamic_params['size']})...",
                            emitter=__event_emitter__,
                        )
                        snippet = (prompt or "")[:200]
                        payload_preview = {
                            "model": model,
                            "prompt": snippet + ("..." if len(prompt or "") > 200 else ""),
                        }
                        if not remix_video_id:
                            payload_preview.update(
                                {
                                    "seconds": seconds_str,
                                    "size": size_value,
                                    "has_reference_image": bool(reference_filename and image_bytes is not None),
                                }
                            )
                        else:
                            payload_preview["remix_video_id"] = remix_video_id
                        create_response = None
                        try:
                            if remix_video_id:
                                log_payload = {
                                    "model": model,
                                    "prompt": prompt[:200],
                                    "remix_video_id": remix_video_id,
                                    "extra_body": extra_body,
                                }
                                logger.info("Video remix request: %s", self._safe_json(log_payload))
                                create_response = await call_videos(
                                    client.videos.remix,
                                    remix_video_id,
                                    prompt=prompt,
                                    extra_body=extra_body or None,
                                    timeout=self.valves.REQUEST_TIMEOUT,
                                )
                            else:
                                create_kwargs = {
                                    "model": model,
                                    "prompt": prompt,
                                    "seconds": seconds_str,
                                    "size": size_value,
                                    "timeout": self.valves.REQUEST_TIMEOUT,
                                }
                                reference_payload = build_reference_payload()
                                reference_meta = None
                                if reference_payload:
                                    filename, buffer, mime_type = reference_payload
                                    buffer.seek(0, 2)
                                    size_bytes = buffer.tell()
                                    buffer.seek(0)
                                    reference_meta = {
                                        "filename": filename,
                                        "mime_type": mime_type,
                                        "size_bytes": size_bytes,
                                    }
                                    create_kwargs["input_reference"] = (filename, buffer, mime_type)
                                if extra_body:
                                    create_kwargs["extra_body"] = extra_body
                                log_payload = {k: v for k, v in create_kwargs.items() if k != "input_reference"}
                                if reference_meta:
                                    log_payload["input_reference"] = reference_meta
                                logger.info("Video create request (intent=%s): %s", intent, self._safe_json(log_payload))
                                create_response = await call_videos(client.videos.create, **create_kwargs)
                        except OpenAIError as e:
                            logger.error("Video %s request failed: %s", action_word, e)
                            error_status = await self.emit_status(
                                "An error occurred while submitting the video job",
                                True,
                                True,
                                emitter=__event_emitter__,
                            )
                            raise RuntimeError(f"{error_status}Error from API: {str(e)}") from e
                        except Exception as e:
                            logger.error("Unexpected error during video %s: %s", action_word, e)
                            error_status = await self.emit_status(
                                "An error occurred while submitting the video job",
                                True,
                                True,
                                emitter=__event_emitter__,
                            )
                            raise RuntimeError(f"{error_status}Error from API: {str(e)}") from e
                        response_data = create_response.model_dump()
                        job_id = response_data.get("id")
                        if not job_id:
                            logger.error("Video API response missing job id")
                            error_status = await self.emit_status(
                                "Video API did not return a job id",
                                True,
                                True,
                                emitter=__event_emitter__,
                            )
                            raise RuntimeError(f"{error_status}Video API did not return a job id.")
                        await self.emit_status(
                            f"{variation_label}Video job created. Polling for status...",
                            emitter=__event_emitter__,
                        )
                        start_time = time.time()
                        last_status: Optional[str] = None
                        last_progress: Optional[int] = None
                        final_status_data: Optional[Dict[str, Any]] = response_data
                        max_consecutive_errors = self.valves.STATUS_POLL_MAX_ERRORS
                        consecutive_errors = 0
                        while True:
                            if time.time() - start_time > self.valves.MAX_POLL_TIME_SECONDS:
                                error_status = await self.emit_status(
                                    f"{variation_label}Video job timed out while waiting for completion.",
                                    True,
                                    True,
                                    emitter=__event_emitter__,
                                )
                                raise RuntimeError(f"{error_status}Video generation timed out.")
                            try:
                                status_response = await call_videos(
                                    client.videos.retrieve,
                                    job_id,
                                    timeout=self.valves.REQUEST_TIMEOUT,
                                )
                                consecutive_errors = 0
                            except OpenAIError as e:
                                consecutive_errors += 1
                                if consecutive_errors >= max_consecutive_errors:
                                    logger.error(
                                        "Video status poll failed %s times consecutively: %s",
                                        consecutive_errors,
                                        e,
                                    )
                                    error_status = await self.emit_status(
                                        "Failed to poll video job status",
                                        True,
                                        True,
                                        emitter=__event_emitter__,
                                    )
                                    raise RuntimeError(f"{error_status}Unable to poll job status: {str(e)}") from e
                                logger.warning(
                                    "Video status poll failed (attempt %s/%s): %s",
                                    consecutive_errors,
                                    max_consecutive_errors,
                                    e,
                                )
                                await asyncio.sleep(min(2, self.valves.POLL_INTERVAL_SECONDS))
                                continue
                            status_data = status_response.model_dump()
                            final_status_data = status_data
                            status = (status_data.get("status") or "").lower()
                            progress = status_data.get("progress")
                            should_emit = status != last_status or progress != last_progress
                            if should_emit or last_status is None:
                                await self._emit_video_progress(
                                    status_data,
                                    __event_emitter__,
                                    variation_index=variation_index,
                                    variation_total=variation_total,
                                )
                                last_status, last_progress = status, progress
                            job_status = status
                            if job_status in ("completed", "succeeded"):
                                break
                            if job_status in ("failed", "error", "cancelled"):
                                failure_reason = self._summarize_video_failure(status_data)
                                payload_dump = self._safe_json(status_data)
                                if len(payload_dump) > 2000:
                                    payload_dump = payload_dump[:2000] + "...(truncated)"
                                if failure_reason:
                                    logger.error(
                                        "Video job %s (%s) failed: %s | payload=%s",
                                        job_id,
                                        job_status,
                                        failure_reason,
                                        payload_dump,
                                    )
                                else:
                                    logger.error(
                                        "Video job %s (%s) failed without details. payload=%s",
                                        job_id,
                                        job_status,
                                        payload_dump,
                                    )
                                detail_message = f"{variation_label}Video job {job_status}."
                                if failure_reason:
                                    detail_message = f"{detail_message} Details: {failure_reason}"
                                error_status = await self.emit_status(
                                    detail_message,
                                    True,
                                    True,
                                    emitter=__event_emitter__,
                                )
                                raise RuntimeError(error_status or detail_message)
                            await asyncio.sleep(self.valves.POLL_INTERVAL_SECONDS)
                        await self.emit_status(
                            f"{variation_label}Video generation is complete. Downloading your video...",
                            emitter=__event_emitter__,
                        )
                        try:
                            content_response = await call_videos(
                                client.videos.download_content,
                                job_id,
                                timeout=self.valves.REQUEST_TIMEOUT,
                            )
                            temp_path, video_size_bytes = await asyncio.to_thread(
                                self._write_response_body_to_tempfile,
                                content_response,
                            )
                        except OpenAIError as e:
                            logger.error("Failed to download video content: %s", e)
                            error_status = await self.emit_status(
                                "Failed to download generated video",
                                True,
                                True,
                                emitter=__event_emitter__,
                            )
                            raise RuntimeError(f"{error_status}Unable to download video: {str(e)}") from e
                        try:
                            video_url = await self._upload_video(
                                __request__=__request__,
                                user=user_obj,
                                file_path=temp_path,
                                mime_type="video/mp4",
                                metadata={
                                    "size_bytes": video_size_bytes,
                                    "variation_index": variation_index,
                                    "variation_total": variation_total,
                                },
                            )
                        except Exception as e:
                            error_status = await self.emit_status(
                                "Failed to store generated video",
                                True,
                                True,
                                emitter=__event_emitter__,
                            )
                            raise RuntimeError(f"{error_status}Video upload failed: {str(e)}") from e
                        finally:
                            with contextlib.suppress(Exception):
                                os.unlink(temp_path)
                        job_seconds = str((final_status_data or {}).get("seconds") or seconds_str)
                        job_size_value = str((final_status_data or {}).get("size") or size_value)
                        embed_html = self._build_video_viewer_html(
                            video_url=video_url,
                            prompt=prompt,
                            model_used=model,
                            seconds_bucket=job_seconds,
                            size_used=job_size_value,
                            file_size_bytes=video_size_bytes,
                            variation_index=variation_index,
                            variation_total=variation_total,
                        )
                        await self._emit_video_embed(__event_emitter__, embed_html)
                        if getattr(self.valves, "IMMEDIATE_DELETE_JOBS", False):
                            try:
                                await call_videos(
                                    client.videos.delete,
                                    job_id,
                                    timeout=self.valves.REQUEST_TIMEOUT,
                                )
                            except OpenAIError as e:
                                logger.error("Failed to delete video job %s: %s", job_id, e)
                        result = {
                            "job_id": job_id,
                            "video_url": video_url,
                            "embed_html": embed_html,
                            "seconds_bucket": job_seconds,
                            "size_value": job_size_value,
                            "model": model,
                            "file_size_bytes": video_size_bytes,
                            "variation_index": variation_index,
                            "variation_total": variation_total,
                            "usage": (final_status_data or {}).get("usage") if isinstance(final_status_data, dict) else None,
                        }
                        return result
                    for variation_index in range(1, variation_total + 1):
                        try:
                            job_result = await run_single_job(variation_index)
                            job_results.append(job_result)
                            last_completed_data = job_result.get("usage") if job_result.get("usage") else last_completed_data
                        except RuntimeError as job_error:
                            contextual_error = f"[take {variation_index}/{variation_total}] {job_error}"
                            return {"ok": False, "error": contextual_error, "model": model}
                        except Exception as unexpected_error:
                            logger.error("Variation %s failed unexpectedly: %s", variation_index, unexpected_error)
                            contextual_error = f"[take {variation_index}/{variation_total}] Unexpected error: {unexpected_error}"
                            return {"ok": False, "error": contextual_error, "model": model}
                if not job_results:
                    message = "Video generation did not return any results."
                    await self.emit_status(message, True, True, emitter=__event_emitter__)
                    return {"ok": False, "error": message, "model": model}
                embed_html = "\n".join(result["embed_html"] for result in job_results if result.get("embed_html"))
                timestamp = int(time.time())
                hidden_blob = "".join(
                    self._encode_hidden_artifact(
                        {
                            "video_id": result["job_id"],
                            "model": result["model"],
                            "size": result["size_value"],
                            "seconds": result["seconds_bucket"],
                            "prompt": prompt,
                            "created_at": timestamp,
                            "variation_index": result["variation_index"],
                            "variation_total": result["variation_total"],
                        }
                    )
                    for result in job_results
                )
                chat_message_with_artifacts = hidden_blob
                primary = job_results[0]
                result_payload = {
                    "ok": True,
                    "model": model,
                    "usage": last_completed_data if isinstance(last_completed_data, dict) else None,
                    "video_url": primary["video_url"],
                    "prompt": prompt,
                    "seconds_bucket": primary["seconds_bucket"],
                    "size_value": primary["size_value"],
                    "chat_message": chat_message_with_artifacts,
                    "embed_html": embed_html,
                    "videos": job_results,
                }
                logger.info(
                    "handle_request succeeded: %s",
                    {k: result_payload[k] for k in ("model", "video_url", "seconds_bucket", "size_value")},
                )
                return result_payload
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                error_status = await self.emit_status(
                    "An error occurred while processing request",
                    True,
                    True,
                    emitter=__event_emitter__,
                )
                return {
                    "ok": False,
                    "error": f"{error_status}Error processing request: {str(e)}",
                    "model": self.valves.MODEL,
                }
        is_stream = bool(body.get("stream", False))
        if not is_stream:
            result = await handle_request()
            if result.get("ok"):
                if result.get("help"):
                    help_markdown = result.get("help_markdown", "") or "_No help text available._"
                    return PlainTextResponse(help_markdown)
                embed_html = result.get("embed_html")
                if embed_html:
                    return HTMLResponse(
                        content=embed_html,
                        headers={"Content-Disposition": "inline"},
                    )
                chat_message = result.get("chat_message") or "Your video has been generated and embedded above."
                return PlainTextResponse(chat_message)
            error_message = result.get("error", "An unknown error occurred.")
            return PlainTextResponse(error_message, status_code=400)

        async def stream_response():
            """Server-sent events generator for streaming chat responses."""
            task = asyncio.create_task(handle_request())
            priming_model = self.valves.MODEL
            acceptance_message = "Job accepted. Working on your video..."
            try:
                if __event_emitter__:
                    yield self._format_data(
                        is_stream=True,
                        model=priming_model,
                        content=None,
                        finish_reason=None,
                    )
                else:
                    yield self._format_data(
                        is_stream=True,
                        model=priming_model,
                        content=acceptance_message,
                        finish_reason=None,
                    )
                result = await task
                if result.get("ok"):
                    model_used = result.get("model") or priming_model
                    if result.get("help"):
                        help_markdown = result.get("help_markdown", "") or "_No help text available._"
                        yield self._format_data(
                            is_stream=True,
                            model=model_used,
                            content=help_markdown,
                            finish_reason=None,
                        )
                        yield self._format_data(
                            is_stream=True,
                            model=model_used,
                            content=None,
                            finish_reason="stop",
                        )
                        yield "data: [DONE]\n\n"
                        return
                    chat_message = result.get("chat_message") or "Your video has been generated and embedded above."
                    logger.info("Returning video confirmation for url=%s", result.get("video_url"))
                    yield self._format_data(
                        is_stream=True,
                        model=model_used,
                        content=chat_message,
                        finish_reason=None,
                    )
                    yield self._format_data(
                        is_stream=True,
                        model=model_used,
                        content=None,
                        usage=result.get("usage"),
                        finish_reason="stop",
                    )
                    yield "data: [DONE]\n\n"
                    return
                error_message = result.get("error", "An unknown error occurred.")
                model_used = result.get("model") or priming_model
                logger.info("Returning error payload for model=%s", model_used)
                yield self._format_data(
                    is_stream=True,
                    model=model_used,
                    content=error_message,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"stream_response encountered error: {exc}")
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                fallback_model = self.valves.MODEL
                fallback_message = f"Error emitting response: {exc}"
                yield self._format_data(
                    is_stream=True,
                    model=fallback_model,
                    content=fallback_message,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")
