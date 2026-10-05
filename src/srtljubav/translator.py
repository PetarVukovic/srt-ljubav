"""Gemini translator wrapper: per-response token-usage tracking + model caching.

``gemini_srt_translator`` creates its genai client on demand in
``_get_client()``; we wrap the returned client so every
``generate_content(_stream)`` response's ``usage_metadata`` is recorded and
transient failures (5xx, network) are re-sent before gst counts them.
Caches are module-level and per-process (workers each get their own).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import httpx
from gemini_srt_translator.main import GeminiSRTTranslator
from google.genai import errors as genai_errors

_MODELS_CACHE: dict[str, list[str]] = {}
_TOKEN_LIMIT_CACHE: dict[tuple[str, str], int] = {}
_MAX_SOURCE_ECHO_STREAK = 3
# Waits between re-sends of ONE request that hit a transient failure (~7.8 min
# in total). gst counts every exception toward its consecutive-error ceiling
# BEFORE its own 503 pause, so three 503s in a row used to abandon a language.
TRANSIENT_RETRY_DELAYS: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0, 240.0)

_STREAM_END = object()


def is_transient_error(exc: BaseException) -> bool:
    """True for failures a re-send can cure: 5xx answers and network faults.

    429 / quota errors are deliberately NOT transient here -- gst reacts to
    them by switching to the second API key, which a silent retry would hide.
    """
    if isinstance(exc, genai_errors.ServerError):
        return True
    return isinstance(exc, (httpx.TransportError, OSError))


def call_with_transient_retry[T](
    call: Callable[[], T],
    delays: tuple[float, ...] = TRANSIENT_RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run *call*, re-sending it after each delay while it fails transiently."""
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return call()
        except Exception as exc:
            if delay is None or not is_transient_error(exc):
                raise
            print(
                f"Transient Gemini failure (attempt {attempt}/{len(delays) + 1}), "
                f"retrying in {delay:.0f}s: {type(exc).__name__}: {str(exc)[:160]}",
                file=sys.stderr,
            )
            sleep(delay)
    raise AssertionError("unreachable: the last attempt either returns or raises")


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_nontrivial(text: str) -> bool:
    return len(text) >= 12 and sum(char.isalpha() for char in text) >= 8


def validate_translated_lines(
    translated_lines: list[Any], batch: list[Any]
) -> list[dict[str, Any]]:
    """Return one valid translated object per source index or reject the batch.

    ``gemini-srt-translator`` starts its output as a copy of the Croatian source.
    Filling missing model objects from that source therefore hides skipped
    translation blocks. Raising here makes gst retry only the current batch;
    after gst exhausts its retries, the outer pipeline switches to a smaller
    batch size.
    """
    expected_items = {str(item["index"]): item for item in batch}
    returned: dict[str, str] = {}
    malformed = unexpected = duplicated = empty = 0

    for line in translated_lines:
        if not isinstance(line, dict) or "index" not in line or "text" not in line:
            malformed += 1
            continue
        index = str(line["index"])
        if index not in expected_items:
            unexpected += 1
            continue
        if index in returned:
            duplicated += 1
            continue
        text = line.get("text")
        source_text = str(expected_items[index]["text"])
        if not isinstance(text, str) or (source_text.strip() and not text.strip()):
            empty += 1
            continue
        returned[index] = text

    missing = len(expected_items) - len(returned)
    if malformed or unexpected or duplicated or empty or missing:
        raise ValueError(
            "Batch coverage gap: "
            f"{missing} missing, {duplicated} duplicated, {empty} empty, "
            f"{malformed} malformed and {unexpected} unexpected line(s)."
        )

    ordered = [
        {"index": item["index"], "text": returned[str(item["index"])]}
        for item in batch
    ]

    streak = 0
    for item, translated in zip(batch, ordered, strict=True):
        source_norm = _normalized(str(item["text"]))
        translated_norm = _normalized(str(translated["text"]))
        if _is_nontrivial(source_norm) and source_norm == translated_norm:
            streak += 1
            if streak >= _MAX_SOURCE_ECHO_STREAK:
                raise ValueError(
                    f"Batch content gap: {streak} consecutive nontrivial lines "
                    "were left identical to the Croatian source."
                )
        else:
            streak = 0

    return ordered


class UsageTrackingModelsProxy:
    """Records usage per response and absorbs transient failures before gst sees them."""

    def __init__(self, models: Any, translator: CachedGeminiSRTTranslator):
        self._models = models
        self._translator = translator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        response = call_with_transient_retry(
            lambda: self._models.generate_content(*args, **kwargs),
            delays=self._translator.transient_retry_delays,
        )
        self._translator.record_usage_from_response(response)
        return response

    def generate_content_stream(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        # Only the opening of the stream is retried: once chunks have reached
        # gst, a re-send would duplicate text it has already consumed.
        def open_stream() -> tuple[Iterator[Any], Any]:
            chunks = iter(self._models.generate_content_stream(*args, **kwargs))
            return chunks, next(chunks, _STREAM_END)

        chunks, first = call_with_transient_retry(
            open_stream, delays=self._translator.transient_retry_delays
        )
        if first is _STREAM_END:
            return
        self._translator.record_usage_from_response(first)
        yield first
        for chunk in chunks:
            self._translator.record_usage_from_response(chunk)
            yield chunk


class UsageTrackingClientProxy:
    def __init__(self, client: Any, translator: CachedGeminiSRTTranslator):
        self._client = client
        self.models = UsageTrackingModelsProxy(client.models, translator)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class CachedGeminiSRTTranslator(GeminiSRTTranslator):
    transient_retry_delays: tuple[float, ...] = TRANSIENT_RETRY_DELAYS

    # Any: gst's constructor takes ~30 loosely typed keyword arguments.
    def __init__(self, *args: Any, max_consecutive_errors: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if max_consecutive_errors is not None:
            # gst hardcodes 3 in __init__ and exposes no parameter for it.
            self.max_consecutive_errors = max_consecutive_errors
        self.usage_records: list[dict[str, Any]] = []
        self._seen_usage_records: set[tuple[int, int, int, int]] = set()

    def _get_client(self):
        return UsageTrackingClientProxy(super()._get_client(), self)

    def _process_translated_lines(
        self,
        translated_lines: list[Any],
        translated_subtitle: list[Any],
        batch: list[Any],
        finished: bool,
    ) -> Any:
        if finished:
            translated_lines = validate_translated_lines(translated_lines, batch)
        return super()._process_translated_lines(
            translated_lines=translated_lines,
            translated_subtitle=translated_subtitle,
            batch=batch,
            finished=finished,
        )

    def getmodels(self):
        cache_key = self.current_api_key or ""
        if cache_key not in _MODELS_CACHE:
            _MODELS_CACHE[cache_key] = super().getmodels()
        return _MODELS_CACHE[cache_key]

    def _get_token_limit(self):
        cache_key = (self.current_api_key or "", self.model_name)
        if cache_key not in _TOKEN_LIMIT_CACHE:
            super()._get_token_limit()
            _TOKEN_LIMIT_CACHE[cache_key] = self.token_limit
        else:
            self.token_limit = _TOKEN_LIMIT_CACHE[cache_key]

    def record_usage_from_response(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return

        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        tool_tokens = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
        marker = (prompt_tokens, candidate_tokens, thoughts_tokens, total_tokens)
        if marker in self._seen_usage_records:
            return
        self._seen_usage_records.add(marker)

        self.usage_records.append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": getattr(response, "model_version", None) or self.model_name,
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "thoughts_tokens": thoughts_tokens,
                "tool_use_prompt_tokens": tool_tokens,
                "output_tokens": candidate_tokens + thoughts_tokens,
                "total_tokens": total_tokens,
            }
        )
