from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .pricing import get_context_window, get_price
from .tokenizer import count_tokens


STATE_SCHEMA_VERSION = 1
DEFAULT_STATE_FILE = Path.home() / ".agentvitals" / "codex_state.json"
DEFAULT_EVENTS_FILE = Path.home() / ".agentvitals" / "codex_events.jsonl"

_INPUT_TOKEN_KEYS = {
    "input_tokens",
    "prompt_tokens",
    "prompt_token_count",
    "input_token_count",
    "num_input_tokens",
    "in_tokens",
    "tokens_in",
}
_OUTPUT_TOKEN_KEYS = {
    "output_tokens",
    "completion_tokens",
    "completion_token_count",
    "num_output_tokens",
    "out_tokens",
    "tokens_out",
}
_TOTAL_TOKEN_KEYS = {
    "total_tokens",
    "token_count",
    "num_tokens",
}


def default_state_file() -> Path:
    path = os.getenv("AGENT_VITALS_STATE_FILE")
    return Path(path).expanduser() if path else DEFAULT_STATE_FILE


def default_events_file() -> Path:
    path = os.getenv("AGENT_VITALS_EVENTS_FILE")
    return Path(path).expanduser() if path else DEFAULT_EVENTS_FILE


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_codex_payload_parts(payload_parts: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Parse Codex notifier payload from arg parts."""
    if not payload_parts:
        return None

    joined = " ".join(payload_parts).strip()
    parsed = _try_parse_json_object(joined)
    if parsed is not None:
        return parsed

    result: Dict[str, Any] = {}
    loose_parts: list[str] = []
    for part in payload_parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            if key:
                result[key] = value.strip()
            continue
        loose_parts.append(part)

    if loose_parts and len(loose_parts) % 2 == 0:
        for index in range(0, len(loose_parts), 2):
            key = loose_parts[index].strip()
            if key:
                result[key] = loose_parts[index + 1]
    elif loose_parts and not result:
        result["raw_payload"] = joined

    return result or None


def parse_stdin_json(payload_text: str) -> Optional[Dict[str, Any]]:
    return _try_parse_json_object(payload_text)


def extract_codex_event(payload: Dict[str, Any]) -> str:
    for key in (
        "type",
        "event",
        "hook_event_name",
        "hookEventName",
        "event_name",
        "eventName",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    legacy_keys = ("thread-id", "turn-id", "input-messages", "last-assistant-message")
    if any(key in payload for key in legacy_keys):
        return "after_agent"
    return ""


def payload_event_matches(hook_event: str, target_event: str) -> bool:
    actual = _normalize_event_name(hook_event)
    target = _normalize_event_name(target_event)
    if actual == target:
        return True

    aliases = {
        "after-agent": {"after-agent", "agent-turn-complete"},
        "agent-turn-complete": {"after-agent", "agent-turn-complete"},
        "after_agent": {"after_agent", "after-agent", "agent-turn-complete"},
    }
    return target in aliases.get(actual, {actual})


def extract_text_value(payload: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def extract_turn_id(payload: Dict[str, Any]) -> str:
    return extract_text_value(payload, ("turn-id", "turn_id", "turnId"))


def extract_model_name(payload: Dict[str, Any]) -> str:
    return extract_text_value(payload, ("model", "model_name", "modelName"))


def extract_token_counts(payload: Dict[str, Any], encoding_name: str) -> Dict[str, int]:
    input_tokens = _find_numeric_token_value(payload, _INPUT_TOKEN_KEYS)
    output_tokens = _find_numeric_token_value(payload, _OUTPUT_TOKEN_KEYS)
    total_tokens = _find_numeric_token_value(payload, _TOTAL_TOKEN_KEYS)

    input_text = extract_text_value(
        payload,
        (
            "input-messages",
            "input_messages",
            "prompt",
            "user_prompt",
            "input",
        ),
    )
    output_text = extract_text_value(
        payload,
        (
            "last-assistant-message",
            "last_assistant_message",
            "response",
            "assistant_response",
            "output",
        ),
    )

    if input_tokens is None and input_text:
        input_tokens = count_tokens(input_text, encoding_name)
    if output_tokens is None and output_text:
        output_tokens = count_tokens(output_text, encoding_name)

    if total_tokens is None:
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
    elif input_tokens is None and output_tokens is not None:
        input_tokens = max(total_tokens - output_tokens, 0)
    elif output_tokens is None and input_tokens is not None:
        output_tokens = max(total_tokens - input_tokens, 0)
    elif input_tokens is None and output_tokens is None:
        output_tokens = total_tokens
        input_tokens = 0

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def load_state(state_file: Path) -> Optional[Dict[str, Any]]:
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        return data
    return None


def write_state(state_file: Path, state: Dict[str, Any]) -> None:
    ensure_parent(state_file)
    tmp_file = state_file.with_suffix(f"{state_file.suffix}.tmp")
    tmp_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_file, state_file)


def append_event(events_file: Path, event: Dict[str, Any]) -> None:
    ensure_parent(events_file)
    with events_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True))
        handle.write("\n")


def reset_files(state_file: Path, events_file: Path) -> None:
    for path in (state_file, events_file):
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def build_initial_state(
    *,
    source: str,
    model_name: str,
    encoding_name: str,
    context_window_tokens: Optional[int],
    started_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "source": source,
        "model_name": model_name,
        "encoding_name": encoding_name,
        "context_window_tokens": context_window_tokens,
        "start_time": started_at,
        "last_updated": started_at,
        "turn_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "last_turn_id": "",
        "last_event": "",
    }


def update_codex_state(
    *,
    payload: Dict[str, Any],
    state_file: Path,
    events_file: Path,
    model_name_override: Optional[str],
    context_window_override: Optional[int],
    encoding_name: str,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    loaded = load_state(state_file)

    payload_model_name = extract_model_name(payload)
    initial_model_name = (
        model_name_override
        or payload_model_name
        or (loaded or {}).get("model_name")
        or "codex"
    )
    model_name = str(initial_model_name)

    inferred_context = get_context_window(model_name)
    if loaded:
        base_context = loaded.get("context_window_tokens")
    else:
        base_context = inferred_context
    if context_window_override is not None:
        base_context = context_window_override

    if loaded is None:
        state = build_initial_state(
            source="codex",
            model_name=model_name,
            encoding_name=encoding_name,
            context_window_tokens=base_context,
            started_at=now,
        )
    else:
        state = loaded
        state["schema_version"] = STATE_SCHEMA_VERSION
        state["source"] = "codex"
        state["model_name"] = model_name
        state["encoding_name"] = encoding_name
        state["context_window_tokens"] = base_context

    hook_event = extract_codex_event(payload)
    turn_id = extract_turn_id(payload)
    token_counts = extract_token_counts(payload, encoding_name)

    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    state["input_tokens"] = int(state.get("input_tokens", 0)) + token_counts["input_tokens"]
    state["output_tokens"] = int(state.get("output_tokens", 0)) + token_counts["output_tokens"]
    state["total_tokens"] = int(state.get("total_tokens", 0)) + token_counts["total_tokens"]
    state["last_updated"] = now
    state["last_turn_id"] = turn_id
    state["last_event"] = hook_event

    input_cost = (state["input_tokens"] / 1_000_000) * get_price(model_name, "input")
    output_cost = (state["output_tokens"] / 1_000_000) * get_price(model_name, "output")
    state["estimated_cost_usd"] = round(input_cost + output_cost, 6)

    write_state(state_file, state)

    event = {
        "timestamp": now,
        "event": hook_event,
        "turn_id": turn_id,
        "model_name": model_name,
        "input_tokens": token_counts["input_tokens"],
        "output_tokens": token_counts["output_tokens"],
        "total_tokens": token_counts["total_tokens"],
    }
    append_event(events_file, event)
    return state


def compute_state_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    model_name = str(state.get("model_name", "codex"))
    input_tokens = int(state.get("input_tokens", 0))
    output_tokens = int(state.get("output_tokens", 0))
    total_tokens = int(state.get("total_tokens", 0))

    context_window_tokens = state.get("context_window_tokens")
    if context_window_tokens is None:
        context_window_tokens = get_context_window(model_name)
    if isinstance(context_window_tokens, (int, float)):
        context_window_tokens = int(context_window_tokens)
    else:
        context_window_tokens = None

    start_time_raw = state.get("start_time")
    now = datetime.now(timezone.utc)
    if isinstance(start_time_raw, str):
        try:
            started = datetime.fromisoformat(start_time_raw)
        except ValueError:
            started = now
    else:
        started = now
    duration_seconds = max((now - started).total_seconds(), 0.0)
    tokens_per_minute = (
        (total_tokens / duration_seconds) * 60 if duration_seconds > 0 else 0.0
    )

    input_cost = (input_tokens / 1_000_000) * get_price(model_name, "input")
    output_cost = (output_tokens / 1_000_000) * get_price(model_name, "output")
    estimated_cost = input_cost + output_cost

    if context_window_tokens is None:
        context_remaining_tokens = None
        context_utilization_pct = None
        is_context_window_exceeded = None
    else:
        context_remaining_tokens = max(context_window_tokens - total_tokens, 0)
        context_utilization_pct = (total_tokens / context_window_tokens) * 100
        is_context_window_exceeded = total_tokens > context_window_tokens

    return {
        "source": state.get("source", "codex"),
        "model_name": model_name,
        "start_time": started.isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "interaction_count": int(state.get("turn_count", 0)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": f"${estimated_cost:.6f}",
        "average_agent_response_time_seconds": 0.0,
        "tokens_per_minute": round(tokens_per_minute, 2),
        "context_window_tokens": context_window_tokens,
        "context_remaining_tokens": context_remaining_tokens,
        "context_utilization_pct": (
            round(context_utilization_pct, 2)
            if context_utilization_pct is not None
            else None
        ),
        "is_context_window_exceeded": is_context_window_exceeded,
        "last_event": state.get("last_event", ""),
        "last_turn_id": state.get("last_turn_id", ""),
        "last_updated": state.get("last_updated", ""),
    }


def _try_parse_json_object(payload_text: str) -> Optional[Dict[str, Any]]:
    text = payload_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_event_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _normalize_key(name: str) -> str:
    normalized = name.strip().lower()
    for token in ("-", ".", " "):
        normalized = normalized.replace(token, "_")
    return normalized


def _iter_nested_items(value: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, inner in value.items():
            yield _normalize_key(str(key)), inner
            yield from _iter_nested_items(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _iter_nested_items(inner)


def _find_numeric_token_value(payload: Dict[str, Any], key_names: set[str]) -> Optional[int]:
    for key, value in _iter_nested_items(payload):
        if key not in key_names:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
    return None
