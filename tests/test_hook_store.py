import json

from agent_vitals.hook_store import (
    compute_state_metrics,
    extract_token_counts,
    load_state,
    parse_codex_payload_parts,
    payload_event_matches,
    update_codex_state,
)


def test_parse_codex_payload_parts_supports_json() -> None:
    payload = {"type": "agent-turn-complete", "turn-id": "turn-1"}
    parsed = parse_codex_payload_parts((json.dumps(payload),))
    assert parsed == payload


def test_parse_codex_payload_parts_supports_key_value_parts() -> None:
    parsed = parse_codex_payload_parts(("type", "agent-turn-complete", "turn-id", "t2"))
    assert parsed == {"type": "agent-turn-complete", "turn-id": "t2"}


def test_payload_event_matches_aliases() -> None:
    assert payload_event_matches("agent-turn-complete", "after_agent")
    assert payload_event_matches("after-agent", "agent-turn-complete")


def test_extract_token_counts_prefers_usage_fields() -> None:
    payload = {"usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}
    counts = extract_token_counts(payload, "cl100k_base")
    assert counts == {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}


def test_extract_token_counts_falls_back_to_text() -> None:
    payload = {
        "input-messages": "Please run tests.",
        "last-assistant-message": "I ran tests and they passed.",
    }
    counts = extract_token_counts(payload, "cl100k_base")
    assert counts["input_tokens"] > 0
    assert counts["output_tokens"] > 0
    assert counts["total_tokens"] == counts["input_tokens"] + counts["output_tokens"]


def test_update_codex_state_persists_and_accumulates(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    events_file = tmp_path / "events.jsonl"

    payload_1 = {
        "type": "agent-turn-complete",
        "turn-id": "turn-1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    state_1 = update_codex_state(
        payload=payload_1,
        state_file=state_file,
        events_file=events_file,
        model_name_override="gpt-4-turbo",
        context_window_override=None,
        encoding_name="cl100k_base",
    )
    assert state_1["turn_count"] == 1
    assert state_1["input_tokens"] == 10
    assert state_1["output_tokens"] == 5
    assert state_1["total_tokens"] == 15

    payload_2 = {
        "type": "agent-turn-complete",
        "turn-id": "turn-2",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    state_2 = update_codex_state(
        payload=payload_2,
        state_file=state_file,
        events_file=events_file,
        model_name_override="gpt-4-turbo",
        context_window_override=None,
        encoding_name="cl100k_base",
    )
    assert state_2["turn_count"] == 2
    assert state_2["input_tokens"] == 12
    assert state_2["output_tokens"] == 8
    assert state_2["total_tokens"] == 20

    loaded = load_state(state_file)
    assert loaded is not None
    metrics = compute_state_metrics(loaded)
    assert metrics["interaction_count"] == 2
    assert metrics["context_window_tokens"] == 128_000
    assert metrics["context_remaining_tokens"] == 127_980
    assert metrics["is_context_window_exceeded"] is False

    events = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 2
