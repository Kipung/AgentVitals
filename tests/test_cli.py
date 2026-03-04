import json
import sys
from typing import Optional

import pytest

from agent_vitals import cli


def _invoke_cli(args) -> int:
    original_argv = sys.argv
    try:
        sys.argv = ["agent-vitals", *args]
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        return int(exc_info.value.code)
    finally:
        sys.argv = original_argv


def test_codex_hook_and_watch_once_flow(tmp_path, capsys) -> None:
    state_file = tmp_path / "codex_state.json"
    events_file = tmp_path / "codex_events.jsonl"

    payload = {
        "type": "agent-turn-complete",
        "turn-id": "turn-101",
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    hook_code = _invoke_cli(
        [
            "codex-hook",
            "--model",
            "gpt-4-turbo",
            "--state-file",
            str(state_file),
            "--events-file",
            str(events_file),
            json.dumps(payload),
        ]
    )
    assert hook_code == 0
    assert state_file.exists()
    assert events_file.exists()

    watch_code = _invoke_cli(
        [
            "watch",
            "--once",
            "--state-file",
            str(state_file),
            "--events-file",
            str(events_file),
        ]
    )
    assert watch_code == 0
    output = capsys.readouterr().out
    assert "Total Tokens:" in output
    assert "30" in output
    assert "turn-101" in output


def test_legacy_mode_dispatches_without_subcommand(monkeypatch) -> None:
    captured = {}

    def fake_run(
        command, model_name: str, context_window_tokens: Optional[int]
    ) -> None:
        captured["command"] = command
        captured["model_name"] = model_name
        captured["context_window_tokens"] = context_window_tokens

    monkeypatch.setattr(cli, "run_and_track_live", fake_run)
    code = _invoke_cli(
        ["--model", "gpt-4-turbo", "python", "-c", "print('hello from legacy')"]
    )
    assert code == 0
    assert captured["command"][0] == "python"
    assert captured["model_name"] == "gpt-4-turbo"
    assert captured["context_window_tokens"] is None


def test_reset_command_removes_state_files(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    events_file = tmp_path / "events.jsonl"
    state_file.write_text("{}", encoding="utf-8")
    events_file.write_text("{}", encoding="utf-8")

    code = _invoke_cli(
        [
            "reset",
            "--state-file",
            str(state_file),
            "--events-file",
            str(events_file),
        ]
    )
    assert code == 0
    assert not state_file.exists()
    assert not events_file.exists()
