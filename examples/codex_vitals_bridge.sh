#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AGENT_VITALS_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
AGENT_VITALS_BIN="${AGENT_VITALS_BIN:-$(command -v agent-vitals || true)}"

if [[ -n "$AGENT_VITALS_BIN" ]]; then
  exec "$AGENT_VITALS_BIN" codex-hook "$@"
fi

if [[ -z "${AGENT_VITALS_PYTHON:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    AGENT_VITALS_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    AGENT_VITALS_PYTHON="$(command -v python3)"
  else
    AGENT_VITALS_PYTHON="$(command -v python)"
  fi
fi

exec env PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$AGENT_VITALS_PYTHON" -m agent_vitals.cli codex-hook "$@"
