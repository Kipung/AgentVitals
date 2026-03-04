from __future__ import annotations

import argparse
import os
import pty
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .hook_store import (
    compute_state_metrics,
    default_events_file,
    default_state_file,
    extract_codex_event,
    load_state,
    parse_codex_payload_parts,
    parse_stdin_json,
    payload_event_matches,
    reset_files,
    update_codex_state,
)
from .tracker import VitalsTracker

console = Console()


def _build_context_bar(context_pct: Optional[float], *, width: int = 18) -> str:
    if context_pct is None:
        return "N/A"
    bounded_pct = max(0.0, min(context_pct, 100.0))
    filled = int((bounded_pct / 100) * width)
    bar = ("#" * filled) + ("-" * (width - filled))
    if context_pct < 70:
        color = "green"
    elif context_pct < 90:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{bar}[/{color}] {context_pct:.2f}%"


def build_vitals_panel(summary: Dict[str, Any], *, title: str = "AgentVitals") -> Panel:
    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right", style="bold green")

    table.add_row("Model:", str(summary.get("model_name", "unknown")))
    table.add_row("Input Tokens:", f"{int(summary.get('input_tokens', 0)):,}")
    table.add_row("Output Tokens:", f"{int(summary.get('output_tokens', 0)):,}")
    table.add_row("Total Tokens:", f"{int(summary.get('total_tokens', 0)):,}")
    table.add_row("Est. Cost:", str(summary.get("estimated_cost_usd", "$0.000000")))

    context_window = summary.get("context_window_tokens")
    context_remaining = summary.get("context_remaining_tokens")
    if context_window is None:
        context_text = "Unknown (use --context-window)"
    else:
        overflow = " (exceeded)" if summary.get("is_context_window_exceeded") else ""
        context_text = (
            f"{int(context_remaining or 0):,} left / {int(context_window):,} total{overflow}"
        )
    table.add_row("Context:", context_text)
    table.add_row(
        "Context Usage:",
        _build_context_bar(summary.get("context_utilization_pct")),
    )

    table.add_row("Token Rate:", f"{float(summary.get('tokens_per_minute', 0.0)):.2f} tok/min")
    table.add_row(
        "Avg. Resp Time:",
        f"{float(summary.get('average_agent_response_time_seconds', 0.0)):.2f}s",
    )
    table.add_row("Duration:", f"{float(summary.get('duration_seconds', 0.0)):.2f}s")

    if "interaction_count" in summary:
        table.add_row("Turns:", str(int(summary.get("interaction_count", 0))))
    if summary.get("last_event"):
        table.add_row("Last Event:", str(summary["last_event"]))
    if summary.get("last_turn_id"):
        table.add_row("Last Turn:", str(summary["last_turn_id"]))

    return Panel(table, title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan")


def generate_tracker_panel(tracker: VitalsTracker) -> Panel:
    summary = tracker.get_summary()
    return build_vitals_panel(summary)


def _build_waiting_panel(state_file: Path, events_file: Path) -> Panel:
    content = (
        "Waiting for hook events...\n\n"
        "Start Codex with hook configured to call:\n"
        "  agent-vitals codex-hook\n\n"
        f"State file: {state_file}\n"
        f"Events file: {events_file}"
    )
    return Panel(content, title="[bold cyan]AgentVitals[/bold cyan]", border_style="cyan")


def run_and_track_live(
    command: list[str], model_name: str, context_window_tokens: Optional[int]
) -> None:
    """
    Execute a command in a pseudo-terminal with a live vitals TUI.
    """
    tracker = VitalsTracker(
        model_name=model_name, context_window_tokens=context_window_tokens
    )
    layout = Layout()
    layout.split_column(Layout(name="main"), Layout(size=13, name="footer"))

    pid, master_fd = pty.fork()

    if pid == pty.CHILD:
        try:
            os.execvp(command[0], command)
        except FileNotFoundError:
            print(f"Error: Command not found: {command[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        if not sys.stdin.isatty():
            console.print(
                "[bold red]Error:[/bold red] agent-vitals requires an interactive TTY."
            )
            os.close(master_fd)
            os.waitpid(pid, 0)
            return

        old_tty_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())

            with Live(layout, screen=True, redirect_stderr=False, transient=True):
                read_fds = [master_fd, sys.stdin]
                last_update_time = 0.0

                while True:
                    if time.time() - last_update_time > 0.1:
                        layout["footer"].update(generate_tracker_panel(tracker))
                        last_update_time = time.time()

                    ready_fds, _, _ = select.select(read_fds, [], [], 0.1)
                    if not ready_fds:
                        continue

                    if master_fd in ready_fds:
                        try:
                            data = os.read(master_fd, 1024)
                        except OSError:
                            break
                        if not data:
                            break
                        text = data.decode(errors="ignore")
                        console.print(text, end="")
                        tracker.add_interaction(source="agent", text=text)

                    if sys.stdin in ready_fds:
                        data = os.read(sys.stdin.fileno(), 1024)
                        if not data:
                            break
                        os.write(master_fd, data)
                        tracker.add_interaction(
                            source="user", text=data.decode(errors="ignore")
                        )
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)
            os.close(master_fd)
            os.waitpid(pid, 0)

            console.print("\n" + "-" * 50)
            console.print("⚡️ [bold green]AgentVitals Final Summary[/bold green]:")
            console.print(generate_tracker_panel(tracker))


def watch_state_live(
    *,
    state_file: Path,
    events_file: Path,
    refresh_seconds: float,
    once: bool,
) -> int:
    if once:
        state = load_state(state_file)
        if state is None:
            console.print(_build_waiting_panel(state_file, events_file))
            return 0
        console.print(build_vitals_panel(compute_state_metrics(state)))
        return 0

    try:
        with Live(
            _build_waiting_panel(state_file, events_file),
            screen=False,
            refresh_per_second=max(int(1 / max(refresh_seconds, 0.1)), 1),
            transient=False,
        ) as live:
            while True:
                state = load_state(state_file)
                if state is None:
                    panel = _build_waiting_panel(state_file, events_file)
                else:
                    panel = build_vitals_panel(compute_state_metrics(state))
                live.update(panel)
                time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        return 0


def _run_legacy_command_mode(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command interactively with a live vitals dashboard."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_VITALS_MODEL"),
        help=(
            "Model name for pricing/context estimation "
            "(e.g., gpt-4-turbo, claude-3-sonnet)."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Override context window token size used for utilization tracking.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run and track (e.g., python my_agent.py)",
    )
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.print_help()
        return 1
    model_name = args.model if args.model else " ".join(command)
    run_and_track_live(command, model_name, args.context_window)
    return 0


def _parse_path(path_text: str) -> Path:
    return Path(path_text).expanduser()


def _handle_run_command(args: argparse.Namespace) -> int:
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        console.print("[bold red]Error:[/bold red] Missing command to run.")
        return 1
    model_name = args.model if args.model else " ".join(command)
    run_and_track_live(command, model_name, args.context_window)
    return 0


def _handle_watch_command(args: argparse.Namespace) -> int:
    return watch_state_live(
        state_file=_parse_path(args.state_file),
        events_file=_parse_path(args.events_file),
        refresh_seconds=args.refresh,
        once=args.once,
    )


def _handle_codex_hook_command(args: argparse.Namespace) -> int:
    payload = parse_codex_payload_parts(tuple(args.payload_parts))
    if payload is None and not sys.stdin.isatty():
        payload = parse_stdin_json(sys.stdin.read())
    if payload is None:
        if args.verbose:
            console.print(
                "[yellow]Codex hook payload missing/invalid; skipping state update.[/yellow]"
            )
        return 0

    hook_event = extract_codex_event(payload)
    if not hook_event:
        if args.verbose:
            console.print("[yellow]Codex hook payload missing event details.[/yellow]")
        return 0
    if not payload_event_matches(hook_event, args.event):
        return 0

    state = update_codex_state(
        payload=payload,
        state_file=_parse_path(args.state_file),
        events_file=_parse_path(args.events_file),
        model_name_override=args.model,
        context_window_override=args.context_window,
        encoding_name=args.encoding,
    )

    if args.verbose:
        metrics = compute_state_metrics(state)
        console.print(
            "[green]Updated[/green] "
            f"turns={metrics['interaction_count']} "
            f"total_tokens={metrics['total_tokens']:,} "
            f"cost={metrics['estimated_cost_usd']}"
        )
    return 0


def _handle_reset_command(args: argparse.Namespace) -> int:
    state_file = _parse_path(args.state_file)
    events_file = _parse_path(args.events_file)
    reset_files(state_file, events_file)
    console.print(
        f"[green]Reset complete[/green] state={state_file} events={events_file}"
    )
    return 0


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AgentVitals utilities: run wrapped commands, ingest hook events, "
            "and watch live token/context usage in a separate terminal."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    run_parser = subparsers.add_parser(
        "run", help="Run a command interactively with a live vitals dashboard."
    )
    run_parser.add_argument(
        "--model",
        default=os.getenv("AGENT_VITALS_MODEL"),
        help="Model name for pricing/context estimation.",
    )
    run_parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Override context window token size used for utilization tracking.",
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run and track (e.g., -- python my_agent.py).",
    )

    watch_parser = subparsers.add_parser(
        "watch", help="Watch persisted vitals live (run in a separate terminal)."
    )
    watch_parser.add_argument(
        "--state-file",
        default=str(default_state_file()),
        help="Path to persisted state JSON file.",
    )
    watch_parser.add_argument(
        "--events-file",
        default=str(default_events_file()),
        help="Path to events JSONL file (informational in watch mode).",
    )
    watch_parser.add_argument(
        "--refresh",
        type=float,
        default=0.5,
        help="Refresh interval in seconds.",
    )
    watch_parser.add_argument(
        "--once",
        action="store_true",
        help="Render one snapshot and exit.",
    )

    codex_hook_parser = subparsers.add_parser(
        "codex-hook",
        help=(
            "Ingest Codex notifier payload and update rolling vitals state "
            "(for passive tracking)."
        ),
    )
    codex_hook_parser.add_argument(
        "--event",
        default="agent-turn-complete",
        help="Codex event type to process.",
    )
    codex_hook_parser.add_argument(
        "--model",
        default=os.getenv("AGENT_VITALS_MODEL"),
        help="Override model name used for pricing/context window estimates.",
    )
    codex_hook_parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Override context window token size.",
    )
    codex_hook_parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Tokenizer encoding used for fallback text token counting.",
    )
    codex_hook_parser.add_argument(
        "--state-file",
        default=str(default_state_file()),
        help="Path to persisted state JSON file.",
    )
    codex_hook_parser.add_argument(
        "--events-file",
        default=str(default_events_file()),
        help="Path to append-only events JSONL file.",
    )
    codex_hook_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print ingestion updates.",
    )
    codex_hook_parser.add_argument("payload_parts", nargs="*")

    reset_parser = subparsers.add_parser(
        "reset",
        help="Reset persisted vitals files used by hook/watch mode.",
    )
    reset_parser.add_argument(
        "--state-file",
        default=str(default_state_file()),
        help="Path to persisted state JSON file.",
    )
    reset_parser.add_argument(
        "--events-file",
        default=str(default_events_file()),
        help="Path to append-only events JSONL file.",
    )

    return parser


def main() -> None:
    argv = sys.argv[1:]
    subcommands = {"run", "watch", "codex-hook", "reset"}

    if argv and argv[0] not in subcommands and argv[0] not in {"-h", "--help"}:
        raise SystemExit(_run_legacy_command_mode(argv))

    parser = _build_subcommand_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "run":
        raise SystemExit(_handle_run_command(args))
    if args.subcommand == "watch":
        raise SystemExit(_handle_watch_command(args))
    if args.subcommand == "codex-hook":
        raise SystemExit(_handle_codex_hook_command(args))
    if args.subcommand == "reset":
        raise SystemExit(_handle_reset_command(args))

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
