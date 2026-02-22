import argparse
import os
import pty
import select
import subprocess
import sys
import tty
import termios
import time
from .tracker import VitalsTracker

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.console import Console

console = Console()

def generate_vitals_panel(tracker: VitalsTracker) -> Panel:
    """Generates a Rich Panel with the latest vitals."""
    summary = tracker.get_summary()
    cost = tracker.get_cost()

    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right", style="bold green")

    table.add_row("Model:", summary['model_name'])
    table.add_row("Total Tokens:", str(summary['total_tokens']))
    table.add_row("Est. Cost:", f"${cost:.6f}")
    table.add_row("Avg. Resp Time:", f"{summary['average_agent_response_time_seconds']:.2f}s")
    table.add_row("Duration:", f"{summary['duration_seconds']:.2f}s")

    return Panel(table, title="[bold cyan]AgentVitals[/bold cyan]", border_style="cyan")


def run_and_track_live(command: list):
    """
    Executes a command in a pseudo-terminal with a live vitals TUI.
    """
    tracker = VitalsTracker(model_name=" ".join(command))
    layout = Layout()
    layout.split_column(Layout(name="main"), Layout(size=8, name="footer"))

    # Fork a new process
    pid, master_fd = pty.fork()

    if pid == pty.CHILD:
        try:
            os.execvp(command[0], command)
        except FileNotFoundError:
            print(f"Error: Command not found: {command[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        # Parent process
        old_tty_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())

            with Live(layout, screen=True, redirect_stderr=False, transient=True) as live:
                read_fds = [master_fd, sys.stdin]
                last_update_time = 0

                while True:
                    # Update the live display periodically
                    if time.time() - last_update_time > 0.1:
                        layout["footer"].update(generate_vitals_panel(tracker))
                        last_update_time = time.time()

                    # Use select with a timeout
                    ready_fds, _, _ = select.select(read_fds, [], [], 0.1)

                    if not ready_fds:
                        continue # Timeout occurred, loop to update display

                    if master_fd in ready_fds:
                        try:
                            data = os.read(master_fd, 1024)
                        except OSError:
                            break
                        if not data:
                            break
                        
                        console.print(data.decode(errors='ignore'), end='')
                        tracker.add_interaction(source="agent", text=data.decode(errors='ignore'))

                    if sys.stdin in ready_fds:
                        data = os.read(sys.stdin.fileno(), 1024)
                        if not data:
                            break
                        os.write(master_fd, data)
                        tracker.add_interaction(source="user", text=data.decode(errors='ignore'))
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)
            os.close(master_fd)
            os.waitpid(pid, 0)
            
            console.print("\n" + "-" * 50)
            console.print("⚡️ [bold green]AgentVitals Final Summary[/bold green]:")
            final_panel = generate_vitals_panel(tracker)
            console.print(final_panel)

def main():
    parser = argparse.ArgumentParser(
        description="Run a command interactively with a live vitals dashboard."
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run and track (e.g., python my_agent.py)",
    )
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    run_and_track_live(args.command)

if __name__ == "__main__":
    main()
