"""
Launch helper for the PyLithics Streamlit dashboard.

Spawns ``streamlit run`` in headless mode (so its built-in startup banner
is suppressed), waits until the server is listening, then prints our own
launch message and opens the user's default browser.
"""

import logging
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8501
DATA_DIR_ENV = "PYLITHICS_DASHBOARD_DATA_DIR"

_STARTUP_TIMEOUT_SECONDS = 30
_POLL_INTERVAL_SECONDS = 0.2


def launch_dashboard(processed_dir: str, port: int = DEFAULT_PORT) -> int:
    """
    Start the PyLithics Streamlit dashboard pointed at ``processed_dir``.

    Parameters
    ----------
    processed_dir : str
        Path to the folder containing ``processed_metrics.csv`` and the
        accompanying labeled / Voronoi PNGs and per-lithic JSON files.
    port : int
        TCP port for Streamlit to bind. Defaults to 8501.

    Returns
    -------
    int
        Process return code. ``0`` on graceful shutdown.
    """
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():  # pragma: no cover — install bug
        logging.error("Dashboard entrypoint missing: %s", app_path)
        return 1

    env = os.environ.copy()
    env[DATA_DIR_ENV] = str(Path(processed_dir).resolve())

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
        "--client.toolbarMode", "minimal",
    ]

    log_path = Path(processed_dir).resolve() / "dashboard.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"http://localhost:{port}"

    print("\nStarting PyLithics data explorer...", flush=True)
    print(f"URL: {url}", flush=True)
    print(" loading modules and PyLithics data",
          flush=True)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )

        try:
            if not _wait_for_port("localhost", port, proc):
                logging.error(
                    "Dashboard did not start within %ds. "
                    "Check %s for details.",
                    _STARTUP_TIMEOUT_SECONDS, log_path,
                )
                proc.terminate()
                return 1

            print(f"Dashboard ready — opening {url} in your browser.",
                  flush=True)
            print("Press Ctrl+C in this terminal to stop.\n", flush=True)
            webbrowser.open(url)

            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return 0


def _wait_for_port(
    host: str, port: int, proc: subprocess.Popen,
) -> bool:
    """Return True once the server accepts TCP connections on (host, port).

    Shows a rich spinner with elapsed seconds on a TTY; falls back to
    silent polling on non-TTY stdout (the upfront banner in
    ``launch_dashboard`` already announces the wait).
    """
    from rich.console import Console

    console = Console()
    if not console.is_terminal:
        return _poll_until_ready(host, port, proc)

    with console.status(
        "[cyan]Preparing dashboard...", spinner="dots"
    ) as status:
        started = time.time()
        deadline = started + _STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                elapsed = time.time() - started
                status.update(
                    f"[cyan]Preparing dashboard...[/] "
                    f"({elapsed:4.1f}s)"
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
        return False


def _poll_until_ready(
    host: str, port: int, proc: subprocess.Popen,
) -> bool:
    """Silent polling loop for non-TTY stdout."""
    deadline = time.time() + _STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(_POLL_INTERVAL_SECONDS)
    return False
