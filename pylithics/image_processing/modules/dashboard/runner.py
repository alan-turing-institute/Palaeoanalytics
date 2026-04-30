"""
Launch helper for the PyLithics Streamlit dashboard.

Spawns ``streamlit run`` in headless mode (so its built-in startup banner
is suppressed), waits until the server is listening, then prints our own
launch message and opens the user's default browser.
"""

import itertools
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
_SPINNER_FRAMES = ("|", "/", "-", "\\")


def launch_dashboard(data_dir: str, port: int = DEFAULT_PORT) -> int:
    """
    Start the PyLithics Streamlit dashboard pointed at ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Path containing a ``processed/processed_metrics.csv``.
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
    env[DATA_DIR_ENV] = str(Path(data_dir).resolve())

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
        "--client.toolbarMode", "minimal",
    ]

    log_path = (
        Path(data_dir).resolve() / "processed" / "dashboard.log"
    )
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

    Shows an animated spinner with elapsed seconds while polling on a TTY,
    so the user can see the wait is progressing rather than wonder if the
    process has hung. Stays silent on non-TTY stdout (the upfront banner
    in ``launch_dashboard`` already announces the wait).
    """
    is_tty = sys.stdout.isatty()
    spinner = itertools.cycle(_SPINNER_FRAMES)
    deadline = time.time() + _STARTUP_TIMEOUT_SECONDS
    started = time.time()

    while time.time() < deadline:
        if proc.poll() is not None:
            if is_tty:
                _clear_spinner_line()
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                if is_tty:
                    _clear_spinner_line()
                return True
        except OSError:
            if is_tty:
                elapsed = time.time() - started
                sys.stdout.write(
                    f"\r{next(spinner)} Preparing dashboard… ({elapsed:4.1f}s) "
                )
                sys.stdout.flush()
            time.sleep(_POLL_INTERVAL_SECONDS)

    if is_tty:
        _clear_spinner_line()
    return False


def _clear_spinner_line() -> None:
    """Erase the spinner line so the next message starts clean."""
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
