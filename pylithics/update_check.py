"""
Tell the user when a newer PyLithics release is on GitHub.

PyLithics is installed from a clone of the repository, not from PyPI,
so the check asks GitHub for the latest release and compares its tag
with the installed version. It runs at most once a day, never blocks
a run, and stays silent on any error, including no network.

Only a clone on the ``main`` branch is told. A developer on another
branch is not nagged about a release they may be preparing. When the
clone is found and the user says yes, the update is ``git pull`` and
``pip install .`` in that clone.

Off switches: ``update_check.enabled: false`` in ``config.yaml``, or
the environment variable ``PYLITHICS_NO_UPDATE_CHECK``.
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from typing import Callable, Optional, Tuple

REPO = 'alan-turing-institute/Palaeoanalytics'
RELEASES_URL = f'https://api.github.com/repos/{REPO}/releases/latest'
RELEASE_BRANCH = 'main'
ENV_OFF = 'PYLITHICS_NO_UPDATE_CHECK'
CHECK_INTERVAL = 24 * 60 * 60
TIMEOUT = 2.0
# Written by setup.py at install time: the absolute path of the clone.
_SOURCE_FILE = os.path.join(os.path.dirname(__file__), '_source_path.txt')


def check_for_update(
    enabled: bool = True,
    say: Callable[[str], None] = print,
    ask: Callable[[str], str] = input,
) -> Optional[str]:
    """
    Tell the user about a newer release, and offer to install it.

    Parameters
    ----------
    enabled : bool
        The ``update_check.enabled`` setting.
    say, ask : callable
        How to print a line and how to put a question. Replaced in
        tests.

    Returns
    -------
    str or None
        The tag offered, or ``None`` when there was nothing to say.
    """
    if not enabled or os.environ.get(ENV_OFF):
        return None
    clone = find_clone()
    if clone is None or current_branch(clone) != RELEASE_BRANCH:
        return None
    latest = latest_release()
    if latest is None:
        return None
    tag, url = latest
    if not is_newer(tag, installed_version()):
        return None

    manual = f'cd {clone} && git pull origin {RELEASE_BRANCH} && pip install .'
    say(f'PyLithics {tag} is available. You have v{installed_version()}.')
    if not sys.stdin.isatty():
        say(f'To update: {manual}')
        return tag
    if not _yes(ask, 'Update PyLithics? [y/N] '):
        return tag
    error = run_update(clone)
    if error:
        say(f'The update did not complete: {error}')
        say(f'To update by hand: {manual}')
    else:
        say(f'PyLithics is updated to {tag}. Release notes: {url}')
    return tag


def installed_version() -> str:
    """Return the installed PyLithics version, or ``0`` if unknown."""
    try:
        from importlib.metadata import version
        return version('pylithics')
    except Exception:  # pragma: no cover - metadata missing
        return '0'


def is_newer(tag: str, installed: str) -> bool:
    """Report whether release ``tag`` is newer than ``installed``."""
    latest, current = _parse(tag), _parse(installed)
    if latest is None or current is None:
        return False
    return latest > current


def _parse(text: str) -> Optional[Tuple[int, ...]]:
    """Turn ``v2.1.0`` or ``2.1.0`` into ``(2, 1, 0)``; None if not numeric."""
    parts = text.strip().lstrip('vV').split('.')
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def find_clone() -> Optional[str]:
    """
    Locate the PyLithics clone: the install source, else a parent of cwd.

    A clone is a folder with ``.git`` whose ``origin`` names this
    repository. Without one there is no branch to check and no place
    to run the update, so the check does nothing.
    """
    candidates = []
    if os.path.isfile(_SOURCE_FILE):
        with open(_SOURCE_FILE, encoding='utf-8') as handle:
            candidates.append(handle.read().strip())
    folder = os.getcwd()
    while True:
        candidates.append(folder)
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    for candidate in candidates:
        if candidate and _is_this_repo(candidate):
            return candidate
    return None


def _is_this_repo(folder: str) -> bool:
    """Report whether ``folder`` is a clone of the PyLithics repository."""
    if not os.path.isdir(os.path.join(folder, '.git')):
        return False
    origin = _git(folder, 'remote', 'get-url', 'origin')
    return origin is not None and 'Palaeoanalytics' in origin


def current_branch(clone: str) -> Optional[str]:
    """Return the checked-out branch of ``clone``; None when detached."""
    return _git(clone, 'symbolic-ref', '--short', 'HEAD')


def _git(folder: str, *args: str) -> Optional[str]:
    """Run a git command in ``folder`` and return its output, or None."""
    try:
        done = subprocess.run(
            ['git', '-C', folder, *args], capture_output=True, text=True,
            timeout=TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def latest_release() -> Optional[Tuple[str, str]]:
    """
    Return ``(tag, release notes URL)`` of the latest release.

    Read from the cache when it is less than a day old, else from
    GitHub, then cached. None when GitHub cannot be reached.
    """
    cached = _read_cache()
    if cached is not None:
        return cached
    try:
        request = urllib.request.Request(
            RELEASES_URL, headers={'User-Agent': 'PyLithics'},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = json.load(response)
        tag, url = data['tag_name'], data['html_url']
    except Exception as exc:  # any failure: no network, rate limit, bad JSON
        logging.debug('Update check skipped: %s', exc)
        return None
    _write_cache(tag, url)
    return tag, url


def _cache_path() -> str:
    """The cache file, under ``$XDG_CACHE_HOME`` or ``~/.cache``."""
    base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'pylithics', 'update_check.json')


def _read_cache() -> Optional[Tuple[str, str]]:
    """Return the cached release if it was checked within the interval."""
    try:
        with open(_cache_path(), encoding='utf-8') as handle:
            data = json.load(handle)
        if time.time() - float(data['checked_at']) < CHECK_INTERVAL:
            return data['tag'], data['url']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _write_cache(tag: str, url: str) -> None:
    """Record the release and the time of the check."""
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'checked_at': time.time(), 'tag': tag, 'url': url}, handle)
    except OSError:
        pass


def run_update(clone: str) -> Optional[str]:
    """
    Pull the release branch and reinstall. Return an error text, or None.

    The install uses the Python that runs PyLithics, so the update
    lands in the same environment.
    """
    steps = (
        ['git', '-C', clone, 'pull', 'origin', RELEASE_BRANCH],
        [sys.executable, '-m', 'pip', 'install', '--quiet', clone],
    )
    for step in steps:
        try:
            done = subprocess.run(step, capture_output=True, text=True, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return str(exc)
        if done.returncode != 0:
            output = (done.stderr or done.stdout).strip()
            if output:
                return output.splitlines()[-1]
            return f'{step[0]} returned {done.returncode}'
    return None


def _yes(ask: Callable[[str], str], question: str) -> bool:
    """Put a yes/no question. No is the default."""
    try:
        return ask(question).strip().lower() in ('y', 'yes')
    except EOFError:
        return False
