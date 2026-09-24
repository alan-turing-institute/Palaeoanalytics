"""
The update check: once a day, on main only, never in the way.

GitHub and git are replaced with fakes, so the tests run offline and
never touch the real clone.
"""

import json
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from pylithics import update_check


@pytest.fixture
def clone(tmp_path):
    """A throwaway git clone of 'the repository', on main."""
    repo = tmp_path / "Palaeoanalytics"
    repo.mkdir()
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
    )
    run("init", "-q", "-b", "main")
    run("remote", "add", "origin",
        "https://github.com/alan-turing-institute/Palaeoanalytics.git")
    return str(repo)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv(update_check.ENV_OFF, raising=False)
    return tmp_path / "cache"


def _github(tag="v9.9.9", url="https://github.com/x/releases/tag/v9.9.9"):
    """Patch the GitHub call to return one release."""
    return patch.object(update_check, "latest_release", return_value=(tag, url))


@pytest.mark.unit
class TestVersions:
    def test_newer_tag(self):
        assert update_check.is_newer("v2.1.0", "2.0.0")

    def test_same_or_older_is_not_newer(self):
        assert not update_check.is_newer("v2.0.0", "2.0.0")
        assert not update_check.is_newer("v1.9.9", "2.0.0")

    def test_non_numeric_tag_is_ignored(self):
        assert not update_check.is_newer("latest", "2.0.0")


@pytest.mark.unit
class TestClone:
    def test_finds_the_clone_from_a_folder_inside_it(self, clone, monkeypatch):
        inside = os.path.join(clone, "pylithics", "data")
        os.makedirs(inside)
        monkeypatch.chdir(inside)
        with patch.object(update_check, "_SOURCE_FILE", "/nonexistent"):
            assert update_check.find_clone() == clone

    def test_a_repo_with_another_origin_is_not_the_clone(self, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(other), "remote", "add", "origin",
                        "https://example.com/else.git"], check=True)
        monkeypatch.chdir(other)
        with patch.object(update_check, "_SOURCE_FILE", "/nonexistent"):
            assert update_check.find_clone() is None

    def test_the_install_source_file_wins(self, clone, tmp_path, monkeypatch):
        source = tmp_path / "_source_path.txt"
        source.write_text(clone)
        monkeypatch.chdir(tmp_path)
        with patch.object(update_check, "_SOURCE_FILE", str(source)):
            assert update_check.find_clone() == clone

    def test_branch(self, clone):
        assert update_check.current_branch(clone) == "main"


@pytest.mark.unit
class TestCache:
    def test_a_fresh_cache_avoids_github(self, cache_dir):
        update_check._write_cache("v3.0.0", "u")
        with patch("urllib.request.urlopen", side_effect=AssertionError("called")):
            assert update_check.latest_release() == ("v3.0.0", "u")

    def test_a_stale_cache_is_refreshed(self, cache_dir):
        path = update_check._cache_path()
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as handle:
            json.dump({"checked_at": time.time() - 2 * update_check.CHECK_INTERVAL,
                       "tag": "v1.0.0", "url": "old"}, handle)

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"tag_name": "v4.0.0", "html_url": "new"}'

        with patch("urllib.request.urlopen", return_value=Response()):
            assert update_check.latest_release() == ("v4.0.0", "new")
        assert update_check._read_cache() == ("v4.0.0", "new")

    def test_no_network_is_silent(self, cache_dir):
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            assert update_check.latest_release() is None


@pytest.mark.unit
class TestOffer:
    """What the user sees, and when."""

    def _run(self, clone, answer="", tty=True, branch="main", enabled=True):
        said = []
        with _github(), \
             patch.object(update_check, "find_clone", return_value=clone), \
             patch.object(update_check, "current_branch", return_value=branch), \
             patch.object(update_check, "installed_version", return_value="2.0.0"), \
             patch.object(update_check, "run_update", return_value=None) as updated, \
             patch("sys.stdin.isatty", return_value=tty):
            tag = update_check.check_for_update(
                enabled, say=said.append, ask=lambda q: answer,
            )
        return tag, said, updated

    def test_newer_release_on_main_is_offered(self, clone, cache_dir):
        tag, said, updated = self._run(clone, answer="y")
        assert tag == "v9.9.9"
        assert said[0] == "PyLithics v9.9.9 is available. You have v2.0.0."
        updated.assert_called_once_with(clone)
        assert said[-1].startswith("PyLithics is updated to v9.9.9. Release notes:")

    def test_no_means_notice_only(self, clone, cache_dir):
        _, said, updated = self._run(clone, answer="")
        assert len(said) == 1
        updated.assert_not_called()

    def test_a_script_gets_the_command_and_no_question(self, clone, cache_dir):
        _, said, updated = self._run(clone, tty=False)
        assert said[1].startswith("To update: cd ")
        updated.assert_not_called()

    def test_another_branch_is_not_told(self, clone, cache_dir):
        tag, said, _ = self._run(clone, branch="develop")
        assert tag is None and said == []

    def test_switched_off_in_config(self, clone, cache_dir):
        tag, said, _ = self._run(clone, enabled=False)
        assert tag is None and said == []

    def test_switched_off_by_environment(self, clone, cache_dir, monkeypatch):
        monkeypatch.setenv(update_check.ENV_OFF, "1")
        tag, said, _ = self._run(clone)
        assert tag is None and said == []

    def test_no_clone_means_silence(self, cache_dir):
        tag, said, _ = self._run(None)
        assert tag is None and said == []

    def test_a_failed_update_gives_the_manual_command(self, clone, cache_dir):
        said = []
        with _github(), \
             patch.object(update_check, "find_clone", return_value=clone), \
             patch.object(update_check, "current_branch", return_value="main"), \
             patch.object(update_check, "installed_version", return_value="2.0.0"), \
             patch.object(update_check, "run_update", return_value="pull failed"), \
             patch("sys.stdin.isatty", return_value=True):
            update_check.check_for_update(True, say=said.append, ask=lambda q: "y")
        assert said[1] == "The update did not complete: pull failed"
        assert said[2].startswith("To update by hand: cd ")
