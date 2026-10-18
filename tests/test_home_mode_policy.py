"""Home permissions must not depend on whether startup already loaded .env."""

import os
import stat

import pytest

from hermes_constants import get_scratch_dir
from hermes_cli.config import ensure_hermes_home


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("override, expected", [(None, 0o770), ("0750", 0o750), ("invalid", 0o700)])
def test_home_mode_without_loading_dotenv(tmp_path, monkeypatch, override, expected):
    home = tmp_path / "home"
    home.mkdir(mode=0o770)
    (home / ".env").write_text("export HERMES_HOME_MODE='0770' # shared home\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_HOME_MODE", "HERMES_MANAGED", "HERMES_CONTAINER", "HERMES_SKIP_CHMOD"):
        monkeypatch.delenv(key, raising=False)
    if override is not None:
        monkeypatch.setenv("HERMES_HOME_MODE", override)
    # These paths run before dotenv loading in CLI/dashboard startup.
    ensure_hermes_home()
    scratch = get_scratch_dir(home, prune=False)
    for directory in (home, home / "memories", scratch):
        assert stat.S_IMODE(directory.stat().st_mode) == expected
    assert ("HERMES_HOME_MODE" in os.environ) == (override is not None)
    # Boot can ask for another home before switching the effective profile.
    other = tmp_path / "other"
    other.mkdir()
    (other / ".env").write_text("HERMES_HOME_MODE=0750\n")
    other_scratch = get_scratch_dir(other, prune=False)
    assert stat.S_IMODE(other_scratch.stat().st_mode) == (0o750 if override is None else expected)
