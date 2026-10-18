"""Atomic memory updates must keep shared-home access, including inherited ACLs."""

import shutil
import stat
import subprocess

import pytest

from tools.memory_tool import MemoryStore


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("target, filename", [("memory", "MEMORY.md"), ("user", "USER.md")])
@pytest.mark.parametrize("home_mode, file_mode", [("0770", 0o660), ("0750", 0o640), ("0701", 0o600)])
def test_memory_updates_follow_home_sharing(tmp_path, monkeypatch, target, filename, home_mode, file_mode):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"HERMES_HOME_MODE={home_mode}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    store = MemoryStore()
    store.load_from_disk()
    path = home / "memories" / filename
    assert store.add(target, "first note")["success"]
    assert stat.S_IMODE(path.stat().st_mode) == file_mode
    # Repair an old update's 0600 mask on the next write too.
    path.chmod(0o600)
    assert store.add(target, "second note")["success"]
    assert stat.S_IMODE(path.with_suffix(".md.lock").stat().st_mode) == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == file_mode
    assert "second note" in path.read_text()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("managed", [False, True])
def test_memory_update_preserves_existing_mode_without_explicit_policy(tmp_path, monkeypatch, managed):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    if managed:
        (home / ".managed").write_text("nixos")
    path = memories / "MEMORY.md"
    path.write_text("first note")
    path.chmod(0o640)
    store = MemoryStore()
    store.load_from_disk()
    assert store.add("memory", "second note")["success"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.platforms("linux")
def test_memory_replacement_keeps_default_acl_effective(tmp_path, monkeypatch):
    if not shutil.which("setfacl") or not shutil.which("getfacl"):
        pytest.skip("POSIX ACL utilities required")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("HERMES_HOME_MODE=0770\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    # A distinct numeric user avoids depending on the host's account database.
    subprocess.run(["setfacl", "-m", "u:12345:rwx,d:u:12345:rwx", str(home)], check=True)
    from hermes_cli.config import ensure_hermes_home

    ensure_hermes_home()
    store = MemoryStore()
    store.load_from_disk()
    for note in ("first note", "second note"):
        assert store.add("memory", note)["success"]
        acl = subprocess.check_output(
            ["getfacl", "-cpn", str(home / "memories" / "MEMORY.md")], text=True)
        assert "user:12345:rwx\t#effective:rw-" in acl
        assert "mask::rw-" in acl
        assert stat.S_IMODE(home.stat().st_mode) == 0o770
