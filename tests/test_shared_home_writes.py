"""Exercise real config/auth writes with an inherited POSIX named-user ACL."""
import os
import shutil
import subprocess

import pytest


@pytest.mark.platforms("linux")
def test_config_and_auth_keep_shared_acl(tmp_path, monkeypatch):
    if not shutil.which("setfacl") or not shutil.which("getfacl"):
        pytest.skip("POSIX ACL tools required")
    home = tmp_path / "shared"
    home.mkdir()
    home.chmod(0o770)
    # A numeric, non-owner uid avoids depending on the host's account database.
    peer = os.getuid() + 1
    subprocess.run(["setfacl", "-m", f"u:{peer}:rwx,d:u:{peer}:rwx", str(home)], check=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_MODE", "0770")
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    from hermes_cli.config import save_config
    from hermes_cli.auth import _save_auth_store
    save_config({"model": "test-model"})
    from hermes_cli.personality import persist_personality
    assert persist_personality("default")
    _save_auth_store({"providers": {}}, target_path=home / "auth.json")
    for path in (home, home / "config.yaml", home / "auth.json"):
        acl = subprocess.check_output(["getfacl", "-cpn", str(path)], text=True)
        entry = next(row for row in acl.splitlines() if row.startswith(f"user:{peer}:"))
        assert "#effective:---" not in entry, (path, acl)
        assert path.stat().st_mode & 0o060 == 0o060, (path, acl)
        assert path.stat().st_mode & 0o007 == 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("configured,actual,expected", [
    ("0770", 0o770, 0o660), ("2770", 0o770, 0o660),
    ("0770", 0o750, 0o600), ("0777", 0o777, 0o600),
    ("invalid", 0o770, 0o600), ("0700", 0o700, 0o600),
])
def test_sharing_requires_matching_opt_in(tmp_path, monkeypatch, configured, actual, expected):
    from hermes_home_permissions import home_write_mode
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(actual)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_MODE", configured)
    assert home_write_mode(home / "key", 0o600) == expected
    assert home_write_mode(tmp_path / "outside", 0o600) == 0o600
    (home / "escape").symlink_to(tmp_path / "outside")
    assert home_write_mode(home / "escape", 0o600) == 0o600


@pytest.mark.platforms("posix")
def test_target_home_policy_does_not_use_active_home(tmp_path, monkeypatch):
    from hermes_home_permissions import home_write_mode
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o770)
    (shared / ".env").write_text("HERMES_HOME_MODE=0770\n")
    monkeypatch.setenv("HERMES_HOME", str(private))
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    assert home_write_mode(shared / "key", 0o600, home=shared) == 0o660
    assert home_write_mode(private / "key", 0o600) == 0o600


@pytest.mark.platforms("posix")
def test_shared_credential_store_and_existing_vault_key(tmp_path, monkeypatch):
    from hermes_constants import ensure_private_store
    from agent.vault_store import VaultStore
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o770)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_MODE", "0770")
    assert ensure_private_store(home / "credentials") == 0o007
    assert (home / "credentials").stat().st_mode & 0o777 == 0o770
    store = VaultStore()
    key = store._key()
    assert store._key() == key
    assert (home / "vault" / "vault.key").stat().st_mode & 0o777 == 0o660
