"""Permission policy for explicitly shared Hermes homes, not arbitrary workspaces.

Group bits are the POSIX ACL mask. Clearing them silently disables named users
as well as the owning group, even when the ACL entries themselves survive.
"""
from pathlib import Path
import stat
from typing import overload


def shared_home_mode(home=None, *, node=None) -> int | None:
    """Return the verified opt-in mode, or None (private/fail closed).

    Merely being group writable is not consent. The operator's mode must match
    the actual home, must grant group write, and must never grant world write.
    Resolve at call time so routed profiles don't borrow the launch home's policy.
    """
    from hermes_constants import get_configured_home_mode, get_hermes_home

    home = Path(home) if home is not None else get_hermes_home()
    try:
        configured = int(get_configured_home_mode(home), 8)
        actual = stat.S_IMODE((node if node is not None else home.stat()).st_mode)
    except (OSError, ValueError):
        return None
    if configured < 0 or configured > 0o7777 or configured & 0o002:
        return None
    if configured & 0o777 != actual & 0o777 or not configured & 0o020:
        return None
    return configured


@overload
def home_write_mode(path, mode: int, *, home=None, directory=False) -> int: ...


@overload
def home_write_mode(path, mode: None, *, home=None, directory=False) -> int | None: ...


def home_write_mode(path, mode: int | None, *, home=None, directory=False) -> int | None:
    """Preserve the shared ACL mask only inside the selected, verified home.

    Outside-home targets (including symlinks escaping it) retain the caller's
    private mode. Sockets/ownership locks intentionally do not use this policy.
    """
    from hermes_constants import get_hermes_home

    home = Path(home) if home is not None else get_hermes_home()
    try:
        if not Path(path).resolve().is_relative_to(home.resolve()):
            return mode
    except (OSError, ValueError, RuntimeError):
        return mode
    shared = shared_home_mode(home)
    if shared is None:
        return mode
    return ((mode if mode is not None else 0o600) | (shared & (0o070 if directory else 0o060))) & ~0o007
