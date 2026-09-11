from .configs import MODES


def list_modes() -> list[str]:
    """All available summary modes (used for API validation)."""
    return list(MODES.keys())


def get_mode(name: str) -> dict:
    """Return the config for a mode: {"template", "description", "focus"}. Raises ValueError if unknown."""
    if name not in MODES:
        raise ValueError(
            f"Unknown summary mode: {name!r}. Available modes: {list_modes()}"
        )
    return MODES[name]