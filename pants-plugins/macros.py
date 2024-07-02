def is_standalone():
    return True


def emote_dependency_path(suffix: str) -> str:
    if suffix.startswith("/"):
        return f"/{suffix}"
    return f"//{suffix}"


def emote_root_dir() -> str:
    return "."
