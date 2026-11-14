"""Identifier style conversion."""


def snake_to_camel(name: str) -> str:
    """lower_snake_case -> lowerCamelCase. Leading underscores are kept as they are; empty
    parts from doubled underscores are dropped."""
    stripped = name.lstrip("_")
    prefix = name[: len(name) - len(stripped)]
    parts = [p for p in stripped.split("_") if p]
    if not parts:
        return name
    return prefix + parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
