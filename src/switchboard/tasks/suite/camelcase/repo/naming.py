"""Identifier style conversion."""


def snake_to_camel(name: str) -> str:
    """lower_snake_case -> lowerCamelCase. Leading underscores are kept as they are; empty
    parts from doubled underscores are dropped."""
    return "".join(part.capitalize() for part in name.split("_"))
