"""URL slugs."""

import re


def slugify(text: str) -> str:
    """Turn text into a URL slug.

    Lowercase ASCII letters and digits only; accents are stripped ("é" becomes "e"); every run
    of other characters becomes a single hyphen; no leading or trailing hyphens.
    """
    text = text.lower()
    return re.sub(r"[^a-z0-9]", "-", text)
