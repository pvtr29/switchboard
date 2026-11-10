"""URL slugs."""

import re
import unicodedata


def slugify(text: str) -> str:
    """Turn text into a URL slug.

    Lowercase ASCII letters and digits only; accents are stripped ("é" becomes "e"); every run
    of other characters becomes a single hyphen; no leading or trailing hyphens.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
