"""The google-genai SDK, imported once with its own compile-time noise silenced.

google-genai 3.0.0 ships string literals with invalid backslash escapes, so compiling it without a
cached .pyc emits a SyntaxWarning. Under `filterwarnings = error` that would turn the first
import into a failure, so the import happens here, under a local ignore filter.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from google import genai
    from google.genai import errors, types

__all__ = ["errors", "genai", "types"]
