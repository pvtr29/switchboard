import asyncio
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _main_thread_event_loop() -> Iterator[None]:
    # looptime's pytest_pyfunc_call hook calls policy.get_event_loop() for every test. In a sync
    # test with no loop set, Python 3.12 emits a DeprecationWarning (fatal under
    # filterwarnings=error) or implicitly creates a loop that is never closed. Owning one
    # explicit loop for the session avoids both, whatever order the tests run in.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    asyncio.set_event_loop(None)
    loop.close()
