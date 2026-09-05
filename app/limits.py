import asyncio
from contextlib import asynccontextmanager

from app.settings import load_settings


class ModelCallGate:
    # ponytail: process-local; add a distributed limiter before multiple Python workers.
    def __init__(self, max_concurrent: int, min_interval_ms: int = 0):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pace_lock = asyncio.Lock()
        self._min_interval_seconds = max(0, min_interval_ms) / 1000
        self._next_start = 0.0

    @asynccontextmanager
    async def slot(self):
        async with self._semaphore:
            async with self._pace_lock:
                loop = asyncio.get_running_loop()
                delay = self._next_start - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                self._next_start = loop.time() + self._min_interval_seconds
            yield


_settings = load_settings()
generation_gate = ModelCallGate(
    _settings.model_max_concurrent_requests, _settings.model_min_interval_ms,
)
local_gate = ModelCallGate(_settings.model_max_concurrent_requests)
