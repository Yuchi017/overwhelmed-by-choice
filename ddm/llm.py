"""Async client for a vLLM (OpenAI-compatible) server, with per-request cost logging."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


def _usage_tokens(resp: Any) -> Dict[str, Optional[int]]:
    usage = getattr(resp, "usage", None)
    get = lambda name: getattr(usage, name, None) if usage is not None else None  # noqa: E731
    prompt, completion = get("prompt_tokens"), get("completion_tokens")
    total = get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


class LLMClient:
    """Greedy-decoding chat client with a concurrency limit.

    `complete(prompt)` returns `(text, metrics)` where metrics records queue wait, API
    latency, character counts and token usage for the request.
    """

    def __init__(self, model: str, port: int = 8000, max_tokens: int = 100,
                 concurrency: int = 5, timeout: int = 3600, base_url: Optional[str] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url or f"http://localhost:{port}/v1",
                                  timeout=timeout)
        self._sem = asyncio.Semaphore(concurrency)
        self.request_log: List[Dict[str, Any]] = []

    async def complete(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        queued = time.perf_counter()
        async with self._sem:
            started = time.perf_counter()
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                max_tokens=self.max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            finished = time.perf_counter()
        text = resp.choices[0].message.content or ""
        metrics = {
            "request_id": len(self.request_log) + 1,
            "queue_wait_seconds": started - queued,
            "api_latency_seconds": finished - started,
            "end_to_end_request_seconds": time.perf_counter() - queued,
            "prompt_characters": len(prompt),
            "response_characters": len(text),
            **_usage_tokens(resp),
        }
        self.request_log.append(metrics)
        return text, metrics

    async def complete_all(self, prompts: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        return await asyncio.gather(*(self.complete(p) for p in prompts))
