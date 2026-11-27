"""LLM access through the Claude subscription (Claude Agent SDK), tuned for low latency and few tokens.

- Every question runs in a *fresh* session: reusing a session makes the second turn load
  ~28k extra tokens, while a fresh one costs ~600 tokens of overhead.
- Opening a session takes ~5 s, so one is kept connected and waiting ("warm spare");
  after each question the next spare is prepared in the background.
- No tools, no settings files, no account connectors, a short custom system prompt.
- One shared instance for every task (each waiting session costs ~150 MB of RAM), so the
  system prompt is generic and each task puts its instructions at the top of the message.
"""

GENERIC_SYSTEM_PROMPT = (
    "You are the assistant inside a live-class app used by a student. "
    "Each message starts with the task instructions; follow them exactly. "
    "Be concise. When asked for JSON, reply with the JSON object only."
)


def parse_json(text: str) -> dict | None:
    """Extracts the first JSON object from a model reply (tolerates code fences and stray text)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass, field

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import ResultMessage, StreamEvent


@dataclass
class Answer:
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    ttft_ms: int | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)


class SubscriptionLLM:
    def __init__(self, system_prompt: str = GENERIC_SYSTEM_PROMPT, model: str = "sonnet",
                 effort: str | None = "low"):
        # Force the subscription login even if an API key is set in the environment.
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            effort=effort,
            tools=[],
            setting_sources=[],
            # Without this, the account's claude.ai connectors (Drive, Calendar…) load as ~40
            # tools once the session has warmed up: ~28k extra tokens per question.
            strict_mcp_config=True,
            env={"ENABLE_CLAUDEAI_MCP_SERVERS": "false"},
            max_turns=1,
            include_partial_messages=True,
            cwd=tempfile.gettempdir(),
        )
        self._spare: asyncio.Task | None = None
        self.totals = {"calls": 0, "tokens_in": 0, "tokens_out": 0}  # this run, for the notes
        self._lock = asyncio.Lock()

    async def _open(self) -> ClaudeSDKClient:
        client = ClaudeSDKClient(self.options)
        await client.connect()
        return client

    def warm_up(self):
        """Start opening the next session in the background (call from the event loop)."""
        if self._spare is None:
            self._spare = asyncio.create_task(self._open())

    async def _take(self) -> ClaudeSDKClient:
        self.warm_up()
        task, self._spare = self._spare, None
        try:
            return await task
        except Exception:
            return await self._open()  # the spare died (e.g. network); try once more

    async def stream(self, prompt: str, on_delta):
        """Streams the answer, calling on_delta(text) for each chunk. Returns an Answer."""
        async with self._lock:  # one question at a time keeps usage predictable
            ans = Answer()
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            try:
                client = await self._take()
            except Exception as e:
                ans.error = f"Could not open a Claude session: {e}"
                return ans
            self.warm_up()  # prepare the next one while this answer streams
            try:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, StreamEvent):
                        ev = msg.event
                        if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                            if ans.ttft_ms is None:
                                ans.ttft_ms = int((loop.time() - t0) * 1000)
                            ans.text += ev["delta"]["text"]
                            await on_delta(ev["delta"]["text"])
                    elif isinstance(msg, ResultMessage):
                        u = msg.usage or {}
                        ans.usage = u
                        ans.tokens_in = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                                         + u.get("cache_creation_input_tokens", 0))
                        ans.tokens_out = u.get("output_tokens", 0)
                        self.totals["calls"] += 1
                        self.totals["tokens_in"] += ans.tokens_in
                        self.totals["tokens_out"] += ans.tokens_out
                        if msg.is_error:
                            ans.error = msg.result or "Claude returned an error"
            except Exception as e:
                ans.error = str(e)
            finally:
                asyncio.create_task(self._close(client))
            return ans

    async def complete(self, prompt: str) -> Answer:
        """Same as stream(), without incremental output."""
        async def ignore(_):
            pass
        return await self.stream(prompt, ignore)

    @staticmethod
    async def _close(client):
        try:
            await client.disconnect()
        except Exception:
            pass

    async def close(self):
        if self._spare:
            try:
                await self._close(await self._spare)
            except Exception:
                pass
