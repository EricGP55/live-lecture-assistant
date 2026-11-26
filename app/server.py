"""Local web server: full page (/), mini window (/mini) and event WebSocket (/ws).

The WebSocket carries live class events to every client, and answers to questions
only to the client that asked.
"""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import ROOT

WEB = ROOT / "web"
SCOPES = {"class", "subject", "all"}


class Hub:
    """Bridge from the pipeline thread to the WebSocket clients."""

    def __init__(self):
        self.clients: dict[WebSocket, asyncio.Lock] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.q: asyncio.Queue | None = None
        self.latest: dict[str, dict] = {}  # last status/class event, replayed to new clients

    def emit(self, ev: dict):
        """Thread-safe: called from the pipeline thread."""
        if ev["type"] in ("status", "class"):
            self.latest[ev["type"]] = ev
        if self.loop and self.q:
            self.loop.call_soon_threadsafe(self.q.put_nowait, ev)

    async def send(self, ws: WebSocket, ev: dict):
        lock = self.clients.get(ws)
        if lock is None:
            return
        try:
            async with lock:  # one writer at a time per socket
                await ws.send_text(json.dumps(ev, ensure_ascii=False))
        except Exception:
            self.clients.pop(ws, None)

    async def broadcast(self):
        while True:
            ev = await self.q.get()
            for ws in list(self.clients):
                await self.send(ws, ev)


def create_app(hub: Hub, pipeline_factory, qa_factory=None):
    state = {"pipeline": None, "qa": None}

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        hub.q = asyncio.Queue()
        task = asyncio.create_task(hub.broadcast())
        p = pipeline_factory()
        state["pipeline"] = p
        p.start()
        if qa_factory:
            state["qa"] = qa_factory()
            state["qa"].llm.warm_up()  # open the first Claude session in the background
        yield
        p.stop()
        await asyncio.to_thread(p.join, 30)
        if state["qa"]:
            await state["qa"].llm.close()
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/mini")
    async def mini():
        return FileResponse(WEB / "mini.html")

    @app.post("/api/stop")
    async def stop():
        state["pipeline"].stop()
        return {"ok": True}

    async def answer(sock, question, scope):
        qa, p = state["qa"], state["pipeline"]
        if qa is None or p is None or p.class_id is None:
            await hub.send(sock, {"type": "answer_start", "scope": scope, "sources": 0})
            await hub.send(sock, {"type": "answer_done", "error": "The class has not started yet."})
            return
        try:
            await qa.ask(question, scope, p, lambda ev: hub.send(sock, ev))
        except Exception as e:  # never leave the page waiting
            await hub.send(sock, {"type": "answer_done", "error": str(e)})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        hub.clients[sock] = asyncio.Lock()
        p = state["pipeline"]
        for kind in ("class", "status"):
            if kind in hub.latest:
                await hub.send(sock, hub.latest[kind])
        if p and p.class_id:
            for s in p.store.sentences(p.class_id):
                await hub.send(sock, {"type": "sentence", "id": s["id"], "t0": s["t_start"],
                                      "t1": s["t_end"], "text": s["text"], "speaker": s["speaker"],
                                      "p": s["p_teacher"]})
            for q in p.store.questions(p.class_id):
                await hub.send(sock, {"type": "past_answer", "question": q["question"],
                                      "answer": q["answer"] or "", "scope": q["scope"]})
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if msg.get("type") == "ask" and msg.get("question", "").strip():
                    scope = msg.get("scope") if msg.get("scope") in SCOPES else "class"
                    asyncio.create_task(answer(sock, msg["question"].strip()[:500], scope))
        except (WebSocketDisconnect, json.JSONDecodeError):
            pass
        finally:
            hub.clients.pop(sock, None)

    return app
