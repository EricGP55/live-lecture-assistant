"""Local web server: full page (/), mini window (/mini) and event WebSocket (/ws)."""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import ROOT

WEB = ROOT / "web"


class Hub:
    """Bridge from the pipeline thread to the WebSocket clients."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.q: asyncio.Queue | None = None
        self.latest: dict[str, dict] = {}  # last status/class event, replayed to new clients

    def emit(self, ev: dict):
        if ev["type"] in ("status", "class"):
            self.latest[ev["type"]] = ev
        if self.loop and self.q:
            self.loop.call_soon_threadsafe(self.q.put_nowait, ev)

    async def broadcast(self):
        while True:
            ev = await self.q.get()
            msg = json.dumps(ev, ensure_ascii=False)
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self.clients.discard(ws)


def create_app(hub: Hub, pipeline_factory):
    state = {"pipeline": None}

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        hub.q = asyncio.Queue()
        task = asyncio.create_task(hub.broadcast())
        p = pipeline_factory()
        state["pipeline"] = p
        p.start()
        yield
        p.stop()
        await asyncio.to_thread(p.join, 30)
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

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        p = state["pipeline"]
        for kind in ("class", "status"):
            if kind in hub.latest:
                await sock.send_text(json.dumps(hub.latest[kind], ensure_ascii=False))
        if p and p.class_id:
            for s in p.store.sentences(p.class_id):
                await sock.send_text(json.dumps({
                    "type": "sentence", "id": s["id"], "t0": s["t_start"], "t1": s["t_end"],
                    "text": s["text"], "speaker": s["speaker"], "p": s["p_teacher"]},
                    ensure_ascii=False))
        hub.clients.add(sock)
        try:
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            hub.clients.discard(sock)

    return app
