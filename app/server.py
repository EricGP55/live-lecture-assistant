"""Servidor web local: página completa (/), ventana mini (/mini) y WebSocket de eventos (/ws)."""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import RAIZ

WEB = RAIZ / "web"


class Hub:
    """Puente hilo del pipeline -> clientes WebSocket."""

    def __init__(self):
        self.clientes: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.q: asyncio.Queue | None = None
        self.ultimo: dict[str, dict] = {}  # último evento de tipo estado/clase para nuevos clientes

    def emitir(self, ev: dict):
        if ev["tipo"] in ("estado", "clase"):
            self.ultimo[ev["tipo"]] = ev
        if self.loop and self.q:
            self.loop.call_soon_threadsafe(self.q.put_nowait, ev)

    async def difundir(self):
        while True:
            ev = await self.q.get()
            msg = json.dumps(ev, ensure_ascii=False)
            for ws in list(self.clientes):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self.clientes.discard(ws)


def crear_app(hub: Hub, pipeline_factory):
    estado = {"pipeline": None}

    @asynccontextmanager
    async def vida(app):
        hub.loop = asyncio.get_running_loop()
        hub.q = asyncio.Queue()
        tarea = asyncio.create_task(hub.difundir())
        p = pipeline_factory()
        estado["pipeline"] = p
        p.start()
        yield
        p.parar()
        await asyncio.to_thread(p.join, 30)
        tarea.cancel()

    app = FastAPI(lifespan=vida)

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/mini")
    async def mini():
        return FileResponse(WEB / "mini.html")

    @app.post("/api/parar")
    async def parar():
        estado["pipeline"].parar()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        p = estado["pipeline"]
        for tipo in ("clase", "estado"):
            if tipo in hub.ultimo:
                await sock.send_text(json.dumps(hub.ultimo[tipo], ensure_ascii=False))
        if p and p.clase_id:
            for f in p.store.frases(p.clase_id):
                await sock.send_text(json.dumps({
                    "tipo": "frase", "id": f["id"], "t0": f["t_inicio"], "t1": f["t_fin"],
                    "texto": f["texto"], "hablante": f["hablante"], "p": f["p_profe"]},
                    ensure_ascii=False))
        hub.clientes.add(sock)
        try:
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            hub.clientes.discard(sock)

    return app
