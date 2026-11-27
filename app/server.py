"""Local web server: full page (/), mini window (/mini) and event WebSocket (/ws).

The WebSocket carries live class events and doubt updates to every client; answers to
questions and suggestions go only to the client that asked.
"""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from pathlib import Path

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
        self.listeners = []                # async fn(event) called for every pipeline event

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

    async def send_all(self, ev: dict):
        for ws in list(self.clients):
            await self.send(ws, ev)

    async def run(self):
        while True:
            ev = await self.q.get()
            await self.send_all(ev)
            for fn in self.listeners:
                asyncio.create_task(fn(ev))


def create_app(hub: Hub, pipeline_factory, services_factory=None):
    """services_factory(send_all) -> {"llm": ..., "qa": ..., "doubts": ...}"""
    state = {"pipeline": None, "svc": {}}

    async def on_event(ev):
        p, doubts = state["pipeline"], state["svc"].get("doubts")
        alerts = state["svc"].get("alerts")
        if not doubts or not p or p.class_id is None:
            return
        try:
            if ev["type"] == "sentence":
                await asyncio.gather(doubts.on_sentence(ev, p), alerts.on_sentence(ev, p),
                                     state["svc"]["notes"].on_sentence(p))
            elif ev["type"] == "class_end":
                state["svc"]["notes"].generate_once(ev["class_id"])
            elif ev["type"] == "pause":
                await doubts.on_pause(ev, p)
        except Exception as e:  # watchers must never break the class
            print("watcher error:", e)

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        hub.q = asyncio.Queue()
        task = asyncio.create_task(hub.run())
        p = pipeline_factory()
        state["pipeline"] = p
        p.start()
        if services_factory:
            state["svc"] = services_factory(hub.send_all)
            state["svc"]["llm"].warm_up()  # open the first Claude session in the background
            hub.listeners.append(on_event)
        yield
        p.stop()
        await asyncio.to_thread(p.join, 30)
        notes = state["svc"].get("notes")
        if notes and p.class_id is not None:
            # Ctrl+C: finish (or write) the notes before closing; the "End class" button already started them
            print("Writing the class notes…")
            try:
                path = await asyncio.wait_for(asyncio.shield(notes.generate_once(p.class_id)), 240)
                print(f"Notes: {path}" if path else "No notes: nothing was transcribed.")
            except Exception as e:
                print("Could not write the notes:", e)
        if "llm" in state["svc"]:
            await state["svc"]["llm"].close()
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/mini")
    async def mini():
        return FileResponse(WEB / "mini.html")

    @app.get("/notes")
    @app.get("/notes/{class_id}")
    async def notes_page(class_id: int | None = None):
        return FileResponse(WEB / "notes.html")

    @app.get("/api/classes")
    async def classes():
        return await asyncio.to_thread(state["pipeline"].store.classes)

    @app.get("/api/notes/{class_id}")
    async def notes_md(class_id: int):
        from fastapi.responses import PlainTextResponse
        info = await asyncio.to_thread(state["pipeline"].store.class_info, class_id)
        path = info and info.get("notes_path")
        if not path or not Path(path).exists():
            return PlainTextResponse("", status_code=404)
        return PlainTextResponse(Path(path).read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")

    @app.post("/api/notes/{class_id}")
    async def notes_regen(class_id: int):
        notes = state["svc"].get("notes")
        notes.done.pop(class_id, None)
        path = await notes.generate_once(class_id)
        return {"ok": bool(path), "path": path}

    @app.post("/api/mini")
    async def open_mini():
        """Opens the mini page in a small always-on-top native window (pywebview)."""
        import subprocess
        import sys
        url = f"http://{app.state.host}:{app.state.port}/mini"
        try:
            subprocess.Popen([sys.executable, str(ROOT / "mini_window.py"), url])
            return {"ok": True}
        except OSError as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/stop")
    async def stop():
        state["pipeline"].stop()
        return {"ok": True}

    def ready():
        p = state["pipeline"]
        return state["svc"] and p is not None and p.class_id is not None

    async def answer(sock, question, scope):
        if not ready():
            await hub.send(sock, {"type": "answer_start", "scope": scope, "sources": 0})
            await hub.send(sock, {"type": "answer_done", "error": "The class has not started yet."})
            return
        try:
            await state["svc"]["qa"].ask(question, scope, state["pipeline"], lambda ev: hub.send(sock, ev))
        except Exception as e:  # never leave the page waiting
            await hub.send(sock, {"type": "answer_done", "error": str(e)})

    async def handle(sock, msg):
        kind = msg.get("type")
        p = state["pipeline"]
        if kind == "ask" and msg.get("question", "").strip():
            scope = msg.get("scope") if msg.get("scope") in SCOPES else "class"
            await answer(sock, msg["question"].strip()[:500], scope)
            return
        if not ready():
            await hub.send(sock, {"type": "error", "text": "The class has not started yet."})
            return
        doubts = state["svc"]["doubts"]
        if kind == "doubt_add" and msg.get("text", "").strip():
            await doubts.add(msg["text"].strip()[:400], p)
        elif kind == "doubt_recheck":
            await doubts.check(int(msg["id"]), p)
        elif kind == "doubt_dismiss":
            await doubts.dismiss(int(msg["id"]), p)
        elif kind == "suggest":
            qs, err = await doubts.suggest(p)
            await hub.send(sock, {"type": "suggestions", "questions": qs, "error": err})
        elif kind == "alert_feedback":
            await state["svc"]["alerts"].feedback(int(msg["id"]), int(msg["value"]))
        elif kind in ("rule_toggle", "rule_add", "rule_delete"):
            alerts = state["svc"]["alerts"]
            if kind == "rule_toggle":
                await asyncio.to_thread(p.store.update_rule, msg["key"], enabled=int(bool(msg.get("enabled"))))
            elif kind == "rule_add" and msg.get("description", "").strip():
                await asyncio.to_thread(alerts.add_rule, msg.get("name") or msg["description"][:40],
                                        msg["description"].strip(), msg.get("kind", "important"))
            elif kind == "rule_delete":
                await asyncio.to_thread(p.store.delete_rule, msg["key"])
            await asyncio.to_thread(alerts.reload)
            await hub.send_all({"type": "alert_rules", "rules": alerts.rules_view()})
        elif kind == "profile_set":
            style = {"address": msg.get("address", "informal"), "length": msg.get("length", "short")}
            await asyncio.to_thread(p.store.set_profile, p.subject["id"], msg.get("concerns", "")[:1500], style)
            await hub.send_all({"type": "profile", **await asyncio.to_thread(p.store.profile, p.subject["id"])})

    async def safe_handle(sock, msg):
        try:
            await handle(sock, msg)
        except Exception as e:
            await hub.send(sock, {"type": "error", "text": str(e)})

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
            if state["svc"]:
                for d in await asyncio.to_thread(state["svc"]["doubts"].views, p):
                    await hub.send(sock, {"type": "doubt", **d})
                alerts = state["svc"]["alerts"]
                await hub.send(sock, {"type": "alert_rules", "rules": alerts.rules_view()})
                for a in await asyncio.to_thread(p.store.alerts, p.class_id):
                    await hub.send(sock, alerts.view(a))
            await hub.send(sock, {"type": "profile", **p.store.profile(p.subject["id"])})
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                asyncio.create_task(safe_handle(sock, msg))
        except (WebSocketDisconnect, json.JSONDecodeError):
            pass
        finally:
            hub.clients.pop(sock, None)

    return app
