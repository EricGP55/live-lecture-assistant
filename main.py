#!/usr/bin/env python3
"""Live class assistant: live transcription, who is speaking, and questions to the AI.

Usage:
  python main.py --subject "Calculus"                     # default microphone
  python main.py --subject "Calculus" --device "USB"
  python main.py --devices                                # list microphones
  python main.py --subject Demo --file tests/class_sample.wav   # simulate a class
"""
import argparse
import subprocess
import threading

import uvicorn

from app.audio import list_devices
from app.config import load
from app.embeddings import Embedder
from app.pipeline import Pipeline
from app.qa import QA
from app.search import Retriever
from app.server import Hub, create_app
from app.store import Store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", help="subject name (created if it does not exist)")
    ap.add_argument("--device", help="microphone: index or part of its name (default: config.yaml)")
    ap.add_argument("--devices", action="store_true", help="list available microphones and exit")
    ap.add_argument("--file", help="use an audio file instead of the microphone (testing)")
    ap.add_argument("--fast", action="store_true", help="with --file: process without waiting for real time")
    ap.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    a = ap.parse_args()

    if a.devices:
        for d in list_devices():
            print(f"[{d['index']}] {d['name']}  ({d['channels']} channels)")
        return
    if not a.subject:
        ap.error("--subject is required")

    cfg = load()
    if a.device is not None:
        cfg["audio"]["device"] = a.device
    store = Store(cfg["data"] / "class.db")
    hub = Hub()
    embedder = Embedder()
    app = create_app(
        hub,
        lambda: Pipeline(cfg, store, a.subject, hub.emit, audio_file=a.file, fast=a.fast,
                         embedder=embedder),
        lambda: QA(cfg, Retriever(store, embedder)))
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    url = f"http://{host}:{port}/"
    print(f"Opening {url}  (mini: {url}mini) · Ctrl+C to end the class")
    if not a.no_browser:
        threading.Timer(1.5, lambda: subprocess.run(["open", url])).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
