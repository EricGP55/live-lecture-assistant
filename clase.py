#!/usr/bin/env python3
"""Asistente de clase en vivo — fase 1: transcripción en directo + quién habla.

Uso:
  python clase.py --asignatura "Cálculo"                # micro por defecto
  python clase.py --asignatura "Cálculo" --dispositivo "USB"
  python clase.py --dispositivos                        # lista los micros
  python clase.py --asignatura Prueba --archivo tests/clase_prueba.wav   # simula una clase
"""
import argparse
import subprocess
import threading

import uvicorn

from app.audio import listar_dispositivos
from app.config import cargar
from app.pipeline import Pipeline
from app.server import Hub, crear_app
from app.store import Store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asignatura", help="nombre de la asignatura (se crea si no existe)")
    ap.add_argument("--dispositivo", help="micro: índice o parte del nombre (por defecto, el de config.yaml)")
    ap.add_argument("--dispositivos", action="store_true", help="lista los micros disponibles y sale")
    ap.add_argument("--archivo", help="usar un archivo de audio en lugar del micro (pruebas)")
    ap.add_argument("--rapido", action="store_true", help="con --archivo: procesar sin esperar al tiempo real")
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir la página automáticamente")
    a = ap.parse_args()

    if a.dispositivos:
        for d in listar_dispositivos():
            print(f"[{d['indice']}] {d['nombre']}  ({d['canales']} canales)")
        return
    if not a.asignatura:
        ap.error("falta --asignatura")

    cfg = cargar()
    if a.dispositivo is not None:
        cfg["audio"]["dispositivo"] = a.dispositivo
    store = Store(cfg["datos"] / "clase.db")
    hub = Hub()
    app = crear_app(hub, lambda: Pipeline(cfg, store, a.asignatura, hub.emitir,
                                          archivo=a.archivo, rapido=a.rapido))
    host, puerto = cfg["servidor"]["host"], cfg["servidor"]["puerto"]
    url = f"http://{host}:{puerto}/"
    print(f"Abriendo {url}  (mini: {url}mini) · Ctrl+C para terminar la clase")
    if not a.sin_navegador:
        threading.Timer(1.5, lambda: subprocess.run(["open", url])).start()
    uvicorn.run(app, host=host, port=puerto, log_level="warning")


if __name__ == "__main__":
    main()
