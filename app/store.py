"""Almacén SQLite: asignaturas -> clases -> frases (+ FTS5 para búsqueda por texto)."""
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

ESQUEMA = """
CREATE TABLE IF NOT EXISTS asignaturas (
    id INTEGER PRIMARY KEY,
    nombre TEXT UNIQUE NOT NULL,
    perfil_inquietudes TEXT DEFAULT '',
    estilo_preguntas TEXT DEFAULT '{"trato":"tu","longitud":"corta"}',
    huella_profe BLOB,
    huella_n INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS clases (
    id INTEGER PRIMARY KEY,
    asignatura_id INTEGER NOT NULL REFERENCES asignaturas(id),
    inicio REAL NOT NULL,
    fin REAL,
    audio_path TEXT,
    idioma TEXT
);
CREATE TABLE IF NOT EXISTS frases (
    id INTEGER PRIMARY KEY,
    clase_id INTEGER NOT NULL REFERENCES clases(id),
    t_inicio REAL NOT NULL,
    t_fin REAL NOT NULL,
    texto TEXT NOT NULL,
    hablante TEXT NOT NULL,          -- 'P' profesor, 'A' alumno, '?' desconocido
    p_profe REAL,
    cluster INTEGER,
    embedding BLOB                   -- embedding de texto (fase 2)
);
CREATE INDEX IF NOT EXISTS frases_clase ON frases(clase_id, t_inicio);
CREATE VIRTUAL TABLE IF NOT EXISTS frases_fts USING fts5(texto, content='frases', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS frases_ai AFTER INSERT ON frases BEGIN
    INSERT INTO frases_fts(rowid, texto) VALUES (new.id, new.texto);
END;
CREATE TRIGGER IF NOT EXISTS frases_au AFTER UPDATE OF texto ON frases BEGIN
    INSERT INTO frases_fts(frases_fts, rowid, texto) VALUES ('delete', old.id, old.texto);
    INSERT INTO frases_fts(rowid, texto) VALUES (new.id, new.texto);
END;
"""


class Store:
    def __init__(self, ruta: Path):
        self.db = sqlite3.connect(ruta, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(ESQUEMA)
        self.lock = threading.Lock()

    def _x(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # --- asignaturas
    def asignatura(self, nombre: str) -> dict:
        self._x("INSERT OR IGNORE INTO asignaturas(nombre) VALUES (?)", (nombre,))
        return self._q("SELECT * FROM asignaturas WHERE nombre=?", (nombre,))[0]

    def huella_profe(self, asignatura_id: int) -> np.ndarray | None:
        r = self._q("SELECT huella_profe FROM asignaturas WHERE id=?", (asignatura_id,))
        b = r[0]["huella_profe"] if r else None
        return np.frombuffer(b, dtype=np.float32).copy() if b else None

    def guardar_huella_profe(self, asignatura_id: int, centroide: np.ndarray, n: int):
        """Media ponderada con la huella previa, para que mejore clase a clase."""
        r = self._q("SELECT huella_profe, huella_n FROM asignaturas WHERE id=?", (asignatura_id,))[0]
        c = centroide.astype(np.float32)
        if r["huella_profe"]:
            prev = np.frombuffer(r["huella_profe"], dtype=np.float32)
            pn = min(r["huella_n"], 200)
            c = (prev * pn + c * n) / (pn + n)
        c = c / (np.linalg.norm(c) + 1e-9)
        self._x("UPDATE asignaturas SET huella_profe=?, huella_n=? WHERE id=?",
                (c.astype(np.float32).tobytes(), min(r["huella_n"] + n, 200), asignatura_id))

    # --- clases
    def nueva_clase(self, asignatura_id: int, idioma: str) -> int:
        return self._x("INSERT INTO clases(asignatura_id, inicio, idioma) VALUES (?,?,?)",
                       (asignatura_id, time.time(), idioma)).lastrowid

    def set_audio(self, clase_id: int, ruta: str):
        self._x("UPDATE clases SET audio_path=? WHERE id=?", (ruta, clase_id))

    def cerrar_clase(self, clase_id: int):
        self._x("UPDATE clases SET fin=? WHERE id=?", (time.time(), clase_id))

    # --- frases
    def nueva_frase(self, clase_id, t0, t1, texto, hablante, p_profe, cluster) -> int:
        return self._x(
            "INSERT INTO frases(clase_id,t_inicio,t_fin,texto,hablante,p_profe,cluster) VALUES (?,?,?,?,?,?,?)",
            (clase_id, t0, t1, texto, hablante, p_profe, cluster)).lastrowid

    def set_hablante(self, frase_id: int, hablante: str, p_profe: float):
        self._x("UPDATE frases SET hablante=?, p_profe=? WHERE id=?", (hablante, p_profe, frase_id))

    def frases(self, clase_id: int) -> list[dict]:
        return self._q("SELECT id,t_inicio,t_fin,texto,hablante,p_profe FROM frases "
                       "WHERE clase_id=? ORDER BY t_inicio", (clase_id,))
