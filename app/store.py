"""SQLite store: subjects -> classes -> sentences (+ FTS5 for full-text search)."""
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    concerns_profile TEXT DEFAULT '',
    question_style TEXT DEFAULT '{"address":"informal","length":"short"}',
    teacher_voiceprint BLOB,
    voiceprint_n INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    started_at REAL NOT NULL,
    ended_at REAL,
    audio_path TEXT,
    language TEXT
);
CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT NOT NULL,           -- 'T' teacher, 'S' student, '?' unknown
    p_teacher REAL,
    cluster INTEGER,
    embedding BLOB                   -- text embedding (phase 2)
);
CREATE INDEX IF NOT EXISTS sentences_class ON sentences(class_id, t_start);
CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts USING fts5(text, content='sentences', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS sentences_ai AFTER INSERT ON sentences BEGIN
    INSERT INTO sentences_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS sentences_au AFTER UPDATE OF text ON sentences BEGIN
    INSERT INTO sentences_fts(sentences_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO sentences_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _query(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # --- subjects
    def subject(self, name: str) -> dict:
        self._exec("INSERT OR IGNORE INTO subjects(name) VALUES (?)", (name,))
        return self._query("SELECT * FROM subjects WHERE name=?", (name,))[0]

    def teacher_voiceprint(self, subject_id: int) -> np.ndarray | None:
        r = self._query("SELECT teacher_voiceprint FROM subjects WHERE id=?", (subject_id,))
        b = r[0]["teacher_voiceprint"] if r else None
        return np.frombuffer(b, dtype=np.float32).copy() if b else None

    def save_teacher_voiceprint(self, subject_id: int, centroid: np.ndarray, n: int):
        """Weighted average with the previous voiceprint, so it improves class after class."""
        r = self._query("SELECT teacher_voiceprint, voiceprint_n FROM subjects WHERE id=?",
                        (subject_id,))[0]
        c = centroid.astype(np.float32)
        if r["teacher_voiceprint"]:
            prev = np.frombuffer(r["teacher_voiceprint"], dtype=np.float32)
            pn = min(r["voiceprint_n"], 200)
            c = (prev * pn + c * n) / (pn + n)
        c = c / (np.linalg.norm(c) + 1e-9)
        self._exec("UPDATE subjects SET teacher_voiceprint=?, voiceprint_n=? WHERE id=?",
                   (c.astype(np.float32).tobytes(), min(r["voiceprint_n"] + n, 200), subject_id))

    # --- classes
    def new_class(self, subject_id: int, language: str) -> int:
        return self._exec("INSERT INTO classes(subject_id, started_at, language) VALUES (?,?,?)",
                          (subject_id, time.time(), language)).lastrowid

    def set_audio(self, class_id: int, path: str):
        self._exec("UPDATE classes SET audio_path=? WHERE id=?", (path, class_id))

    def close_class(self, class_id: int):
        self._exec("UPDATE classes SET ended_at=? WHERE id=?", (time.time(), class_id))

    # --- sentences
    def new_sentence(self, class_id, t0, t1, text, speaker, p_teacher, cluster) -> int:
        return self._exec(
            "INSERT INTO sentences(class_id,t_start,t_end,text,speaker,p_teacher,cluster) "
            "VALUES (?,?,?,?,?,?,?)",
            (class_id, t0, t1, text, speaker, p_teacher, cluster)).lastrowid

    def set_speaker(self, sentence_id: int, speaker: str, p_teacher: float):
        self._exec("UPDATE sentences SET speaker=?, p_teacher=? WHERE id=?",
                   (speaker, p_teacher, sentence_id))

    def sentences(self, class_id: int) -> list[dict]:
        return self._query("SELECT id,t_start,t_end,text,speaker,p_teacher FROM sentences "
                           "WHERE class_id=? ORDER BY t_start", (class_id,))
