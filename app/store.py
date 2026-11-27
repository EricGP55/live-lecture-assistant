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
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    class_id INTEGER REFERENCES classes(id),
    asked_at REAL NOT NULL,
    t_class REAL,                    -- class time (s) when it was asked
    scope TEXT NOT NULL,             -- 'class' | 'subject' | 'all'
    question TEXT NOT NULL,
    answer TEXT,
    sources TEXT,                    -- JSON list of sentence ids used as context
    tokens_in INTEGER,
    tokens_out INTEGER
);
CREATE TABLE IF NOT EXISTS doubts (
    id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    class_id INTEGER REFERENCES classes(id),       -- class where it was written
    created_at REAL NOT NULL,
    t_class REAL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',        -- pending | partial | answered | dismissed
    answer TEXT DEFAULT '',
    evidence TEXT DEFAULT '[]',                    -- JSON list of "mm:ss"
    missing TEXT DEFAULT '',
    questions TEXT DEFAULT '[]',                   -- JSON list of suggested questions
    answered_class_id INTEGER,
    answered_t REAL,
    embedding BLOB,
    tokens INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS doubts_subject ON doubts(subject_id, status);
CREATE TABLE IF NOT EXISTS alert_rules (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,          -- 'exam', 'emphasis', ... or 'custom_<n>'
    kind TEXT NOT NULL,                -- 'important' | 'complex'
    name TEXT NOT NULL,
    description TEXT NOT NULL,         -- also used as the semantic query for custom rules
    enabled INTEGER NOT NULL DEFAULT 1,
    threshold REAL NOT NULL,           -- adjusted by 👍/👎
    builtin INTEGER NOT NULL DEFAULT 0,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS alert_examples (
    id INTEGER PRIMARY KEY,
    rule_key TEXT NOT NULL,
    polarity INTEGER NOT NULL,         -- +1 useful alert, -1 useless alert
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    sentence_id INTEGER REFERENCES sentences(id),
    t REAL NOT NULL,
    kind TEXT NOT NULL,                -- 'important' | 'complex'
    rules TEXT NOT NULL,               -- JSON list of rule keys that fired
    reason TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'rule' | 'semantic+claude' | 'repeat' | 'local'
    feedback INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_class ON alerts(class_id, t);
"""

DOUBT_FIELDS = ("status", "answer", "evidence", "missing", "questions", "answered_class_id",
                "answered_t", "embedding", "tokens")

SCOPE_SQL = {
    "class": "s.class_id = :class_id",
    "subject": "c.subject_id = :subject_id",
    "all": "1",
}


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

    def set_embedding(self, sentence_id: int, vec: np.ndarray):
        self._exec("UPDATE sentences SET embedding=? WHERE id=?",
                   (vec.astype(np.float32).tobytes(), sentence_id))

    # --- search
    def scope_rows(self, scope: str, class_id: int, subject_id: int) -> list[dict]:
        """Sentences in scope, with class date, for retrieval."""
        return self._query(
            "SELECT s.id, s.class_id, s.t_start, s.speaker, s.text, s.embedding, c.started_at "
            "FROM sentences s JOIN classes c ON c.id = s.class_id "
            f"WHERE {SCOPE_SQL[scope]} ORDER BY s.class_id, s.t_start",
            {"class_id": class_id, "subject_id": subject_id})

    def fts(self, match: str, scope: str, class_id: int, subject_id: int, limit=20) -> list[int]:
        """Sentence ids ranked by BM25."""
        try:
            rows = self._query(
                "SELECT s.id FROM sentences_fts f JOIN sentences s ON s.id = f.rowid "
                "JOIN classes c ON c.id = s.class_id "
                f"WHERE sentences_fts MATCH :m AND {SCOPE_SQL[scope]} ORDER BY bm25(sentences_fts) LIMIT :n",
                {"m": match, "class_id": class_id, "subject_id": subject_id, "n": limit})
        except sqlite3.OperationalError:
            return []
        return [r["id"] for r in rows]

    # --- questions
    def save_question(self, class_id, t_class, scope, question, answer, sources, tokens_in, tokens_out):
        import json
        return self._exec(
            "INSERT INTO questions(class_id,asked_at,t_class,scope,question,answer,sources,tokens_in,tokens_out) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (class_id, time.time(), t_class, scope, question, answer, json.dumps(sources),
             tokens_in, tokens_out)).lastrowid

    def questions(self, class_id: int) -> list[dict]:
        return self._query("SELECT id, question, answer, scope FROM questions WHERE class_id=? ORDER BY id",
                           (class_id,))

    # --- concerns profile and question style (per subject)
    def set_profile(self, subject_id: int, concerns: str, style: dict):
        import json
        self._exec("UPDATE subjects SET concerns_profile=?, question_style=? WHERE id=?",
                   (concerns, json.dumps(style), subject_id))

    def profile(self, subject_id: int) -> dict:
        import json
        r = self._query("SELECT concerns_profile, question_style FROM subjects WHERE id=?", (subject_id,))[0]
        return {"concerns": r["concerns_profile"] or "", "style": json.loads(r["question_style"] or "{}")}

    # --- doubts
    def new_doubt(self, subject_id, class_id, t_class, text, embedding) -> int:
        return self._exec(
            "INSERT INTO doubts(subject_id,class_id,created_at,t_class,text,embedding) VALUES (?,?,?,?,?,?)",
            (subject_id, class_id, time.time(), t_class, text,
             embedding.astype(np.float32).tobytes())).lastrowid

    def update_doubt(self, doubt_id: int, **fields):
        cols = [k for k in fields if k in DOUBT_FIELDS]
        if cols:
            self._exec(f"UPDATE doubts SET {', '.join(c + '=?' for c in cols)} WHERE id=?",
                       (*[fields[c] for c in cols], doubt_id))

    def doubt(self, doubt_id: int) -> dict | None:
        r = self._query("SELECT * FROM doubts WHERE id=?", (doubt_id,))
        return r[0] if r else None

    def open_doubts(self, subject_id: int, class_id: int) -> list[dict]:
        """Doubts of this class, plus unresolved ones from earlier classes of the subject."""
        return self._query(
            "SELECT * FROM doubts WHERE subject_id=? AND (class_id=? OR status IN ('pending','partial')) "
            "ORDER BY id", (subject_id, class_id))

    # --- alerts
    def alert_rules(self) -> list[dict]:
        return self._query("SELECT * FROM alert_rules ORDER BY builtin DESC, id")

    def upsert_rule(self, key, kind, name, description, threshold, builtin, embedding=None):
        self._exec(
            "INSERT INTO alert_rules(key,kind,name,description,threshold,builtin,embedding) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET kind=excluded.kind, name=excluded.name, description=excluded.description, "
            "embedding=COALESCE(excluded.embedding, alert_rules.embedding)",
            (key, kind, name, description, threshold, int(builtin),
             embedding.astype(np.float32).tobytes() if embedding is not None else None))

    def update_rule(self, key, **fields):
        cols = [k for k in fields if k in ("enabled", "threshold")]
        if cols:
            self._exec(f"UPDATE alert_rules SET {', '.join(c + '=?' for c in cols)} WHERE key=?",
                       (*[fields[c] for c in cols], key))

    def delete_rule(self, key):
        self._exec("DELETE FROM alert_rules WHERE key=? AND builtin=0", (key,))
        self._exec("DELETE FROM alert_examples WHERE rule_key=?", (key,))

    def add_example(self, rule_key, polarity, text, embedding):
        self._exec("INSERT INTO alert_examples(rule_key,polarity,text,embedding,created_at) VALUES (?,?,?,?,?)",
                   (rule_key, polarity, text, embedding.astype(np.float32).tobytes(), time.time()))

    def examples(self) -> list[dict]:
        return self._query("SELECT rule_key, polarity, text, embedding FROM alert_examples ORDER BY id")

    def new_alert(self, class_id, sentence_id, t, kind, rules, reason, source) -> int:
        import json
        return self._exec(
            "INSERT INTO alerts(class_id,sentence_id,t,kind,rules,reason,source,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (class_id, sentence_id, t, kind, json.dumps(rules), reason, source, time.time())).lastrowid

    def alert(self, alert_id: int) -> dict | None:
        r = self._query("SELECT a.*, s.text FROM alerts a LEFT JOIN sentences s ON s.id=a.sentence_id WHERE a.id=?",
                        (alert_id,))
        return r[0] if r else None

    def set_feedback(self, alert_id: int, value: int):
        self._exec("UPDATE alerts SET feedback=? WHERE id=?", (value, alert_id))

    def alerts(self, class_id: int) -> list[dict]:
        return self._query("SELECT a.*, s.text FROM alerts a LEFT JOIN sentences s ON s.id=a.sentence_id "
                           "WHERE a.class_id=? ORDER BY a.t", (class_id,))

    def sentences_between(self, class_id: int, t0: float, t1: float) -> list[dict]:
        return self._query("SELECT id,t_start,speaker,text FROM sentences "
                           "WHERE class_id=? AND t_start>=? AND t_start<=? ORDER BY t_start",
                           (class_id, t0, t1))

    def sentences(self, class_id: int) -> list[dict]:
        return self._query("SELECT id,t_start,t_end,text,speaker,p_teacher FROM sentences "
                           "WHERE class_id=? ORDER BY t_start", (class_id,))
