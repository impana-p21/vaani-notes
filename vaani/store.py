"""SQLite storage + cosine-similarity semantic search. Everything stays on disk locally."""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np


class NoteStore:
    def __init__(self, path: str = "vaani.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS notes(
          id INTEGER PRIMARY KEY, title TEXT, created REAL, language TEXT,
          transcript TEXT, summary_json TEXT, audio_seconds REAL, process_seconds REAL);
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY, note_id INTEGER, start REAL, text TEXT, vec BLOB);
        """)

    def add_note(self, title, language, transcript, summary, audio_seconds, process_seconds, chunks):
        cur = self.db.execute(
            "INSERT INTO notes(title,created,language,transcript,summary_json,audio_seconds,process_seconds)"
            " VALUES(?,?,?,?,?,?,?)",
            (title, time.time(), language, transcript, json.dumps(summary, ensure_ascii=False),
             audio_seconds, process_seconds))
        nid = cur.lastrowid
        self.db.executemany(
            "INSERT INTO chunks(note_id,start,text,vec) VALUES(?,?,?,?)",
            [(nid, s, t, v.astype(np.float32).tobytes()) for s, t, v in chunks])
        self.db.commit()
        return nid

    def list_notes(self):
        rows = self.db.execute(
            "SELECT id,title,created,language,audio_seconds,process_seconds FROM notes ORDER BY id DESC").fetchall()
        return [dict(zip(("id", "title", "created", "language", "audio_seconds", "process_seconds"), r)) for r in rows]

    def get_note(self, nid: int):
        r = self.db.execute(
            "SELECT id,title,created,language,transcript,summary_json,audio_seconds,process_seconds"
            " FROM notes WHERE id=?", (nid,)).fetchone()
        if not r:
            return None
        d = dict(zip(("id", "title", "created", "language", "transcript", "summary", "audio_seconds",
                      "process_seconds"), r))
        d["summary"] = json.loads(d["summary"])
        return d

    def delete_note(self, nid: int):
        self.db.execute("DELETE FROM chunks WHERE note_id=?", (nid,))
        self.db.execute("DELETE FROM notes WHERE id=?", (nid,))
        self.db.commit()

    def search(self, qvec: np.ndarray, k: int = 5):
        rows = self.db.execute(
            "SELECT c.note_id, n.title, c.start, c.text, c.vec FROM chunks c JOIN notes n ON n.id=c.note_id"
        ).fetchall()
        if not rows:
            return []
        mat = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        sims = mat @ qvec.ravel()
        return [{"note_id": rows[i][0], "title": rows[i][1], "start": rows[i][2],
                 "text": rows[i][3], "score": float(sims[i])} for i in np.argsort(-sims)[:k]]
