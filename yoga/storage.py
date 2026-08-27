"""Session logging (Methodology Stage 4 'progress dashboard', Stage 6 analytics).

Privacy-first, as the methodology requires: **no frame is ever written to
disk**.  Only derived numbers - hold durations, alignment scores, joint angles
and the cues that were spoken - go into a local SQLite file.  That is enough to
build the weekly progress report and, later, to join against a vitals log for
the practice-vs-resting-heart-rate correlation.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions.db"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asana           TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    frames          INTEGER DEFAULT 0,
    avg_fps         REAL,
    best_score      REAL,
    completed_holds INTEGER DEFAULT 0,
    total_hold_s    REAL    DEFAULT 0,
    longest_hold_s  REAL    DEFAULT 0,
    source          TEXT
);
CREATE TABLE IF NOT EXISTS holds (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    started_s   REAL,
    duration_s  REAL,
    avg_score   REAL,
    best_score  REAL,
    completed   INTEGER
);
CREATE TABLE IF NOT EXISTS cues (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    t_s         REAL,
    joint       TEXT,
    level       TEXT,
    text        TEXT
);
CREATE TABLE IF NOT EXISTS joint_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    t_s         REAL,
    joint       TEXT,
    value       REAL,
    deviation   REAL,
    in_tolerance INTEGER
);
CREATE INDEX IF NOT EXISTS idx_holds_session ON holds(session_id);
CREATE INDEX IF NOT EXISTS idx_samples_session ON joint_samples(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class SessionLog:
    def __init__(self, asana_key: str, db_path: str = DEFAULT_DB, source: str = "webcam") -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        cur = self.conn.execute(
            "INSERT INTO sessions (asana, started_at, source) VALUES (?,?,?)",
            (asana_key, _now(), source),
        )
        self.session_id = int(cur.lastrowid)
        self.conn.commit()

    def log_cue(self, t: float, joint: str, level: str, text: str) -> None:
        self.conn.execute(
            "INSERT INTO cues (session_id, t_s, joint, level, text) VALUES (?,?,?,?,?)",
            (self.session_id, t, joint, level, text),
        )

    def log_joints(self, t: float, results) -> None:
        rows = [(self.session_id, t, r.check.key, r.value, r.deviation, int(bool(r.ok)))
                for r in results if r.ok is not None]
        if rows:
            self.conn.executemany(
                "INSERT INTO joint_samples (session_id, t_s, joint, value, deviation, in_tolerance)"
                " VALUES (?,?,?,?,?,?)", rows)

    def finish(self, stats, frames: int, avg_fps: float) -> None:
        for h in stats.holds:
            self.conn.execute(
                "INSERT INTO holds (session_id, started_s, duration_s, avg_score, best_score, completed)"
                " VALUES (?,?,?,?,?,?)",
                (self.session_id, h.started_at, h.duration_s, h.avg_score, h.best_score, int(h.completed)),
            )
        self.conn.execute(
            "UPDATE sessions SET ended_at=?, frames=?, avg_fps=?, best_score=?,"
            " completed_holds=?, total_hold_s=?, longest_hold_s=? WHERE id=?",
            (_now(), frames, avg_fps, stats.best_score, stats.completed,
             stats.total_hold_s, stats.longest_s, self.session_id),
        )
        self.conn.commit()
        self.conn.close()


def recent_sessions(db_path: str = DEFAULT_DB, limit: int = 10) -> list[dict]:
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM sessions WHERE ended_at IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
