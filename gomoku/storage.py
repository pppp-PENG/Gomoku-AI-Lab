import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def data_root():
    path = Path(os.environ.get("GOMOKU_DATA", "data")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


class Store:
    def __init__(self, root=None):
        self.root = Path(root) if root else data_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lab.sqlite"
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            INSERT OR IGNORE INTO meta VALUES ('schema_version','1');
            CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created REAL, status TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, created REAL, experiment TEXT, path TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, experiment TEXT, step INTEGER, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS demonstrations (id TEXT PRIMARY KEY, created REAL,
                fingerprint TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS metrics_experiment ON metrics(experiment, step);
            """)
            if db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] != '1':
                raise RuntimeError("unsupported database schema")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_experiment(self, config):
        eid = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO experiments VALUES (?,?,?,?)",
                       (eid, time.time(), "created", json.dumps(config)))
        return eid

    def status(self, eid, status):
        with self.db() as db:
            db.execute("UPDATE experiments SET status=? WHERE id=?", (status, eid))

    def experiment(self, eid):
        with self.db() as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
        if not row:
            raise KeyError(eid)
        return {**dict(row), "config": json.loads(row["payload"])}

    def experiments(self):
        with self.db() as db:
            rows = db.execute("SELECT id FROM experiments ORDER BY created DESC").fetchall()
        return [self.experiment(r[0]) for r in rows]

    def metric(self, eid, step, payload):
        with self.db() as db:
            db.execute("INSERT INTO metrics(experiment,step,created,payload) VALUES (?,?,?,?)",
                       (eid, step, time.time(), json.dumps(payload)))

    def metrics(self, eid, limit=2000):
        with self.db() as db:
            count = db.execute("SELECT COUNT(*) FROM metrics WHERE experiment=?", (eid,)).fetchone()[0]
            stride = max(1, (count + limit - 1)//limit) if limit else 1
            rows = db.execute("""SELECT step,created,payload FROM metrics WHERE experiment=?
                               AND (id % ? = 0 OR id=(SELECT MAX(id) FROM metrics WHERE experiment=?))
                               ORDER BY id""", (eid, stride, eid)).fetchall()
        return [dict(step=r[0], created=r[1], **json.loads(r[2])) for r in rows]

    def checkpoint(self, cid, eid, path, metadata):
        with self.db() as db:
            db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                       (cid, time.time(), eid, str(path), json.dumps(metadata)))

    def checkpoints(self):
        with self.db() as db:
            rows = db.execute("""SELECT * FROM checkpoints WHERE COALESCE(json_extract(payload,'$.deleted'),0)=0
                                 ORDER BY created DESC""").fetchall()
        return [dict(id=r["id"], created=r["created"], experiment=r["experiment"],
                     path=r["path"], **json.loads(r["payload"])) for r in rows]

    def get_checkpoint(self, cid):
        for ck in self.checkpoints():
            if ck["id"] == cid:
                return ck
        raise KeyError(cid)

    def checkpoint_references(self, cid):
        self.get_checkpoint(cid)
        with self.db() as db:
            games = db.execute("SELECT COUNT(*) FROM games WHERE json_extract(payload,'$.checkpoint')=?", (cid,)).fetchone()[0]
            evaluations = db.execute("""SELECT COUNT(*) FROM evaluations WHERE json_extract(payload,'$.checkpoint')=?
                                        OR json_extract(payload,'$.opponent_id')=?""", (cid, cid)).fetchone()[0]
        return dict(games=games, evaluations=evaluations)

    def delete_checkpoint(self, cid):
        checkpoint = self.get_checkpoint(cid)
        path = Path(checkpoint["path"]).resolve()
        directory = (self.root / "checkpoints").resolve()
        if path.parent != directory or path.name != f"{cid}.pt":
            raise ValueError("refusing to delete a file outside the checkpoint directory")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM checkpoints WHERE id=?", (cid,)).fetchone()
            metadata = json.loads(row[0])
            if metadata.get("deleted"):
                raise KeyError(cid)
            original = row[0]
            metadata.update(deleted=True, deleted_at=time.time())
            db.execute("UPDATE checkpoints SET payload=? WHERE id=?", (json.dumps(metadata), cid))
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            with self.db() as db:
                db.execute("UPDATE checkpoints SET payload=? WHERE id=?", (original, cid))
            raise ValueError(f"无法删除权重文件：{exc}") from exc
        return dict(id=cid, deleted=True)

    def save_game(self, payload, gid=None):
        gid = gid or uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO games VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                       (gid, time.time(), json.dumps(payload)))
        return gid

    def game(self, gid):
        with self.db() as db:
            row = db.execute("SELECT * FROM games WHERE id=?", (gid,)).fetchone()
        if not row:
            raise KeyError(gid)
        return dict(id=gid, created=row["created"], **json.loads(row["payload"]))

    def games(self):
        with self.db() as db:
            rows = db.execute("SELECT id FROM games ORDER BY created DESC LIMIT 500").fetchall()
        return [self.game(r[0]) for r in rows]

    def save_evaluation(self, payload):
        eid = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO evaluations VALUES (?,?,?)", (eid, time.time(), json.dumps(payload)))
        return eid

    def save_demonstration(self, payload):
        """Only validated, explicitly approved terminal games reach this collection."""
        import hashlib
        fingerprint = hashlib.sha256(json.dumps([payload["rules"], payload["size"], payload["moves"]],
                                                separators=(",", ":")).encode()).hexdigest()
        did = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO demonstrations VALUES (?,?,?,?)",
                       (did, time.time(), fingerprint, json.dumps(payload)))
            did = db.execute("SELECT id FROM demonstrations WHERE fingerprint=?", (fingerprint,)).fetchone()[0]
        return self.demonstration(did)

    def demonstration(self, did):
        with self.db() as db:
            row = db.execute("SELECT * FROM demonstrations WHERE id=?", (did,)).fetchone()
        if not row:
            raise KeyError(did)
        return dict(id=did, created=row["created"], **json.loads(row["payload"]))

    def demonstrations(self, size=None):
        with self.db() as db:
            rows = db.execute("""SELECT id FROM demonstrations
                                 WHERE (? IS NULL OR json_extract(payload,'$.size')=?)
                                 ORDER BY created DESC""", (size, size)).fetchall()
        return [self.demonstration(row[0]) for row in rows]

    def evaluations(self):
        with self.db() as db:
            rows = db.execute("SELECT * FROM evaluations ORDER BY created DESC LIMIT 500").fetchall()
        return [dict(id=r["id"], created=r["created"], **json.loads(r["payload"])) for r in rows]
