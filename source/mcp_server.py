import json
import logging
import os
import re
import sys
import subprocess
import importlib.util
import sqlite3
import webbrowser
import shutil
import threading
import time
import tempfile
import atexit
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

def install_missing_dependencies(dependencies: dict[str, str], context: str = "One-Click Install") -> bool:
    """Install missing Python dependencies. Returns False if installation fails."""
    missing = []
    for import_name, pip_name in dependencies.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)

    if not missing:
        return True

    print(f"{context}: Missing dependencies detected: {', '.join(missing)}", file=sys.stderr)
    print("Installing now... (this may take a minute on the first run)", file=sys.stderr)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("Installation complete! Continuing...", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Error: Failed to install dependencies: {e}", file=sys.stderr)
        return False


def check_and_install_dependencies():
    """Check startup dependencies and install them if missing."""
    dependencies = {
        "mcp": "mcp",
        "numpy": "numpy"
    }
    if not install_missing_dependencies(dependencies):
        print("Please try running 'pip install mcp numpy' manually.", file=sys.stderr)
        sys.exit(1)

# Run dependency check before anything else
check_and_install_dependencies()

from mcp.server.fastmcp import FastMCP
import numpy as np

# Configure logging to stderr (NOT stdout!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("claude-memory")

# Get memories directory from environment or use default
raw_dir = os.environ.get('CLAUDE_MEMORIES_DIR')
if raw_dir:
    raw_dir = raw_dir.replace('${HOME}', str(Path.home()))
    MEMORIES_DIR = Path(raw_dir).expanduser().resolve()
else:
    MEMORIES_DIR = Path.home() / '.claude-memories'
MODEL_NAME = os.environ.get('CLAUDE_MEMORY_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
ALLOW_MODEL_DOWNLOAD = os.environ.get('CLAUDE_MEMORY_ALLOW_MODEL_DOWNLOAD', '').lower() in {'1', 'true', 'yes', 'on'}
WARM_EMBEDDINGS_ON_STARTUP = os.environ.get('CLAUDE_MEMORY_WARM_EMBEDDINGS_ON_STARTUP', '1').lower() not in {'0', 'false', 'no', 'off'}
EMBEDDING_WARMUP_DELAY_SECONDS = float(os.environ.get('CLAUDE_MEMORY_EMBEDDING_WARMUP_DELAY_SECONDS', '10'))
EMBEDDING_WARMUP_TIMEOUT_SECONDS = float(os.environ.get('CLAUDE_MEMORY_EMBEDDING_WARMUP_TIMEOUT_SECONDS', '120'))
model: Optional[object] = None
embedding_worker: Optional[object] = None
model_error: Optional[str] = None
embedding_status = 'cold'
embedding_started_at: Optional[float] = None
embedding_finished_at: Optional[float] = None
embedding_lock = threading.Lock()
embedding_warmup_thread: Optional[threading.Thread] = None

# Ensure memories directory exists
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Using memories directory: {MEMORIES_DIR}")


# ============================================================
# SQLite Database Layer
# ============================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    type            TEXT NOT NULL DEFAULT 'fact',
    importance      INTEGER NOT NULL DEFAULT 5,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT,
    supersedes      INTEGER,
    journal_file    TEXT,
    date            TEXT NOT NULL DEFAULT (datetime('now')),
    embedding       BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type, date DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);

CREATE TABLE IF NOT EXISTS journal_entries (
    id               TEXT PRIMARY KEY,
    author           TEXT NOT NULL DEFAULT 'Pulse',
    title            TEXT,
    entry_type       TEXT NOT NULL,
    content          TEXT NOT NULL,
    why_it_mattered  TEXT,
    tags             TEXT NOT NULL DEFAULT '[]',
    importance       INTEGER NOT NULL DEFAULT 5,
    pinned           INTEGER NOT NULL DEFAULT 0,
    resolved         INTEGER,
    date             TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entries(entry_type, date DESC);
"""


class MemoryDatabase:
    """SQLite database for the shared memory system."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        logger.info(f"Database opened: {self.db_path}")

    def close(self):
        self.conn.close()

    # â”€â”€ Memories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def save_memory(self, text: str, tags: list[str] | None = None,
                    type: str = "fact", importance: int = 5,
                    embedding: Optional[bytes] = None,
                    supersedes: Optional[int] = None,
                    journal_file: Optional[str] = None,
                    date: Optional[str] = None) -> int:
        tags_json = json.dumps(tags or [])
        if date:
            cur = self.conn.execute(
                "INSERT INTO memories (text, tags, type, importance, embedding, "
                "supersedes, journal_file, date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (text, tags_json, type, importance, embedding,
                 supersedes, journal_file, date)
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO memories (text, tags, type, importance, embedding, "
                "supersedes, journal_file) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (text, tags_json, type, importance, embedding,
                 supersedes, journal_file)
            )
        self.conn.commit()
        return cur.lastrowid

    def get_memory(self, memory_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            return d
        return None

    def get_all_memories(self, type_filter: Optional[str] = None,
                         exclude_type: Optional[str] = None) -> list[dict]:
        if type_filter:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE type = ? ORDER BY date DESC",
                (type_filter,)
            ).fetchall()
        elif exclude_type:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE type != ? ORDER BY date DESC",
                (exclude_type,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories ORDER BY date DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            result.append(d)
        return result

    def update_memory(self, memory_id: int, **fields) -> bool:
        if not fields:
            return False
        # Handle tags serialization
        if "tags" in fields and isinstance(fields["tags"], list):
            fields["tags"] = json.dumps(fields["tags"])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [memory_id]
        cur = self.conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?", values
        )
        self.conn.commit()
        return cur.rowcount > 0

    def update_retrieval(self, memory_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET retrieval_count = retrieval_count + 1, "
            "last_accessed = datetime('now') WHERE id = ?",
            (memory_id,)
        )
        self.conn.commit()

    def delete_memory(self, memory_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def get_recent_memories(self, limit: int = 5,
                            exclude_type: Optional[str] = None) -> list[dict]:
        if exclude_type:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE type != ? ORDER BY date DESC LIMIT ?",
                (exclude_type, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories ORDER BY date DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [self._parse_memory(row) for row in rows]

    def get_top_importance(self, limit: int = 5,
                           exclude_ids: set | None = None,
                           exclude_type: Optional[str] = None) -> list[dict]:
        if exclude_type:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE type != ? "
                "ORDER BY importance DESC, date DESC",
                (exclude_type,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories ORDER BY importance DESC, date DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = self._parse_memory(row)
            if exclude_ids and d["id"] in exclude_ids:
                continue
            result.append(d)
            if len(result) >= limit:
                break
        return result

    def _parse_memory(self, row) -> dict:
        d = dict(row)
        d["tags"] = json.loads(d["tags"])
        return d

    # â”€â”€ Journal Entries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def save_journal_entry(self, entry_id: str, author: str,
                           title: Optional[str], entry_type: str,
                           content: str, why_it_mattered: Optional[str] = None,
                           tags: list[str] | None = None,
                           importance: int = 5, pinned: bool = False,
                           resolved: Optional[bool] = None,
                           date: Optional[str] = None) -> str:
        tags_json = json.dumps(tags or [])
        resolved_int = None if resolved is None else int(resolved)
        if date:
            self.conn.execute(
                "INSERT OR REPLACE INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, "
                "tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, author, title, entry_type, content,
                 why_it_mattered, tags_json, importance, int(pinned),
                 resolved_int, date)
            )
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, "
                "tags, importance, pinned, resolved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, author, title, entry_type, content,
                 why_it_mattered, tags_json, importance, int(pinned),
                 resolved_int)
            )
        self.conn.commit()
        return entry_id

    def get_journal_entries(self, entry_type: Optional[str] = None,
                            limit: int = 50) -> list[dict]:
        if entry_type:
            rows = self.conn.execute(
                "SELECT * FROM journal_entries WHERE entry_type = ? "
                "ORDER BY date DESC LIMIT ?",
                (entry_type, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM journal_entries ORDER BY date DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [self._parse_journal(row) for row in rows]

    def get_journal_entry(self, entry_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM journal_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            return self._parse_journal(row)
        return None

    def get_journal_latest(self, count: int = 3, max_pins: int = 2) -> list[dict]:
        """Get pinned entries + most recent unpinned, up to count total."""
        pinned = self.conn.execute(
            "SELECT * FROM journal_entries WHERE pinned = 1 "
            "ORDER BY date ASC LIMIT ?",
            (max_pins,)
        ).fetchall()
        pinned_list = [self._parse_journal(r) for r in pinned]

        remaining = max(0, count - len(pinned_list))
        if remaining > 0:
            pinned_ids = [p["id"] for p in pinned_list]
            if pinned_ids:
                placeholders = ",".join("?" for _ in pinned_ids)
                rows = self.conn.execute(
                    f"SELECT * FROM journal_entries WHERE pinned = 0 "
                    f"AND id NOT IN ({placeholders}) "
                    f"ORDER BY date DESC LIMIT ?",
                    pinned_ids + [remaining]
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM journal_entries WHERE pinned = 0 "
                    "ORDER BY date DESC LIMIT ?",
                    (remaining,)
                ).fetchall()
            unpinned_list = [self._parse_journal(r) for r in rows]
        else:
            unpinned_list = []

        return pinned_list, unpinned_list

    def update_journal_pin(self, entry_id: str, pinned: bool) -> bool:
        cur = self.conn.execute(
            "UPDATE journal_entries SET pinned = ? WHERE id = ?",
            (int(pinned), entry_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def count_pinned_journal(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM journal_entries WHERE pinned = 1"
        ).fetchone()
        return row[0]

    def get_pinned_journal(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title FROM journal_entries WHERE pinned = 1 ORDER BY date"
        ).fetchall()
        return [dict(r) for r in rows]

    def _parse_journal(self, row) -> dict:
        d = dict(row)
        d["tags"] = json.loads(d["tags"])
        return d


# â”€â”€ Open database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DB_PATH = MEMORIES_DIR / "shared.db"
db = MemoryDatabase(DB_PATH)


# ============================================================
# Embedding Helpers
# ============================================================

EMBEDDING_WORKER_SCRIPT = r"""
import json
import os
import sys
import time
import tempfile
from pathlib import Path


class StartupLock:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None
        self._locked = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+", encoding="utf-8")
        self.file.seek(0)
        self.file.write("0")
        self.file.flush()
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                    self._locked = True
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
            self._locked = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.file and self._locked:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                try:
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        if self.file:
            self.file.close()


def send(payload):
    print(json.dumps(payload), flush=True)


def main():
    model_name = os.environ.get("CLAUDE_MEMORY_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    allow_download = os.environ.get("CLAUDE_MEMORY_EMBEDDING_ALLOW_DOWNLOAD", "").lower() in {"1", "true", "yes", "on"}
    warmup_text = os.environ.get("CLAUDE_MEMORY_EMBEDDING_WARMUP_TEXT", "Semantic memory embedding warmup")
    lock_path = os.environ.get(
        "CLAUDE_MEMORY_EMBEDDING_WORKER_LOCK",
        str(Path(tempfile.gettempdir()) / "claude-minilm-embedding-worker.lock"),
    )

    try:
        with StartupLock(lock_path):
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, local_files_only=not allow_download)
            model.encode(warmup_text, show_progress_bar=False)
        send({"type": "ready"})
    except Exception as exc:
        send({"type": "error", "error": str(exc)})
        return 1

    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = request.get("command")
            if command == "shutdown":
                send({"id": request_id, "ok": True})
                return 0
            if command != "encode":
                raise ValueError(f"Unknown command: {command}")
            embedding = model.encode(request.get("text", ""), show_progress_bar=False)
            send({"id": request_id, "embedding": embedding.tolist()})
        except Exception as exc:
            send({"id": request_id, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


class EmbeddingWorkerClient:
    """Small JSON-lines client for an embedding worker subprocess."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self._messages: queue.Queue[dict] = queue.Queue()
        self._request_id = 0
        self._lock = threading.Lock()

    def start(self, timeout_seconds: float):
        if self.is_running():
            return

        env = os.environ.copy()
        env["CLAUDE_MEMORY_EMBEDDING_MODEL_NAME"] = MODEL_NAME
        env["CLAUDE_MEMORY_EMBEDDING_ALLOW_DOWNLOAD"] = "1" if ALLOW_MODEL_DOWNLOAD else "0"
        env["CLAUDE_MEMORY_EMBEDDING_WARMUP_TEXT"] = "Semantic memory embedding warmup"
        env["CLAUDE_MEMORY_EMBEDDING_WORKER_LOCK"] = str(Path(tempfile.gettempdir()) / "claude-minilm-embedding-worker.lock")

        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", EMBEDDING_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._read_stdout, name="semantic-memory-embedding-worker-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="semantic-memory-embedding-worker-stderr", daemon=True).start()

        message = self._next_message(timeout_seconds)
        if message.get("type") == "ready":
            return
        if message.get("type") == "error":
            raise RuntimeError(message.get("error", "embedding worker failed"))
        raise RuntimeError(f"Unexpected embedding worker startup response: {message}")

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def encode(self, text: str, **_: object) -> np.ndarray:
        with self._lock:
            if not self.is_running():
                raise RuntimeError("embedding worker is not running")
            self._request_id += 1
            request_id = self._request_id
            self._send({"id": request_id, "command": "encode", "text": text})
            while True:
                message = self._next_message(float(os.environ.get("CLAUDE_MEMORY_EMBEDDING_ENCODE_TIMEOUT_SECONDS", "60")))
                if message.get("id") != request_id:
                    logger.debug(f"Ignoring out-of-order embedding worker message: {message}")
                    continue
                if "error" in message:
                    raise RuntimeError(message["error"])
                return np.array(message["embedding"], dtype=np.float32)

    def shutdown(self):
        if not self.is_running():
            return
        try:
            self._send({"id": 0, "command": "shutdown"})
        except Exception:
            pass
        try:
            self.process.terminate()
        except Exception:
            pass

    def _send(self, payload: dict):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("embedding worker stdin is unavailable")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _next_message(self, timeout_seconds: float) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"embedding worker did not respond within {timeout_seconds:.1f}s")
            if self.process is not None and self.process.poll() is not None and self._messages.empty():
                raise RuntimeError(f"embedding worker exited with code {self.process.returncode}")
            try:
                return self._messages.get(timeout=remaining)
            except queue.Empty:
                continue

    def _read_stdout(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._messages.put(json.loads(line))
            except json.JSONDecodeError:
                logger.info(f"Embedding worker stdout: {line}")

    def _read_stderr(self):
        if self.process is None or self.process.stderr is None:
            return
        for line in self.process.stderr:
            line = line.strip()
            if line:
                logger.info(f"Embedding worker: {line}")


def _set_embedding_status(status: str, error: Optional[str] = None):
    global embedding_status, model_error, embedding_started_at, embedding_finished_at
    with embedding_lock:
        embedding_status = status
        model_error = error
        if status == "warming":
            embedding_started_at = time.monotonic()
            embedding_finished_at = None
        elif status in {"ready", "failed"}:
            embedding_finished_at = time.monotonic()


def _load_embedding_model_for_warmup():
    """Start the embedding worker and wait for its ready signal in the background."""
    global model, model_error, embedding_worker

    if importlib.util.find_spec("sentence_transformers") is None:
        installed = install_missing_dependencies(
            {"sentence_transformers": "sentence-transformers"},
            context="Embedding model dependency"
        )
        if not installed:
            _set_embedding_status("failed", "sentence-transformers is not installed")
            return

    logger.info(f"Starting embedding worker for model: {MODEL_NAME} (allow_download={ALLOW_MODEL_DOWNLOAD})")
    worker = None
    try:
        worker = EmbeddingWorkerClient()
        worker.start(timeout_seconds=EMBEDDING_WARMUP_TIMEOUT_SECONDS)
        with embedding_lock:
            embedding_worker = worker
            model = worker
        _set_embedding_status("ready")
        logger.info("Embedding worker loaded and warmed successfully!")
    except Exception as e:
        if worker is not None:
            worker.shutdown()
        _set_embedding_status("failed", str(e))
        logger.exception(f"Failed to start embedding worker for '{MODEL_NAME}': {e}")


def _delayed_embedding_warmup(delay_seconds: float):
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    with embedding_lock:
        if model is not None or embedding_status not in {"cold", "scheduled"}:
            return
    _set_embedding_status("warming")
    _load_embedding_model_for_warmup()


def start_embedding_warmup(delay_seconds: float = 0) -> bool:
    """Start a background embedding warmup if one is not already running."""
    global embedding_status, embedding_warmup_thread

    with embedding_lock:
        if model is not None or embedding_status in {"warming", "ready"}:
            return False
        if embedding_status == "scheduled":
            return False
        embedding_status = "scheduled"
        embedding_warmup_thread = threading.Thread(
            target=_delayed_embedding_warmup,
            args=(delay_seconds,),
            name="semantic-memory-embedding-warmup",
            daemon=True,
        )
        embedding_warmup_thread.start()
        return True


def get_embedding_model() -> Optional[object]:
    """Return the embedding worker only when it is already warm."""
    with embedding_lock:
        loaded_model = model
        status = embedding_status
    if loaded_model is not None and status == "ready":
        if getattr(loaded_model, "is_running", lambda: True)():
            return loaded_model
        _set_embedding_status("failed", "embedding worker exited")
    return None


def shutdown_embedding_worker():
    with embedding_lock:
        worker = embedding_worker
    if worker is not None:
        worker.shutdown()


atexit.register(shutdown_embedding_worker)


def embedding_status_message() -> str:
    with embedding_lock:
        status = embedding_status
        error = model_error
        started = embedding_started_at
        finished = embedding_finished_at

    if status == "ready":
        if started and finished:
            return f"Embedding model is ready. Warmup took {finished - started:.1f}s."
        return "Embedding model is ready."
    if status == "failed":
        detail = f" Last load error: {error}" if error else ""
        return f"Embedding model failed to load.{detail}"
    if status == "warming":
        elapsed = time.monotonic() - started if started else 0
        return f"Embedding model is still warming up ({elapsed:.1f}s elapsed). Try again shortly."
    if status == "scheduled":
        return "Embedding model warmup is scheduled and will start shortly."
    return "Embedding model is cold. Warmup has not started yet."


def ensure_embedding_model_ready() -> tuple[Optional[object], Optional[str]]:
    """Return a ready embedding worker, or start warmup and return a non-blocking status message."""
    loaded_model = get_embedding_model()
    if loaded_model is not None:
        return loaded_model, None
    with embedding_lock:
        status = embedding_status
    if status == "cold":
        start_embedding_warmup(delay_seconds=0.5)
    return None, embedding_status_message()


def _embedding_unavailable(action: str) -> str:
    return (
        f"Cannot {action} yet. {embedding_status_message()} "
        "Memory listing, context summaries, journal reading, and visualization still work."
    )


def _embedding_to_blob(embedding) -> bytes:
    """Convert a numpy embedding array to bytes for SQLite BLOB storage."""
    return np.array(embedding, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    """Convert a SQLite BLOB back to a numpy vector."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# Search Core (unchanged algorithm â€” matrix multiply + boosts)
# ============================================================

def _search_memories_core(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int,
    embedding_model: object
) -> list[tuple[int, float, float, float]]:
    """Core search logic: matrix multiplication + vectorized boosting.

    Args:
        query: Search query text
        memories: List of memory dicts, each must have 'embedding' (BLOB bytes)
        top_k: Number of results to return
        embedding_model: Loaded sentence-transformer model

    Returns:
        List of (index, base_similarity, final_score, total_boost)
        sorted by descending final_score.
    """
    if not memories:
        return []

    # Stack all embeddings into a matrix and normalize
    # Embeddings come as BLOBs from SQLite â€” convert to numpy
    memory_matrix = np.array([
        _blob_to_vec(m['embedding']) for m in memories
    ])
    norm = np.linalg.norm(memory_matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1
    normalized_matrix = memory_matrix / norm

    # Encode and normalize query
    query_embedding = embedding_model.encode(query, show_progress_bar=False)
    query_norm = np.linalg.norm(query_embedding)
    normalized_query = query_embedding / query_norm if query_norm > 0 else query_embedding

    # Base cosine similarity via single matrix multiply
    base_similarities = np.dot(normalized_matrix, normalized_query)

    # Vectorized boosting
    retrieval_counts = np.array([m.get('retrieval_count', 0) for m in memories])
    retrieval_boosts = np.minimum(retrieval_counts * 0.01, 0.05)

    importances = np.array([m.get('importance', 5) for m in memories])
    importance_boosts = importances * 0.002

    query_terms = set(query.lower().split())
    tag_boosts = np.array([
        0.03 if not query_terms.isdisjoint(set(t.lower() for t in m.get('tags', [])))
        else 0.0
        for m in memories
    ])

    # Final scores
    final_scores = base_similarities + retrieval_boosts + importance_boosts + tag_boosts
    top_k = min(top_k, len(memories))
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        base = float(base_similarities[idx])
        final = float(final_scores[idx])
        boost = final - base
        results.append((int(idx), base, final, boost))
    return results


# ============================================================
# MCP Tools â€” Memories
# ============================================================

@mcp.tool()
def add_memory(
    text: str,
    tags: str = "",
    importance: int = 5,
    memory_type: str = "general",
    date: str = None
) -> str:
    """Add a new memory to the semantic memory system.

    Args:
        text: The memory text to store
        tags: Comma-separated tags (e.g. "project,ai,important")
        importance: Importance rating from 1-10
        memory_type: Type of memory (general, achievement, milestone, etc)
        date: Optional YYYY-MM-DD date (defaults to today)
    """
    logger.info(f"Adding memory: {text[:50]}...")

    importance = max(1, min(10, importance))
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    memory_date = date if date else datetime.now().strftime("%Y-%m-%d")

    embedding_model, _ = ensure_embedding_model_ready()
    if embedding_model is None:
        return _embedding_unavailable("add memory")

    embedding = embedding_model.encode(text, show_progress_bar=False)
    embedding_blob = _embedding_to_blob(embedding)

    # Dedup check: warn if a very similar memory already exists
    existing = db.get_all_memories()
    existing_with_emb = [m for m in existing if m.get('embedding')]
    if existing_with_emb:
        emb_matrix = np.array([_blob_to_vec(m['embedding']) for m in existing_with_emb])
        norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1
        normalized = emb_matrix / norm
        q_norm = np.linalg.norm(embedding)
        q_normalized = embedding / q_norm if q_norm > 0 else embedding
        similarities = np.dot(normalized, q_normalized)
        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim > 0.9:
            match = existing_with_emb[max_idx]
            return (
                f"Duplicate detected (similarity: {max_sim:.3f})!\n"
                f"  Existing memory [{match['id']}]: {match['text'][:100]}...\n"
                f"  Your text: {text[:100]}...\n\n"
                f"Use update_memory to modify the existing one, "
                f"or add_memory_force to save anyway."
            )

    memory_id = db.save_memory(
        text=text,
        tags=tag_list,
        type=memory_type,
        importance=importance,
        embedding=embedding_blob,
        date=memory_date
    )

    logger.info(f"Memory #{memory_id} saved successfully")
    return f"Memory #{memory_id} added successfully: '{text[:50]}...'"


@mcp.tool()
def add_memory_force(
    text: str,
    tags: str = "",
    importance: int = 5,
    memory_type: str = "general",
    date: str = None
) -> str:
    """Force-add a memory, bypassing the duplicate similarity check.
    Use this only after add_memory has flagged a duplicate and you still want to save.

    Args:
        text: The memory text to store
        tags: Comma-separated tags (e.g. "project,ai,important")
        importance: Importance rating from 1-10
        memory_type: Type of memory (general, achievement, milestone, etc)
        date: Optional YYYY-MM-DD date (defaults to today)
    """
    importance = max(1, min(10, importance))
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    memory_date = date if date else datetime.now().strftime("%Y-%m-%d")

    embedding_model, _ = ensure_embedding_model_ready()
    if embedding_model is None:
        return _embedding_unavailable("force-add memory")

    embedding = embedding_model.encode(text, show_progress_bar=False)
    embedding_blob = _embedding_to_blob(embedding)

    memory_id = db.save_memory(
        text=text,
        tags=tag_list,
        type=memory_type,
        importance=importance,
        embedding=embedding_blob,
        date=memory_date
    )

    return f"Memory #{memory_id} force-added: '{text[:50]}...'"


@mcp.tool()
def update_memory(
    memory_id: int,
    text: str = None,
    tags: str = None,
    importance: int = None,
    memory_type: str = None,
    date: str = None
) -> str:
    """Update an existing memory by its ID. Only provided fields will be changed.

    Args:
        memory_id: The integer ID of the memory to update
        text: New text for the memory
        tags: New comma-separated tags
        importance: New importance rating (1-10)
        memory_type: New memory type
        date: New YYYY-MM-DD date
    """
    logger.info(f"Updating memory: {memory_id}")

    existing = db.get_memory(memory_id)
    if not existing:
        return f"Error: Memory #{memory_id} not found."

    fields = {}
    if text is not None:
        embedding_model, _ = ensure_embedding_model_ready()
        if embedding_model is None:
            return _embedding_unavailable("update memory text")
        fields["text"] = text
        fields["embedding"] = _embedding_to_blob(embedding_model.encode(text, show_progress_bar=False))
    if tags is not None:
        fields["tags"] = [t.strip() for t in tags.split(',') if t.strip()]
    if importance is not None:
        fields["importance"] = max(1, min(10, importance))
    if memory_type is not None:
        fields["type"] = memory_type
    if date is not None:
        fields["date"] = date

    if not fields:
        return "No fields to update."

    db.update_memory(memory_id, **fields)
    return f"Memory #{memory_id} updated successfully."


@mcp.tool()
def delete_memory(memory_id: int) -> str:
    """Delete a single memory by its ID. This action is irreversible.

    Args:
        memory_id: The integer ID of the memory to delete
    """
    logger.info(f"Deleting memory: {memory_id}")

    existing = db.get_memory(memory_id)
    if not existing:
        return f"Error: Memory #{memory_id} not found."

    text_preview = existing['text'][:60]
    db.delete_memory(memory_id)
    return f"Memory #{memory_id} deleted: '{text_preview}...'"


@mcp.tool()
def check_embedding_model() -> str:
    """Check whether semantic memory embeddings are ready."""
    return embedding_status_message()


@mcp.tool()
def search_memory(
        query: str,
        top_k: int = 3
) -> str:
    """Search for semantically similar memories using vectorized operations."""
    logger.info(f"Searching for: {query}")

    top_k = max(1, min(10, top_k))

    memories = db.get_all_memories()
    # Filter to only memories with embeddings
    memories_with_emb = [m for m in memories if m.get('embedding')]
    if not memories_with_emb:
        return "No memories found in the system yet. Use add_memory to create your first memory!"

    embedding_model, _ = ensure_embedding_model_ready()
    if embedding_model is None:
        return _embedding_unavailable("search memories")

    results = _search_memories_core(query, memories_with_emb, top_k, embedding_model)

    logger.info(f"Found {len(results)} results")
    output_lines = [f"Found {len(memories_with_emb)} total memories, showing top {len(results)}:\n"]

    for i, (idx, base_sim, final_score, total_boost) in enumerate(results, 1):
        mem = memories_with_emb[idx]
        boost_str = f" (+{total_boost:.3f} boost)" if total_boost > 0 else ""
        mem_type = mem.get('type', 'general')

        db.update_retrieval(mem['id'])

        output_lines.append(f"{i}. [{mem['id']}] ({mem_type}) Similarity: {final_score:.3f}{boost_str}")
        output_lines.append(f"   {mem['text']}")
        output_lines.append(f"   Tags: {', '.join(mem['tags']) if mem['tags'] else 'none'}")
        output_lines.append(
            f"   Importance: {mem['importance']}/10, Retrieved: {mem.get('retrieval_count', 0)} times\n")

    return '\n'.join(output_lines)


@mcp.tool()
def list_memories(limit: int = 10) -> str:
    """List recent memories.

    Args:
        limit: Maximum number of memories to return (default 10, max 50)
    """
    logger.info(f"Listing memories (limit: {limit})")
    limit = max(1, min(50, limit))

    memories = db.get_recent_memories(limit=limit, exclude_type="journal")

    if not memories:
        return "No memories stored yet. Use add_memory to create your first memory!"

    total_row = db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE type != 'journal'"
    ).fetchone()
    total = total_row[0]

    output_lines = [f"Total memories: {total}\nShowing {len(memories)} most recent:\n"]

    for mem in memories:
        output_lines.append(f"[{mem['id']}] {mem['date']} - {mem['text'][:80]}...")
        tags_str = ', '.join(mem['tags']) if mem['tags'] else 'none'
        output_lines.append(f"  Tags: {tags_str}, Importance: {mem['importance']}/10, Type: {mem.get('type', 'general')}\n")

    return '\n'.join(output_lines)


@mcp.tool()
def get_context_summary() -> str:
    """Get a curated summary of memories for session context.

    Returns the 5 most recent memories and 5 highest-importance memories
    to provide the LLM with a 'Smart Context' of the user's history.
    """
    logger.info("Generating context summary...")

    recent = db.get_recent_memories(limit=5, exclude_type="journal")
    if not recent:
        return "No memories found. Start by adding some with add_memory!"

    recent_ids = {m['id'] for m in recent}
    important = db.get_top_importance(limit=5, exclude_ids=recent_ids, exclude_type="journal")

    output = ["### Memory Context Summary\n"]

    output.append("#### Recent History (Continuity)")
    for m in recent:
        output.append(f"- [{m['id']}] {m['date']}: {m['text'][:120]}...")

    if important:
        output.append("\n#### Core Context (High Importance)")
        for m in important:
            output.append(f"- [{m['id']}] {m['date']}: {m['text'][:120]}...")

    output.append("\n*Tip: Use search_memory if you need to dig deeper into specific topics.*")

    return '\n'.join(output)


@mcp.tool()
def visualize_memories() -> str:
    """Export memories and launch the interstellar nebula visualization in your browser.

    This tool sets up the Semantic Nebula dashboard in your memories directory
    and opens it automatically.
    """
    logger.info("Initializing visualization...")

    memories = db.get_all_memories(exclude_type="journal")
    export_data = []
    for mem in memories:
        if not mem.get('embedding'):
            continue
        export_data.append({
            "id": mem["id"],
            "text": mem["text"],
            "tags": mem["tags"],
            "importance": mem["importance"],
            "retrieval_count": mem.get("retrieval_count", 0),
            "date": mem["date"],
            "embedding": _blob_to_vec(mem["embedding"]).tolist()
        })

    # Save to user's memory directory as .js to bypass CORS
    data_file = MEMORIES_DIR / "memories.js"
    try:
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write("var MEMORY_DATA = ")
            json.dump(export_data, f, indent=2)
            f.write(";")
    except IOError as e:
        logger.error(f"Failed to save memories.js: {e}")
        return f"Failed to export data: {str(e)}"

    # Setup HTML (copy from bundle to data dir)
    script_dir = Path(__file__).parent
    viz_src = script_dir / "visualizer" / "index.html"
    viz_dest = MEMORIES_DIR / "index.html"

    try:
        if viz_src.exists():
            shutil.copy2(viz_src, viz_dest)
            logger.info(f"Dashboard HTML copied to {viz_dest}")
        else:
            alt_src = script_dir.parent / "visualizer" / "index.html"
            if alt_src.exists():
                shutil.copy2(alt_src, viz_dest)
                logger.info(f"Dashboard HTML copied from alternative source: {alt_src}")
            else:
                return "Could not find visualizer source files in the bundle."
    except Exception as e:
        logger.error(f"Failed to setup visualizer HTML: {e}")
        return f"Failed to setup dashboard: {str(e)}"

    # Launch Browser
    viz_url = viz_dest.absolute().as_uri()
    try:
        webbrowser.open(viz_url)
        logger.info(f"Browser opened to {viz_url}")
        return f"Nebula Visualization launched! Found {len(export_data)} memories.\nOpening: {viz_dest}"
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        return f"Data updated. Please open this file manually to see the nebula:\n{viz_dest}"


# ============================================================
# MCP Tools â€” Journal
# ============================================================

def _generate_slug(title: str, max_length: int = 50) -> str:
    """Generate a filesystem-safe slug from a title."""
    slug = title.lower()
    slug = re.sub(r'[<>:"/\\|?*]', '', slug)
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    return slug or 'untitled'


@mcp.tool()
def write_journal(
    title: str,
    content: str,
    author: str,
    tags: str = "",
    importance: int = 5,
    summary: str = ""
) -> str:
    """Write a new journal entry. Creates a database record
    and a companion memory for semantic search.

    Args:
        title: Entry title
        content: Full journal entry text (markdown)
        author: Author name (e.g. "Sunshine", "Valentine", "Newbie")
        tags: Comma-separated tags (e.g. "relationship,milestone")
        importance: Importance rating 1-10
        summary: Brief 1-2 sentence summary for search (auto-generated if empty)
    """
    logger.info(f"Writing journal entry: {title}")

    importance = max(1, min(10, importance))
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = _generate_slug(title)
    entry_id = f"{date_str}_{slug}"

    # Check for collision and add counter if needed
    existing = db.get_journal_entry(entry_id)
    if existing:
        counter = 2
        while db.get_journal_entry(f"{entry_id}-{counter}"):
            counter += 1
        entry_id = f"{entry_id}-{counter}"

    tag_list = [t.strip() for t in tags.split(',') if t.strip()]

    embedding_model, _ = ensure_embedding_model_ready()
    if embedding_model is None:
        return _embedding_unavailable("write journal entry")

    # Auto-generate summary if not provided
    if not summary.strip():
        summary_text = content[:150].rstrip()
        if len(content) > 150:
            summary_text += "..."
    else:
        summary_text = summary.strip()

    # Save journal entry to DB
    db.save_journal_entry(
        entry_id=entry_id,
        author=author,
        title=title,
        entry_type="reflection",
        content=content,
        why_it_mattered=summary_text,
        tags=tag_list,
        importance=importance,
        date=date_str
    )
    logger.info(f"Journal entry saved: {entry_id}")

    # Create companion memory for semantic search
    memory_text = f"{summary_text} | Journal: {entry_id}"
    embedding = embedding_model.encode(content, show_progress_bar=False)
    embedding_blob = _embedding_to_blob(embedding)

    memory_id = db.save_memory(
        text=memory_text,
        tags=tag_list,
        type="journal",
        importance=importance,
        embedding=embedding_blob,
        journal_file=entry_id,
        date=date_str
    )
    logger.info(f"Companion memory #{memory_id} created for journal entry")

    return (
        f"Journal entry saved!\n"
        f"  Entry: {entry_id}\n"
        f"  Memory: #{memory_id}\n"
        f"  Author: {author}\n"
        f"  Tags: {', '.join(tag_list) if tag_list else 'none'}\n"
        f"  Importance: {importance}/10"
    )


@mcp.tool()
def read_journal_latest(count: int = 3) -> str:
    """Read the latest journal entries for orientation. Returns pinned entries
    first, then most recent.

    Args:
        count: Number of entries to include (default 3)
    """
    logger.info(f"Reading journal latest (count={count})")

    pinned, unpinned = db.get_journal_latest(count=count, max_pins=2)

    if not pinned and not unpinned:
        return "No journal entries yet. Use write_journal to create your first entry."

    sections = ["# Journal -- Latest Entries\n"]

    if pinned:
        sections.append("## Pinned\n")
        for e in pinned:
            sections.append(f"### {e.get('title', 'Untitled')} ({e.get('date', 'unknown')})")
            sections.append(f"*By {e.get('author', 'Unknown')}*\n")
            sections.append(e.get('content', ''))
            sections.append("\n---\n")

    if unpinned:
        if pinned:
            sections.append("## Recent\n")
        for e in unpinned:
            sections.append(f"### {e.get('title', 'Untitled')} ({e.get('date', 'unknown')})")
            sections.append(f"*By {e.get('author', 'Unknown')}*\n")
            sections.append(e.get('content', ''))
            sections.append("\n---\n")

    return '\n'.join(sections)


@mcp.tool()
def search_journal(query: str, top_k: int = 3) -> str:
    """Search journal entries by semantic similarity.

    Args:
        query: What to search for (e.g. "that night we felt vulnerable")
        top_k: Number of results (default 3, max 10)
    """
    logger.info(f"Searching journal for: {query}")

    top_k = max(1, min(10, top_k))

    journal_memories = db.get_all_memories(type_filter="journal")
    journal_with_emb = [m for m in journal_memories if m.get('embedding')]

    if not journal_with_emb:
        return "No journal entries found. Use write_journal to create your first entry."

    embedding_model, _ = ensure_embedding_model_ready()
    if embedding_model is None:
        return _embedding_unavailable("search journal entries")

    results = _search_memories_core(query, journal_with_emb, top_k, embedding_model)

    output_lines = [f"Found {len(journal_with_emb)} journal entries, showing top {len(results)}:\n"]

    for i, (idx, base_sim, final_score, total_boost) in enumerate(results, 1):
        mem = journal_with_emb[idx]
        boost_str = f" (+{total_boost:.3f} boost)" if total_boost > 0 else ""

        db.update_retrieval(mem['id'])

        # Extract summary and entry id from text
        text = mem.get('text', '')
        if ' | Journal: ' in text:
            display_summary, entry_ref = text.rsplit(' | Journal: ', 1)
        elif ' | File: ' in text:
            display_summary, entry_ref = text.rsplit(' | File: ', 1)
        else:
            display_summary = text
            entry_ref = 'unknown'

        output_lines.append(f"{i}. [{mem['id']}] Similarity: {final_score:.3f}{boost_str}")
        output_lines.append(f"   {display_summary}")
        output_lines.append(f"   Entry: {entry_ref}")
        output_lines.append(f"   Tags: {', '.join(mem['tags']) if mem['tags'] else 'none'}")
        output_lines.append(
            f"   Importance: {mem['importance']}/10, Retrieved: {mem.get('retrieval_count', 0)} times\n")

    return '\n'.join(output_lines)


@mcp.tool()
def list_journal_entries(limit: int = 10) -> str:
    """List journal entries with metadata, newest first.

    Args:
        limit: Maximum entries to show (default 10, max 50)
    """
    logger.info(f"Listing journal entries (limit: {limit})")

    limit = max(1, min(50, limit))
    entries = db.get_journal_entries(limit=limit)

    if not entries:
        return "No journal entries yet. Use write_journal to create your first entry."

    total_row = db.conn.execute("SELECT COUNT(*) FROM journal_entries").fetchone()
    total = total_row[0]

    output_lines = [f"Journal: {total} entries, showing {len(entries)} most recent:\n"]
    output_lines.append(f"{'Date':<12} {'Author':<14} {'Title':<30} {'Imp':>3} {'Pin':>3}  ID")
    output_lines.append(f"{'-'*12} {'-'*14} {'-'*30} {'-'*3} {'-'*3}  {'-'*20}")

    for e in entries:
        date = str(e.get('date', '?'))[:10]
        author = str(e.get('author', '?'))[:14]
        title = str(e.get('title', 'Untitled'))[:30]
        imp = str(e.get('importance', '?'))
        pin = 'Yes' if e.get('pinned') else ' No'
        eid = e.get('id', '?')
        output_lines.append(f"{date:<12} {author:<14} {title:<30} {imp:>3} {pin:>3}  {eid}")

    return '\n'.join(output_lines)


@mcp.tool()
def pin_journal_entry(entry_id: str, pinned: bool = True) -> str:
    """Pin or unpin a journal entry. Pinned entries always appear in latest view.

    Args:
        entry_id: Entry ID (e.g. "2026-02-25_doors-opening")
        pinned: True to pin, False to unpin
    """
    logger.info(f"{'Pinning' if pinned else 'Unpinning'} journal entry: {entry_id}")

    entry = db.get_journal_entry(entry_id)
    if not entry:
        return f"Entry not found: {entry_id}"

    if pinned:
        current_pins = db.count_pinned_journal()
        max_pins = 2
        if current_pins >= max_pins:
            pinned_entries = db.get_pinned_journal()
            pin_list = '\n'.join(f"  - {p['id']}: {p['title']}" for p in pinned_entries)
            return (
                f"Cannot pin: already at maximum ({max_pins} pins).\n"
                f"Currently pinned:\n{pin_list}\n\n"
                f"Unpin one first with pin_journal_entry(entry_id, pinned=False)."
            )

    db.update_journal_pin(entry_id, pinned)

    action = "Pinned" if pinned else "Unpinned"
    return f"{action}: {entry_id} ({entry.get('title', 'Untitled')})."


def main():
    """Run the MCP server"""
    logger.info("Starting Claude Memory MCP server...")
    logger.info(f"Memories stored in: {DB_PATH}")
    if WARM_EMBEDDINGS_ON_STARTUP:
        start_embedding_warmup(delay_seconds=EMBEDDING_WARMUP_DELAY_SECONDS)
        logger.info(f"Embedding warmup scheduled in {EMBEDDING_WARMUP_DELAY_SECONDS:.1f}s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

