"""Local, dependency-free prototype for durable Styligma photo/video jobs.

NOT integrated with the live API; requires a dedicated Railway volume before use.
Use one worker process for MVP. No external API calls in this module.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import tempfile
import threading
import time
import uuid
from typing import Any


class InvalidJob(ValueError):
    pass


class RateLimitExceeded(Exception):
    """Video scan budget reached; HTTP mapping belongs in the API adapter."""


class JobCapacityExceeded(Exception):
    """Atomic active-job capacity exceeded; mapped to HTTP 429 by the API."""

class RequestConflict(Exception):
    """Same caller request UUID reused with a different payload or job kind."""

class RequestExpired(Exception):
    """Original upload is gone; never silently start another paid job."""


class DurableJobStore:
    """SQLite-backed job metadata with atomic, file-backed input/output blobs.

    Leases fence stale workers. A crashed job is NOT automatically re-run because
    doing so might duplicate a paid provider call. Once its lease expires it
    becomes retry_required; only the user can explicitly submit another job.
    The active worker periodically renews its lease.
    """

    TTL = {"photo": 15 * 60, "video": 60 * 60}
    # Queue deadlines only: after a worker claims a job its terminal result
    # gets the appropriate TTL. Pending inputs should not wait hours on disk.
    QUEUE_TTL = {"photo": 15 * 60, "video": 30 * 60}
    REQUEST_KEY_TTL = 48 * 60 * 60  # metadata tombstones, no photo bytes
    MAX_BYTES = {"photo": 20 * 1024 * 1024, "video": 50 * 1024 * 1024}

    def __init__(self, directory: str | os.PathLike, *, clock=time.time):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        if not self.root.is_dir():
            raise ValueError("Job directory must exist")
        self._clock = clock
        self._lock = threading.RLock()
        self.db_path = self.root / "jobs.sqlite3"
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL,
                    expires_at REAL,
                    lease_until REAL,
                    lease_token TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL,
                    error TEXT
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_expiry ON jobs(expires_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(status,lease_until)")
            connection.execute("""CREATE TABLE IF NOT EXISTS scan_rate_events (
                bucket TEXT NOT NULL, timestamp REAL NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS rate_bucket_time ON scan_rate_events(bucket,timestamp)")
            connection.execute("""CREATE TABLE IF NOT EXISTS request_keys (
                request_hash TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                job_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS request_keys_expiry ON request_keys(expires_at)")
        # Job metadata and pseudonymous network quotas are private too.
        os.chmod(self.db_path, 0o600)
        self._rate_salt = self._load_or_create_rate_salt()

    def _load_or_create_rate_salt(self) -> bytes:
        # Hash client network keys; avoid persisting plaintext IP addresses.
        target = self.root / ".video_rate_salt"
        if not target.exists():
            salt = secrets.token_bytes(32)
            try:
                with target.open("xb") as f:
                    os.chmod(target, 0o600)
                    f.write(salt)
                    f.flush()
                    os.fsync(f.fileno())
            except FileExistsError:
                pass  # Another process initialized this store.
        salt = target.read_bytes()
        if len(salt) != 32:
            raise RuntimeError("Persistent video rate salt is invalid")
        return salt

    def enforce_video_rate(self, network_key: str, *, max_scans: int = 5,
                           window_seconds: int = 86400) -> None:
        """Atomically enforce rolling video scan cap across serverless restarts.

        Caller should pass the same network identity as the original video API.
        Not a secure per-account quota: shared and rotating IPs remain imperfect.
        """
        if not isinstance(network_key, str) or not network_key or len(network_key) > 512:
            raise RateLimitExceeded("Unable to validate scan limit.")
        if max_scans < 1 or window_seconds < 1:
            raise ValueError("Invalid rate limiting configuration")
        digest = hmac.new(self._rate_salt, network_key.encode("utf-8"), hashlib.sha256).hexdigest()
        now = self._clock()
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("DELETE FROM scan_rate_events WHERE timestamp<=?", (now-window_seconds,))
            count = con.execute(
                "SELECT COUNT(*) FROM scan_rate_events WHERE bucket=?", (digest,)).fetchone()[0]
            if count >= max_scans:
                raise RateLimitExceeded(
                    "Too many scans from this network today. Please try again tomorrow.")
            con.execute("INSERT INTO scan_rate_events(bucket,timestamp) VALUES (?,?)", (digest, now))

    @contextlib.contextmanager
    def _connect(self):
        # No long-lived DB connection: the service must be able to sleep.
        con = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=FULL")
            # Only metadata lives in SQLite; still scrub removed metadata pages.
            con.execute("PRAGMA secure_delete=ON")
            con.execute("PRAGMA busy_timeout=10000")
            with con:
                yield con
        finally:
            con.close()

    def _blob_path(self, job_id: str, suffix: str) -> Path:
        # job_id never contains user-provided filesystem text.
        if len(job_id) != 32 or not all(c in "0123456789abcdef" for c in job_id):
            raise InvalidJob("Invalid job ID")
        return self.root / f"{job_id}.{suffix}"

    def _atomic_write(self, path: Path, data: bytes):
        with tempfile.NamedTemporaryFile(dir=self.root, prefix="write-", delete=False) as f:
            tmp = Path(f.name)
            try:
                os.chmod(tmp, 0o600)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _request_hash(self, request_id: str) -> str:
        """Stable client-generated UUID4; only a keyed digest goes to SQLite."""
        try:
            normalized = str(uuid.UUID(request_id))
            if normalized != request_id.lower() or uuid.UUID(request_id).version != 4:
                raise ValueError("Not a UUID4")
        except (TypeError, ValueError, AttributeError):
            raise ValueError("Request ID must be a UUID4") from None
        return hmac.new(self._rate_salt, ("request:" + normalized).encode(), hashlib.sha256).hexdigest()

    def find_request(self, request_id: str) -> dict | None:
        """Retrieve a job id safely after an uncertain HTTP response.

        Only the bearer-like random UUID4 is accepted. No photos or results
        are returned by this lookup. A 48h tombstone prevents paid replays even
        after the job result expires; the client needs a new ID for manual retry.
        """
        digest = self._request_hash(request_id)
        with self._lock, self._connect() as con:
            row = con.execute("SELECT kind,job_id,expires_at FROM request_keys WHERE request_hash=?",(digest,)).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= self._clock():
            return {"kind": row["kind"], "status": "expired"}
        status = self.status(row["job_id"], include_result=False)
        return {"kind": row["kind"], "job_id": row["job_id"],
                "status": status["status"] if status else "expired",
                "error": status.get("error") if status else None}

    def submit(self, kind: str, payload: bytes, content_type: str, *,
               metadata: dict | None = None, max_active: int | None = None,
               request_id: str | None = None) -> str:
        if kind not in self.TTL:
            raise ValueError("Unsupported job kind")
        if max_active is not None and max_active < 1:
            raise ValueError("max_active must be positive")
        if not payload or len(payload) > self.MAX_BYTES[kind]:
            raise ValueError("Payload is empty or too large")
        # Avoid storing secrets, push tokens or arbitrary client data in metadata.
        metadata = metadata or {}
        if not isinstance(metadata, dict) or not all(isinstance(k, str) for k in metadata):
            raise ValueError("Invalid metadata")
        serialized = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > 4096:
            raise ValueError("Metadata too large")
        request_hash = self._request_hash(request_id) if request_id else None
        payload_digest = hashlib.sha256(payload).hexdigest() if request_hash else None
        job_id = uuid.uuid4().hex
        path = self._blob_path(job_id, "input")
        self._atomic_write(path, payload)
        now = self._clock()
        existing_id = None
        try:
            with self._lock, self._connect() as con:
                # The old check-before-upload is only advisory. Enforce capacity
                # in the same write transaction as inserting this new job, so
                # simultaneous requests cannot all pass a stale count check.
                # BEGIN IMMEDIATE serializes competing SQLite connections too.
                con.execute("BEGIN IMMEDIATE")
                if request_hash is not None:
                    existing = con.execute("SELECT * FROM request_keys WHERE request_hash=?",(request_hash,)).fetchone()
                    if existing is not None:
                        if existing["kind"] != kind or existing["payload_sha256"] != payload_digest:
                            raise RequestConflict("Request ID was already used for different upload")
                        if existing["expires_at"] <= now or con.execute(
                            "SELECT 1 FROM jobs WHERE job_id=?", (existing["job_id"],)).fetchone() is None:
                            raise RequestExpired("Original upload expired. Start a fresh explicit upload.")
                        existing_id = existing["job_id"]
                if existing_id is None and max_active is not None:
                    active = con.execute("""SELECT COUNT(*) FROM jobs WHERE kind=?
                        AND status IN ('queued','processing')
                        AND (expires_at IS NULL OR expires_at>?)""",
                        (kind, now)).fetchone()[0]
                    if active >= max_active:
                        raise JobCapacityExceeded("Scanner is busy. Please try again shortly.")
                if existing_id is None:
                    con.execute("""INSERT INTO jobs
                        (job_id, kind, status, content_type, created_at, updated_at,
                         expires_at, metadata) VALUES (?,?, 'queued', ?,?,?,?,?)""",
                        (job_id, kind, content_type, now, now,
                         now + self.QUEUE_TTL[kind], serialized))
                    if request_hash is not None:
                        con.execute("""INSERT INTO request_keys
                            (request_hash,kind,payload_sha256,job_id,created_at,expires_at)
                            VALUES (?,?,?,?,?,?)""",
                            (request_hash,kind,payload_digest,job_id,now,now+self.REQUEST_KEY_TTL))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        if existing_id is not None:
            path.unlink(missing_ok=True)
            return existing_id
        return job_id

    def claim(self, job_id: str, *, lease_seconds: int = 600) -> tuple[str, bytes, str, dict] | None:
        if lease_seconds < 30:
            raise ValueError("Lease must be at least 30 seconds")
        now = self._clock()
        token = secrets.token_hex(16)
        # The store lock also fences blob unlink from other operations in-process.
        with self._lock:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if (row is None or row["status"] != "queued" or
                        (row["expires_at"] is not None and row["expires_at"] <= now)):
                    return None
                # A running job must not expire based on its QUEUED deadline.
                con.execute("""UPDATE jobs SET status='processing', lease_until=?,
                    lease_token=?, attempts=attempts+1, updated_at=?,expires_at=NULL
                    WHERE job_id=?""", (now + lease_seconds, token, now, job_id))
                try:
                    payload = self._blob_path(job_id, "input").read_bytes()
                except OSError:
                    con.execute("""UPDATE jobs SET status='error', error='missing_input',
                        finished_at=?, expires_at=?, lease_token=NULL, lease_until=NULL
                        WHERE job_id=?""", (now, now + self.TTL[row["kind"]], job_id))
                    return None
                content_type = row["content_type"]
                metadata = json.loads(row["metadata"])
            # The transaction has committed: it is now impossible to re-claim
            # the upload. Remove its raw bytes from persistent disk immediately;
            # the processor has its own in-memory copy. After a crash no re-run
            # is allowed, even if an old paid provider request already completed.
            self._blob_path(job_id, "input").unlink(missing_ok=True)
            return token, payload, content_type, metadata

    def renew(self, job_id: str, token: str, *, lease_seconds: int = 600) -> bool:
        now = self._clock()
        with self._lock, self._connect() as con:
            c = con.execute("""UPDATE jobs SET lease_until=?,updated_at=?
                WHERE job_id=? AND lease_token=? AND status='processing'""",
                (now + lease_seconds, now, job_id, token))
            return c.rowcount == 1

    def complete(self, job_id: str, token: str, result: dict) -> bool:
        body = json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        # Result limit prevents large base64 responses from filling a tiny volume.
        if len(body) > 80 * 1024 * 1024:
            raise ValueError("Result exceeds 80 MB")
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT kind,lease_token,status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "processing" or row["lease_token"] != token:
                return False
            self._atomic_write(self._blob_path(job_id, "result"), body)
            now = self._clock()
            con.execute("""UPDATE jobs SET status='done', finished_at=?,
                expires_at=?, updated_at=?,lease_token=NULL,lease_until=NULL
                WHERE job_id=? AND lease_token=?""",
                (now, now + self.TTL[row["kind"]], now, job_id, token))
            # Remove input only after the done marker is committed.
        self._blob_path(job_id, "input").unlink(missing_ok=True)
        return True

    def fail(self, job_id: str, token: str, code: str = "processing_failed") -> bool:
        # Never store exception messages: they can contain sensitive data.
        if code not in {"processing_failed", "retry_required", "cancelled", "missing_input"}:
            code = "processing_failed"
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT kind,lease_token,status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "processing" or row["lease_token"] != token:
                return False
            now = self._clock()
            con.execute("""UPDATE jobs SET status='error',error=?, finished_at=?,
                expires_at=?, updated_at=?,lease_token=NULL,lease_until=NULL
                WHERE job_id=? AND lease_token=?""",
                (code, now, now + self.TTL[row["kind"]], now, job_id, token))
        # Failed work must be explicitly resubmitted with a fresh upload.
        # Remove unnecessary user photo/video bytes as soon as state is committed.
        self._blob_path(job_id, "input").unlink(missing_ok=True)
        return True

    def mark_interrupted(self) -> int:
        """Convert lease-expired running jobs into explicit retry-required.

        Does not run any processor or initiate a paid retry. Called at startup
        and on polling. Active workers renew their lease before expiry.
        """
        now = self._clock()
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            rows = con.execute("""SELECT job_id,kind FROM jobs
                WHERE status='processing' AND lease_until IS NOT NULL
                AND lease_until<=?""", (now,)).fetchall()
            for row in rows:
                con.execute("""UPDATE jobs SET status='error', error='retry_required',
                    finished_at=?,expires_at=?,updated_at=?,lease_token=NULL,
                    lease_until=NULL WHERE job_id=? AND status='processing'""",
                    (now, now + self.TTL[row['kind']], now, row['job_id']))
        # In-progress jobs are never replayed; their source bytes are no longer needed.
        for row in rows:
            self._blob_path(row['job_id'], "input").unlink(missing_ok=True)
        return len(rows)

    def status(self, job_id: str, *, include_result: bool = True) -> dict | None:
        self.mark_interrupted()
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            now = self._clock()
            if row["expires_at"] is not None and row["expires_at"] <= now:
                # Never leave an expired image/result on disk just because a
                # status request arrived before the next maintenance sweep.
                # cleanup runs under the same re-entrant lock (RLock).
                self.cleanup()
                return None
            state = row["status"]
            if state == "queued":
                state = "processing"  # backwards-compatible client response
            response = {"status": state, "kind": row["kind"]}
            if state == "error":
                response["error"] = row["error"] or "processing_failed"
            if state == "done" and include_result:
                try:
                    response["result"] = json.loads(self._blob_path(job_id, "result").read_bytes())
                except (OSError, json.JSONDecodeError):
                    response = {"status": "error", "error": "retry_required", "kind": row["kind"]}
            return response

    def ready_ids(self, *, max_attempts: int = 1, limit: int = 8) -> list[str]:
        # Only queued, never-started work may run automatically after a restart.
        with self._lock, self._connect() as con:
            rows = con.execute("""SELECT job_id FROM jobs
                WHERE status='queued' AND attempts<?
                  AND (expires_at IS NULL OR expires_at>?)
                  ORDER BY created_at LIMIT ?""",
                (max_attempts, self._clock(), limit)).fetchall()
            return [r[0] for r in rows]

    def count_active(self, kind: str) -> int:
        with self._lock, self._connect() as con:
            row = con.execute("""SELECT COUNT(*) FROM jobs
                WHERE kind=? AND status IN ('queued','processing')
                  AND (expires_at IS NULL OR expires_at>?)""",
                (kind, self._clock())).fetchone()
            return int(row[0])

    def mark_exhausted(self, *, max_attempts: int = 3) -> int:
        now = self._clock()
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            jobs = con.execute("""SELECT job_id,kind FROM jobs WHERE attempts>=?
              AND (status='queued' OR (status='processing' AND lease_until<=?))""",
              (max_attempts, now)).fetchall()
            for row in jobs:
                con.execute("""UPDATE jobs SET status='error',error='retry_required',
                  finished_at=?,expires_at=?,updated_at=?,lease_until=NULL,lease_token=NULL
                  WHERE job_id=?""", (now,now + self.TTL[row["kind"]],now,row["job_id"]))
            return len(jobs)

    def cleanup(self) -> int:
        """Purge expired job blobs on every boot, during activity and while awake.

        Railway cannot run this when the container is asleep. Thus TTLs bound
        *readability* at all times, but byte deletion is only best-effort upon
        the next wake; strict time-based erasure requires an external scheduler.
        """
        now = self._clock()
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            expired = [r[0] for r in con.execute(
                "SELECT job_id FROM jobs WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))]
            for job_id in expired:
                con.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            # Limit rate-log retention when any maintenance runs.
            con.execute("DELETE FROM scan_rate_events WHERE timestamp<=?", (now - 86400,))
            # Metadata-only tombstones survive results for 48h but not forever.
            con.execute("DELETE FROM request_keys WHERE expires_at<=?", (now,))
            live_ids = {row[0] for row in con.execute("SELECT job_id FROM jobs")}
        for job_id in expired:
            for suffix in ("input", "result"):
                self._blob_path(job_id, suffix).unlink(missing_ok=True)
        # Clean crash-orphaned job blobs older than one hour. This can happen
        # if the container dies after writing an upload but before DB insert.
        # Never delete files for a still-recorded job (even if its worker hung).
        for suffix in ("input", "result"):
            for orphan in self.root.glob(f"*.{suffix}"):
                job_id = orphan.name[:-(len(suffix) + 1)]
                if job_id not in live_ids:
                    with contextlib.suppress(OSError):
                        if orphan.stat().st_mtime < time.time() - 3600:
                            orphan.unlink()
        # Clean crash-orphaned temporary files older than one hour.
        for p in self.root.glob("write-*"):
            with contextlib.suppress(OSError):
                if p.stat().st_mtime < time.time() - 3600:
                    p.unlink()
        return len(expired)
