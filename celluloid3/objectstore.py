"""The bucket: celluloid3's only durable substrate and its only coordinator.

celld's premise, ported verbatim: "Object-storage compare-and-swap ensures
that exactly one node owns a cell at a time, without a membership protocol,
failure detector, or consensus service."  Everything below is written against
the two properties celld requires of a backend -- conditional writes and
read-after-write consistency -- and nothing else.

Three backends implement the same contract:

* ``S3ObjectStore``       -- S3, R2, Tigris, and any S3-compatible endpoint,
                             using ``If-None-Match: *`` (conditional create)
                             and ``If-Match: <etag>`` (compare-and-swap).
* ``FileObjectStore``     -- a local directory that emulates both conditions
                             with ``flock``, so the whole system runs with
                             zero infrastructure (tests, demos, air-gapped
                             agents).
* ``InMemoryObjectStore`` -- a dict behind a lock, for unit tests.

GCS (generation preconditions) and Azure Blob (ETag conditions) expose the
same two primitives; a backend for either is a subclass away.
"""

from __future__ import annotations

import hashlib
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

try:  # POSIX advisory locking for the file backend's conditional writes
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

# S3 GETs are latency-bound, not bandwidth-bound: a restore that reads 200
# small objects one at a time is 200 round trips.  Every multi-object read
# below fans out across this pool instead.
DEFAULT_CONCURRENCY = 32


class PreconditionFailed(RuntimeError):
    """A conditional write lost the race (HTTP 412/409).

    This is the single primitive the whole coordination story rests on: the
    bucket accepts exactly one of the racing writes, so two agents cannot
    acquire the same cell.
    """


def etag_of(data: bytes) -> str:
    """S3-style ETag for a single-part object: the quoted MD5 of the body."""
    return '"' + hashlib.md5(data).hexdigest() + '"'


class ObjectStore(ABC):
    """Minimal object-storage contract: get / put / delete / list, plus
    conditional create and compare-and-swap."""

    concurrency = DEFAULT_CONCURRENCY

    # -- single object ----------------------------------------------------

    @abstractmethod
    def get_with_etag(self, key: str) -> tuple[bytes, str] | None:
        """Return (body, etag), or None if the key does not exist."""

    @abstractmethod
    def put(self, key: str, data: bytes, *, if_none_match: bool = False,
            if_match: str | None = None) -> str:
        """Write an object, returning its new ETag.

        ``if_none_match=True`` makes it a conditional create; ``if_match``
        makes it a compare-and-swap.  Either failing raises
        ``PreconditionFailed``.  With neither, it is a plain unconditional
        PUT -- which is what the hot data path uses (see cell.py: the epoch
        in the key is the fence, so segment writes need no condition).
        """

    @abstractmethod
    def delete(self, key: str, *, if_match: str | None = None) -> bool:
        """Delete an object; returns False if it was already gone."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """All keys under a prefix, lexicographically sorted."""

    @abstractmethod
    def list_prefixes(self, prefix: str) -> list[str]:
        """Immediate child "directories" of a prefix (S3 CommonPrefixes).

        Restore uses this to find the newest epoch prefix without listing
        every segment in the cell's history.
        """

    def get(self, key: str) -> bytes | None:
        entry = self.get_with_etag(key)
        return None if entry is None else entry[0]

    def exists(self, key: str) -> bool:
        return self.get_with_etag(key) is not None

    # -- fan-out ----------------------------------------------------------

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        """Fetch many objects concurrently.  Missing keys are omitted."""
        if not keys:
            return {}
        if len(keys) == 1:
            body = self.get(keys[0])
            return {} if body is None else {keys[0]: body}
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(keys))) as pool:
            bodies = list(pool.map(self.get, keys))
        return {k: b for k, b in zip(keys, bodies) if b is not None}

    def delete_many(self, keys: list[str]) -> int:
        if not keys:
            return 0
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(keys))) as pool:
            return sum(1 for ok in pool.map(self.delete, keys) if ok)


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryObjectStore(ObjectStore):
    """A dict behind a lock.  Conditional writes are exact; useful for tests
    and for measuring the system without any I/O in the way."""

    def __init__(self):
        self._objects: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.RLock()

    def get_with_etag(self, key):
        with self._lock:
            return self._objects.get(key)

    def put(self, key, data, *, if_none_match=False, if_match=None):
        tag = etag_of(data)
        with self._lock:
            current = self._objects.get(key)
            if if_none_match and current is not None:
                raise PreconditionFailed(f"{key} already exists")
            if if_match is not None and (current is None or current[1] != if_match):
                raise PreconditionFailed(f"{key} changed concurrently")
            self._objects[key] = (data, tag)
        return tag

    def delete(self, key, *, if_match=None):
        with self._lock:
            current = self._objects.get(key)
            if current is None:
                return False
            if if_match is not None and current[1] != if_match:
                raise PreconditionFailed(f"{key} changed concurrently")
            del self._objects[key]
            return True

    def list(self, prefix):
        with self._lock:
            return sorted(k for k in self._objects if k.startswith(prefix))

    def list_prefixes(self, prefix):
        with self._lock:
            keys = [k for k in self._objects if k.startswith(prefix)]
        return _common_prefixes(keys, prefix)


def _common_prefixes(keys: list[str], prefix: str) -> list[str]:
    out = set()
    for key in keys:
        rest = key[len(prefix):]
        if "/" in rest:
            out.add(prefix + rest.split("/", 1)[0] + "/")
    return sorted(out)


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class FileObjectStore(ObjectStore):
    """A directory that behaves like a bucket.

    Conditional create and compare-and-swap are emulated with a per-key
    ``flock``, so the semantics hold across processes on one machine.  This
    is what makes celluloid3 runnable with zero infrastructure -- the same code
    path that talks to S3 in production talks to ./agent-memory in a test.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"unsafe object key: {key!r}")
        return self.root / key

    def _lock_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.root / ".locks" / f"{digest}.lock"

    @contextmanager
    def _locked(self, key: str):
        if fcntl is None:  # pragma: no cover - Windows fallback
            with self._locks_guard:
                lock = self._locks.setdefault(key, threading.Lock())
            with lock:
                yield
            return
        path = self._lock_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def get_with_etag(self, key):
        path = self._path(key)
        try:
            data = path.read_bytes()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return None
        return data, etag_of(data)

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Leading dot: object keys never start with one, so an in-flight
        # write can never be listed as an object.
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: readers never see a torn object

    def put(self, key, data, *, if_none_match=False, if_match=None):
        path = self._path(key)
        if not if_none_match and if_match is None:
            self._write(path, data)  # unconditional: the fast data path
            return etag_of(data)
        with self._locked(key):
            current = self.get_with_etag(key)
            if if_none_match and current is not None:
                raise PreconditionFailed(f"{key} already exists")
            if if_match is not None and (current is None or current[1] != if_match):
                raise PreconditionFailed(f"{key} changed concurrently")
            self._write(path, data)
        return etag_of(data)

    def delete(self, key, *, if_match=None):
        path = self._path(key)
        with self._locked(key):
            current = self.get_with_etag(key)
            if current is None:
                return False
            if if_match is not None and current[1] != if_match:
                raise PreconditionFailed(f"{key} changed concurrently")
            path.unlink()
            return True

    def list(self, prefix):
        base = self.root
        keys = []
        start = base / prefix if prefix else base
        walk_root = start if start.is_dir() else start.parent
        if not walk_root.exists():
            return []
        for dirpath, _dirnames, filenames in os.walk(walk_root):
            for name in filenames:
                full = Path(dirpath) / name
                key = str(full.relative_to(base))
                if key.startswith(".locks/") or name.startswith("."):
                    continue
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def list_prefixes(self, prefix):
        directory = self.root / prefix if prefix else self.root
        if not directory.is_dir():
            return []
        return sorted(
            prefix + entry.name + "/"
            for entry in directory.iterdir()
            if entry.is_dir() and entry.name != ".locks"
        )


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------


class S3ObjectStore(ObjectStore):
    """Amazon S3 and every S3-compatible bucket (R2, Tigris, MinIO, ...).

    Conditional create uses ``If-None-Match: "*"`` and compare-and-swap uses
    ``If-Match: <etag>``; both are native S3 PutObject preconditions.  S3 has
    been strongly read-after-write consistent since December 2020, which is
    the second property the ownership protocol needs.

    For the low-latency end of the design, point ``endpoint_url``/``bucket``
    at an S3 Express One Zone directory bucket: same API, same conditional
    writes, single-digit-millisecond PUTs.
    """

    _CONFLICT_CODES = {
        "PreconditionFailed", "ConditionalRequestConflict", "412", "409",
    }

    def __init__(self, bucket: str, prefix: str = "", client=None, **client_kwargs):
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "the S3 backend needs boto3: pip install 'celluloid3[s3]'"
                ) from exc
            client = boto3.client("s3", **client_kwargs)
        self.client = client
        self._client_error = self._resolve_client_error(client)

    @staticmethod
    def _resolve_client_error(client):
        """The exception class this client raises for API errors.

        boto3 hangs it off ``client.exceptions``; taking it from there rather
        than importing botocore keeps the backend testable against any object
        that speaks the same five calls.
        """
        error = getattr(getattr(client, "exceptions", None), "ClientError", None)
        if error is not None:
            return error
        from botocore.exceptions import ClientError
        return ClientError

    def _key(self, key: str) -> str:
        return self.prefix + key

    @staticmethod
    def _is_missing(exc) -> bool:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in ("NoSuchKey", "404", "NotFound") or status == 404

    @classmethod
    def _is_conflict(cls, exc) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in cls._CONFLICT_CODES or status in (409, 412)

    def get_with_etag(self, key):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except self._client_error as exc:
            if self._is_missing(exc):
                return None
            raise
        return response["Body"].read(), response["ETag"]

    def put(self, key, data, *, if_none_match=False, if_match=None):
        kwargs = {}
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        try:
            response = self.client.put_object(
                Bucket=self.bucket, Key=self._key(key), Body=data, **kwargs
            )
        except self._client_error as exc:
            if self._is_conflict(exc):
                raise PreconditionFailed(f"{key}: {exc}") from exc
            raise
        return response["ETag"]

    def delete(self, key, *, if_match=None):
        kwargs = {"IfMatch": if_match} if if_match is not None else {}
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(key), **kwargs)
        except self._client_error as exc:
            if self._is_conflict(exc):
                raise PreconditionFailed(f"{key}: {exc}") from exc
            if self._is_missing(exc):
                return False
            raise
        return True

    def _paginate(self, prefix: str, delimiter: str = ""):
        paginator = self.client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": self.bucket, "Prefix": self._key(prefix)}
        if delimiter:
            kwargs["Delimiter"] = delimiter
        yield from paginator.paginate(**kwargs)

    def list(self, prefix):
        cut = len(self.prefix)
        return sorted(
            obj["Key"][cut:]
            for page in self._paginate(prefix)
            for obj in page.get("Contents", [])
        )

    def list_prefixes(self, prefix):
        cut = len(self.prefix)
        return sorted(
            entry["Prefix"][cut:]
            for page in self._paginate(prefix, delimiter="/")
            for entry in page.get("CommonPrefixes", [])
        )


# ---------------------------------------------------------------------------


def open_object_store(uri: str | os.PathLike | ObjectStore, **kwargs) -> ObjectStore:
    """Resolve a URI to a backend.

    ``s3://bucket/prefix``  -> S3ObjectStore (also R2/Tigris/MinIO via
                               ``endpoint_url=``)
    ``mem://name``          -> InMemoryObjectStore
    anything else           -> FileObjectStore at that path
    """
    if isinstance(uri, ObjectStore):
        return uri
    if not isinstance(uri, (str, os.PathLike)):
        raise TypeError(
            f"expected an ObjectStore, a URI or a path, got {type(uri).__name__}"
        )
    text = str(uri)
    if text.startswith("s3://"):
        rest = text[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        return S3ObjectStore(bucket, prefix, **kwargs)
    if text.startswith("mem://"):
        return _shared_memory_store(text)
    if text.startswith("file://"):
        text = text[len("file://"):]
    return FileObjectStore(text)


_MEMORY_STORES: dict[str, InMemoryObjectStore] = {}
_MEMORY_GUARD = threading.Lock()


def _shared_memory_store(uri: str) -> InMemoryObjectStore:
    """``mem://name`` returns the same bucket for the same name, so two
    agents in one process can contend for a cell the way two nodes would."""
    with _MEMORY_GUARD:
        return _MEMORY_STORES.setdefault(uri, InMemoryObjectStore())
