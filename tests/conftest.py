import pytest

from celloid3 import (
    FileObjectStore, HashingEmbedder, InMemoryObjectStore, MemoryLayer, ObjectStore,
)

DIM = 256


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=DIM)


@pytest.fixture
def bucket(tmp_path):
    """A local directory that behaves like a bucket, with real conditional
    writes.  Same code path the S3 backend uses."""
    return FileObjectStore(tmp_path / "bucket")


@pytest.fixture
def ram_bucket():
    return InMemoryObjectStore()


@pytest.fixture
def mem(bucket, embedder):
    return MemoryLayer(bucket, cell="agent", embedder=embedder, dim=DIM)


def open_layer(bucket, embedder, cell="agent", **kwargs):
    return MemoryLayer(bucket, cell=cell, embedder=embedder, **kwargs)


class CountingStore(ObjectStore):
    """Wraps a bucket and counts round trips.

    Round-trip counts are the thesis of this design -- one PUT per commit,
    zero GETs per recall, a handful of GETs per wake -- so the tests assert
    them directly rather than trusting the prose.
    """

    def __init__(self, inner):
        self.inner = inner
        self.gets = self.puts = self.deletes = self.lists = 0
        self.conditional_puts = 0

    def reset(self):
        self.gets = self.puts = self.deletes = self.lists = 0
        self.conditional_puts = 0

    # -- counted delegation ----------------------------------------------

    def get_with_etag(self, key):
        self.gets += 1
        return self.inner.get_with_etag(key)

    def get(self, key):
        self.gets += 1
        return self.inner.get(key)

    def exists(self, key):
        self.gets += 1
        return self.inner.exists(key)

    def put(self, key, data, *, if_none_match=False, if_match=None):
        self.puts += 1
        if if_none_match or if_match is not None:
            self.conditional_puts += 1
        return self.inner.put(key, data, if_none_match=if_none_match,
                              if_match=if_match)

    def delete(self, key, *, if_match=None):
        self.deletes += 1
        return self.inner.delete(key, if_match=if_match)

    def list(self, prefix):
        self.lists += 1
        return self.inner.list(prefix)

    def list_prefixes(self, prefix):
        self.lists += 1
        return self.inner.list_prefixes(prefix)

    def get_many(self, keys):
        self.gets += len(keys)
        return self.inner.get_many(keys)

    def delete_many(self, keys):
        self.deletes += len(keys)
        return self.inner.delete_many(keys)


@pytest.fixture
def counted(bucket):
    return CountingStore(bucket)
