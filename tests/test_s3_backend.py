"""The S3 backend against a stub client that enforces real S3 semantics.

This exercises the conditional-write wiring -- ``If-None-Match: "*"`` for a
create, ``If-Match: <etag>`` for a compare-and-swap, 412/409 mapped to
PreconditionFailed -- without needing a bucket or botocore.
"""

import io

import pytest

from celluloid3 import PreconditionFailed, S3ObjectStore
from celluloid3.objectstore import etag_of


class StubClientError(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class StubS3:
    """Enough of the S3 API to hold the backend honest."""

    class exceptions:
        ClientError = StubClientError

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple] = []

    def get_object(self, Bucket, Key):
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise StubClientError("NoSuchKey", 404)
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": etag_of(body)}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None):
        self.calls.append(("put", Key, IfNoneMatch, IfMatch))
        current = self.objects.get(Key)
        if IfNoneMatch == "*" and current is not None:
            raise StubClientError("PreconditionFailed", 412)
        if IfMatch is not None and (current is None or etag_of(current) != IfMatch):
            raise StubClientError("PreconditionFailed", 412)
        self.objects[Key] = Body
        return {"ETag": etag_of(Body)}

    def delete_object(self, Bucket, Key, IfMatch=None):
        self.calls.append(("delete", Key, IfMatch))
        current = self.objects.get(Key)
        if IfMatch is not None and (current is None or etag_of(current) != IfMatch):
            raise StubClientError("PreconditionFailed", 412)
        self.objects.pop(Key, None)
        return {}

    def get_paginator(self, name):
        stub = self

        class Paginator:
            def paginate(self, Bucket, Prefix, Delimiter=None):
                keys = sorted(k for k in stub.objects if k.startswith(Prefix))
                if not Delimiter:
                    yield {"Contents": [{"Key": k} for k in keys]}
                    return
                contents, prefixes = [], set()
                for key in keys:
                    rest = key[len(Prefix):]
                    if Delimiter in rest:
                        prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
                    else:
                        contents.append({"Key": key})
                yield {"Contents": contents,
                       "CommonPrefixes": [{"Prefix": p} for p in sorted(prefixes)]}

        return Paginator()


@pytest.fixture
def s3():
    client = StubS3()
    return S3ObjectStore("my-bucket", "agent-memory", client=client), client


def test_prefix_is_applied_to_every_key(s3):
    store, client = s3
    store.put("cells/a/owner.json", b"{}")
    assert "agent-memory/cells/a/owner.json" in client.objects
    assert store.get("cells/a/owner.json") == b"{}"


def test_missing_key_is_none_not_an_error(s3):
    store, _client = s3
    assert store.get("nope") is None
    assert store.get_with_etag("nope") is None


def test_conditional_create_maps_412_to_precondition_failed(s3):
    store, client = s3
    store.put("owner", b"first", if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put("owner", b"second", if_none_match=True)
    assert client.calls[-1] == ("put", "agent-memory/owner", "*", None)


def test_cas_uses_if_match(s3):
    store, client = s3
    tag = store.put("owner", b"v1")
    store.put("owner", b"v2", if_match=tag)
    assert client.calls[-1] == ("put", "agent-memory/owner", None, tag)
    with pytest.raises(PreconditionFailed):
        store.put("owner", b"v3", if_match=tag)


def test_data_path_sends_no_conditions(s3):
    """The hot path must be a plain PUT: the epoch in the key is the fence."""
    store, client = s3
    store.put("cells/a/ltx/e0000000001/000000000000.tqs", b"segment")
    _op, _key, if_none_match, if_match = client.calls[-1]
    assert if_none_match is None and if_match is None


def test_list_strips_the_store_prefix(s3):
    store, _client = s3
    for key in ["cells/a/ltx/e1/0.tqs", "cells/a/ltx/e2/0.tqs"]:
        store.put(key, b"x")
    assert store.list("cells/a/ltx/") == ["cells/a/ltx/e1/0.tqs",
                                          "cells/a/ltx/e2/0.tqs"]
    assert store.list_prefixes("cells/a/ltx/") == ["cells/a/ltx/e1/",
                                                   "cells/a/ltx/e2/"]


def test_a_shared_space_runs_on_the_s3_backend(s3):
    from celluloid3 import HashingEmbedder, MemoryLayer
    store, _client = s3
    embed = HashingEmbedder(dim=64)
    planner = MemoryLayer(store, space="team", agent="planner", embedder=embed,
                          dim=64, ttl=60)
    planner.remember("the bucket is the database")
    coder = MemoryLayer(store, space="team", agent="coder", embedder=embed,
                        ttl=60)
    hit = coder.recall("what is the database?", k=1)[0]
    assert hit.fragment.text == "the bucket is the database"
    assert hit.authors == ("planner",)


def test_lane_keys_carry_the_agent_and_the_epoch(s3):
    from celluloid3 import HashingEmbedder, MemoryLayer
    store, client = s3
    mem = MemoryLayer(store, space="team", agent="planner",
                      embedder=HashingEmbedder(dim=64), dim=64, ttl=60)
    mem.remember("a fact")
    segments = [k for k in client.objects if k.endswith(".tqs")]
    assert segments == [
        "agent-memory/spaces/team/lanes/planner/e0000000001/000000000000.tqs"
    ]
