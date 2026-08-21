"""The two primitives everything rests on: conditional create and
compare-and-swap.  Every backend must implement them identically."""

import pytest

from celloid3 import (
    FileObjectStore, InMemoryObjectStore, PreconditionFailed, open_object_store,
)
from celloid3.objectstore import etag_of

BACKENDS = ["file", "ram"]


@pytest.fixture(params=BACKENDS)
def store(request, tmp_path):
    if request.param == "file":
        return FileObjectStore(tmp_path / "bucket")
    return InMemoryObjectStore()


def test_put_get_delete(store):
    assert store.get("a/b") is None
    tag = store.put("a/b", b"hello")
    assert tag == etag_of(b"hello")
    assert store.get("a/b") == b"hello"
    assert store.exists("a/b")
    assert store.delete("a/b") is True
    assert store.delete("a/b") is False
    assert store.get("a/b") is None


def test_conditional_create_admits_exactly_one(store):
    store.put("owner", b"first", if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put("owner", b"second", if_none_match=True)
    assert store.get("owner") == b"first"


def test_compare_and_swap(store):
    tag = store.put("owner", b"v1")
    with pytest.raises(PreconditionFailed):
        store.put("owner", b"v2", if_match=etag_of(b"stale"))
    new_tag = store.put("owner", b"v2", if_match=tag)
    assert store.get("owner") == b"v2"
    # the old etag is now stale: the loser of a race sees exactly this
    with pytest.raises(PreconditionFailed):
        store.put("owner", b"v3", if_match=tag)
    assert new_tag == etag_of(b"v2")


def test_cas_on_missing_object_fails(store):
    with pytest.raises(PreconditionFailed):
        store.put("nothing", b"x", if_match=etag_of(b"x"))


def test_conditional_delete(store):
    tag = store.put("k", b"v")
    with pytest.raises(PreconditionFailed):
        store.delete("k", if_match=etag_of(b"other"))
    assert store.delete("k", if_match=tag) is True


def test_list_and_prefixes(store):
    for key in ["cells/a/ltx/e1/0.tqs", "cells/a/ltx/e1/1.tqs",
                "cells/a/ltx/e2/0.tqs", "cells/b/owner.json"]:
        store.put(key, b"x")
    assert store.list("cells/a/ltx/e1/") == [
        "cells/a/ltx/e1/0.tqs", "cells/a/ltx/e1/1.tqs"
    ]
    assert store.list_prefixes("cells/a/ltx/") == [
        "cells/a/ltx/e1/", "cells/a/ltx/e2/"
    ]
    assert store.list_prefixes("cells/") == ["cells/a/", "cells/b/"]


def test_get_many_is_a_fan_out_not_a_loop(store):
    keys = [f"seg/{i}" for i in range(20)]
    for key in keys:
        store.put(key, key.encode())
    fetched = store.get_many(keys + ["seg/missing"])
    assert len(fetched) == 20
    assert fetched["seg/7"] == b"seg/7"


def test_file_backend_rejects_escaping_keys(tmp_path):
    store = FileObjectStore(tmp_path / "bucket")
    with pytest.raises(ValueError):
        store.put("../escape", b"x")


def test_open_object_store_uris(tmp_path):
    assert isinstance(open_object_store(str(tmp_path)), FileObjectStore)
    assert isinstance(open_object_store(f"file://{tmp_path}"), FileObjectStore)
    assert isinstance(open_object_store("mem://x"), InMemoryObjectStore)
    # mem:// is shared by name, so two agents can contend for one cell
    assert open_object_store("mem://x") is open_object_store("mem://x")
    assert open_object_store("mem://x") is not open_object_store("mem://y")
