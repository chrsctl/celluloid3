"""The two primitives everything rests on: conditional create and
compare-and-swap.  Every backend must implement them identically."""

import pytest

from celluloid3 import (
    FileObjectStore, InMemoryObjectStore, PreconditionFailed, open_object_store,
)
from celluloid3.objectstore import etag_of

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
    for key in ["spaces/t/lanes/planner/e1/0.tqs", "spaces/t/lanes/planner/e1/1.tqs",
                "spaces/t/lanes/planner/e2/0.tqs", "spaces/t/lanes/coder/owner.json"]:
        store.put(key, b"x")
    assert store.list("spaces/t/lanes/planner/e1/") == [
        "spaces/t/lanes/planner/e1/0.tqs", "spaces/t/lanes/planner/e1/1.tqs"
    ]
    # one LIST over the space is how an agent catches up on the whole team
    assert len(store.list("spaces/t/lanes/")) == 4
    assert store.list_prefixes("spaces/t/lanes/") == [
        "spaces/t/lanes/coder/", "spaces/t/lanes/planner/"
    ]


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


# -- a mistyped store is not a local directory ------------------------------

@pytest.mark.parametrize("uri", ["gs://my-bucket/memory", "redis://host/db",
                                 "azure://container/prefix"])
def test_a_scheme_this_build_does_not_implement_is_an_error(uri, tmp_path, monkeypatch):
    """It used to become a local directory named after the typo and report
    every write durable -- the loudest possible silence."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as raised:
        open_object_store(uri)
    assert uri.split("://")[0] + "://" in str(raised.value)
    assert list(tmp_path.iterdir()) == []          # and created nothing


def test_paths_are_still_paths(tmp_path, monkeypatch):
    """Only a real ``scheme://`` is a scheme: a relative path, a home path and
    a Windows-shaped string are all just paths."""
    monkeypatch.chdir(tmp_path)
    for path in ("./relative/store", "plain", r"C:\Users\dev\memory"):
        assert isinstance(open_object_store(path), FileObjectStore)
    assert isinstance(open_object_store(tmp_path / "as-a-path"), FileObjectStore)
    assert isinstance(open_object_store(f"file://{tmp_path}/via-file"), FileObjectStore)
    assert isinstance(open_object_store(InMemoryObjectStore()), InMemoryObjectStore)


def test_the_directory_an_earlier_typo_created_is_still_openable(tmp_path, monkeypatch):
    """``gs://b/p`` used to create ``./gs:/b/p``.  Those stores exist; the
    error says where they are, and they open as what they are -- a path."""
    monkeypatch.chdir(tmp_path)
    stranded = tmp_path / "gs:" / "b" / "p"
    stranded.mkdir(parents=True)
    with pytest.raises(ValueError) as raised:
        open_object_store("gs://b/p")
    assert "gs:/b/p" in str(raised.value)
    assert isinstance(open_object_store("gs:/b/p"), FileObjectStore)


def test_store_scheme_names_what_it_sees(tmp_path):
    from celluloid3.objectstore import store_scheme
    assert store_scheme("s3://bucket/prefix") == "s3"
    assert store_scheme("file:///tmp/x") == "file"
    assert store_scheme("./agent-memory") is None
    assert store_scheme(r"C:\Users\dev\memory") is None
    assert store_scheme(tmp_path) is None

