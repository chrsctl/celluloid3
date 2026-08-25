"""The agent-facing API: remember, recall, forget, checkpoints, time travel,
metadata filtering, attachments."""

import time

import numpy as np
import pytest

from celluloid3 import HashingEmbedder, MemoryLayer
from celluloid3.__main__ import main

FACTS = [
    "the production deploy failed because DATABASE_URL was missing",
    "the user prefers dark mode in every interface",
    "quarterly revenue target for Q3 is two million dollars",
    "the staging cluster runs kubernetes version 1.29",
    "the user's dog is named biscuit and likes tennis balls",
]


def fill(mem):
    with mem.batch("seed"):
        return [mem.remember(text, metadata={"i": i}) for i, text in enumerate(FACTS)]


def test_remember_recall_roundtrip(mem):
    fill(mem)
    hits = mem.recall("why did the deploy fail?", k=2)
    assert "deploy failed" in hits[0].fragment.text
    assert any("biscuit" in h.fragment.text
               for h in mem.recall("the dog that likes tennis balls", k=2))


def test_metadata_and_parents_survive_a_wake(bucket, embedder):
    first = MemoryLayer(bucket, space="team", agent="a", embedder=embedder,
                        dim=256, ttl=60)
    root = first.remember("the incident started at 02:14", metadata={"kind": "incident"})
    child = first.remember("root cause: an expired certificate",
                           metadata={"kind": "postmortem"}, parents=(root,))
    first.hibernate()

    second = MemoryLayer(bucket, space="team", agent="a", embedder=embedder, ttl=60)
    fragment = second.get(child)
    assert fragment.metadata == {"kind": "postmortem"}
    assert fragment.parents == (root,)


def test_identical_memories_deduplicate(mem):
    """Fragments are named by the hash of their content -- text, metadata and
    parents, deliberately not the timestamp -- so learning the same thing
    twice costs nothing however far apart the two occasions are."""
    first = mem.remember("a repeated observation", metadata={"kind": "note"})
    time.sleep(0.01)
    second = mem.remember("a repeated observation", metadata={"kind": "note"})
    assert first == second
    assert len(mem) == 1
    # ...but a different note, or different metadata, is a different memory
    assert mem.remember("a repeated observation", metadata={"kind": "other"}) != first
    assert len(mem) == 2


def test_short_ids_resolve_like_git_shas(mem):
    fill(mem)
    fid = mem.remember("a memory to address by its short id")
    assert mem.resolve_id(fid[:12]) == fid
    assert mem.resolve_id(fid) == fid
    with pytest.raises(KeyError):
        mem.resolve_id("ffffffffffffffff" * 4)
    with pytest.raises(KeyError, match="ambiguous"):
        mem.resolve_id("")          # every id starts with the empty prefix


def test_forget_is_auditable(mem):
    fill(mem)
    secret = mem.remember("temporary secret: the launch code is 1234")
    mem.checkpoint("with-secret")
    assert any("launch code" in h.fragment.text for h in mem.recall("launch code", k=3))

    assert mem.forget(secret) is True
    assert mem.forget(secret) is False
    assert not any("launch code" in h.fragment.text
                   for h in mem.recall("launch code", k=5))
    # the log still has it: recoverable, and visible to time-travel recall
    assert mem.get(secret, at="with-secret") is not None
    assert any("launch code" in h.fragment.text
               for h in mem.recall("launch code", k=5, at="with-secret"))


def test_time_travel_by_checkpoint_and_by_cut(mem):
    fill(mem)
    early = mem.checkpoint("v1")
    mem.remember("a fact learned after v1 shipped")
    assert len(mem) == len(FACTS) + 1
    assert len(mem.recall("fact", k=99, at="v1")) == len(FACTS)
    assert len(mem.recall("fact", k=99, at=str(early))) == len(FACTS)
    assert len(mem.recall("fact", k=99, at=early)) == len(FACTS)


def test_head_relative_time_travel(mem):
    fill(mem)
    mem.remember("the newest fact")
    before = mem.recall("newest", k=99, at="HEAD~1")
    assert not any("newest fact" in h.fragment.text for h in before)


def test_checkpoint_names_are_immutable(mem):
    fill(mem)
    mem.checkpoint("release")
    with pytest.raises(ValueError):
        mem.checkpoint("release")
    assert mem.checkpoints() == ["release"]


def test_where_dict_filter(mem):
    with mem.batch():
        mem.remember("deploy checklist for the payments service",
                     metadata={"kind": "runbook", "service": "payments"})
        mem.remember("deploy checklist for the search service",
                     metadata={"kind": "runbook", "service": "search"})
        mem.remember("random chatter about deploy weather", metadata={"kind": "chat"})

    hits = mem.recall("deploy checklist", k=5, where={"service": "search"})
    assert len(hits) == 1 and "search service" in hits[0].fragment.text
    hits = mem.recall("deploy checklist", k=5, where={"kind": "runbook"})
    assert len(hits) == 2


def test_where_callable_filter(mem):
    with mem.batch():
        old = mem.remember("ancient fact", metadata={"epoch": 1})
        new = mem.remember("recent fact", metadata={"epoch": 9})
    hits = mem.recall("fact", k=5, where=lambda f: f.metadata.get("epoch", 0) > 5)
    assert [h.fragment.id for h in hits] == [new] != [old]


def test_filtered_scan_reaches_past_the_top_k(mem):
    """The filter is pushed into an exhaustive scan, so a match that ranks
    below many non-matches is still found."""
    with mem.batch():
        for i in range(10):
            mem.remember(f"kubernetes cluster upgrade note number {i}",
                         metadata={"team": "infra"})
        mem.remember("totally unrelated gardening tips", metadata={"team": "web"})
    hits = mem.recall("kubernetes cluster upgrade", k=3, where={"team": "web"})
    assert len(hits) == 1 and "gardening" in hits[0].fragment.text


def test_attachments_live_beside_the_log(bucket, embedder):
    """Segments are replayed in full on every wake, so big payloads go next to
    the log rather than into it."""
    first = MemoryLayer(bucket, space="team", agent="a", embedder=embedder,
                        dim=256, ttl=60)
    payload = b"\x89PNG not really an image" * 500
    key = first.attach("diagram.png", payload)
    first.remember("architecture diagram of the ingest service",
                   metadata={"attachment": key})
    assert first.attach("diagram.png", payload) == key   # content-addressed
    first.hibernate()

    second = MemoryLayer(bucket, space="team", agent="a", embedder=embedder, ttl=60)
    hit = second.recall("architecture diagram", k=1)[0]
    assert second.get_attachment(hit.fragment.metadata["attachment"]) == payload
    assert second.get_attachment("cells/a/blobs/00/nope/x.png") is None


def test_spaces_are_isolated(bucket, embedder):
    """Different spaces share nothing -- a team's memory and a private one are
    just two spaces in the same bucket."""
    planner = MemoryLayer(bucket, space="planning", agent="a", embedder=embedder,
                         dim=256, ttl=60)
    coder = MemoryLayer(bucket, space="engineering", agent="a", embedder=embedder,
                      ttl=60)
    planner.remember("the planner remembers the roadmap")
    coder.remember("the coder remembers the stack trace")
    assert len(planner) == len(coder) == 1
    assert not any("roadmap" in h.fragment.text for h in coder.recall("roadmap", k=5))


def test_store_config_is_created_once_and_then_enforced(bucket, embedder, tmp_path):
    from celluloid3 import FileObjectStore
    MemoryLayer(bucket, space="team", agent="a", embedder=embedder, dim=256)
    # the store's codebook is fixed at creation; a mismatched reopen is refused
    with pytest.raises(ValueError):
        MemoryLayer(bucket, space="team", agent="a", embedder=embedder, dim=64)
    # ...and a brand-new store has to be told the dimension somehow
    with pytest.raises(ValueError):
        MemoryLayer(FileObjectStore(tmp_path / "empty"), space="team", agent="a")


def test_config_creation_is_a_race_only_one_agent_wins(bucket, embedder):
    """Conditional create: two agents opening a fresh bucket at once cannot
    end up with two incompatible codebooks."""
    from celluloid3.memory import load_or_create_config
    first = load_or_create_config(bucket, 256, 4, 11, embedder)
    second = load_or_create_config(bucket, None, 8, 99, embedder)
    assert first == second
    assert second["bit_width"] == 4 and second["rotation_seed"] == 11


def test_recall_without_an_embedder_needs_a_vector(bucket):
    mem = MemoryLayer(bucket, space="team", agent="a", dim=32, ttl=60)
    fid = mem.remember("stored with an explicit vector",
                       embedding=np.arange(32, dtype=float))
    with pytest.raises(ValueError):
        mem.recall("a text query")
    hits = mem.recall(np.arange(32, dtype=float), k=1)
    assert hits[0].fragment.id == fid


def test_context_manager_hands_the_cell_back(bucket, embedder):
    with MemoryLayer(bucket, space="team", agent="a", embedder=embedder,
                        dim=256, ttl=60) as mem:
        mem.remember("written inside the with-block")
        assert mem.space.active
    record, _etag = mem.space.ownership.read()
    assert record.owned is False

    with MemoryLayer(bucket, space="team", agent="a", embedder=embedder, ttl=60) as second:
        assert len(second) == 1


def test_stats_report_the_compression_and_the_wake(mem):
    fill(mem)
    stats = mem.stats()
    assert stats["fragments"] == len(FACTS)
    assert stats["compression"] >= 7          # 4-bit vs float32, minus headers
    assert stats["bit_width"] == 4
    assert stats["epoch"] == 1
    assert stats["head"] == "agent:e1:0"


def test_history_reads_like_a_log(mem):
    fill(mem)
    mem.remember("one more")
    entries = mem.history()
    assert entries[0]["seq"] == 1 and entries[0]["added"] == 1
    assert entries[-1]["note"].startswith("base")
    assert entries[-1]["added"] == len(FACTS)


def test_hibernated_layer_reactivates_on_use(mem):
    fill(mem)
    mem.hibernate()
    assert not mem.space.active
    assert len(mem.recall("deploy", k=2)) == 2      # woke itself back up
    assert mem.epoch == 2


# -- one call to the embedder, not one per memory ---------------------------

class SerialEmbedder:
    """An embedder that takes one text at a time, and counts the calls.

    An embeddings API charges a round trip per call, so the count *is* the
    cost this batching is about.
    """

    def __init__(self, dim=256):
        self.inner = HashingEmbedder(dim=dim)
        self.dim = dim
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, text):
        self.calls += 1
        self.texts_embedded += 1
        return self.inner(text)


class BatchingEmbedder(SerialEmbedder):
    """...and one that takes a list, the way every embeddings API does."""

    def embed_many(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self.inner(text) for text in texts]


def batching_layer(bucket, batched=True, space="team"):
    embedder = (BatchingEmbedder if batched else SerialEmbedder)()
    return MemoryLayer(bucket, space=space, agent="a", embedder=embedder,
                       dim=256, ttl=60), embedder


def test_remember_many_calls_a_batching_embedder_once(bucket):
    mem, embedder = batching_layer(bucket)
    ids = mem.remember_many([f"memory number {i}" for i in range(50)])
    assert embedder.calls == 1                   # not 50
    assert embedder.texts_embedded == 50
    assert len(set(ids)) == 50
    assert len(mem.recall("memory number 7", k=1)) == 1
    assert len(mem) == 50


def test_remember_many_falls_back_to_one_call_per_text(bucket):
    """No ``embed_many``, no problem -- and no difference in what is stored."""
    texts = [f"memory number {i}" for i in range(50)]
    serial, serial_embedder = batching_layer(bucket, batched=False)
    assert not hasattr(serial_embedder, "embed_many")
    serial_ids = serial.remember_many(texts)
    assert serial_embedder.calls == 50
    serial.hibernate()

    batched, _ = batching_layer(bucket, space="other")
    assert batched.remember_many(texts) == serial_ids


def test_remember_many_costs_one_put_and_one_ownership_read(counted, embedder):
    """The bucket side does not move: one segment, one gate read, 50 memories."""
    mem = MemoryLayer(counted, space="team", agent="a", embedder=embedder,
                      dim=256, ttl=60)
    mem.activate()
    counted.reset()
    mem.remember_many([f"memory number {i}" for i in range(50)])
    assert counted.puts == 1
    assert counted.gets == 1
    assert counted.conditional_puts == 0


def test_remember_many_returns_ids_in_input_order(mem):
    texts = ["alpha fact", "beta fact", "gamma fact", "beta fact"]
    ids = mem.remember_many(texts)
    assert len(ids) == 4
    assert ids[1] == ids[3]                      # an exact repeat is free
    assert ids[0] != ids[1] != ids[2]
    assert len(mem) == 3                         # ...and stores one record
    assert [mem.get(fid).text for fid in ids[:3]] == texts[:3]


def test_remember_many_rejects_a_mismatched_metadata_list(mem):
    with pytest.raises(ValueError):
        mem.remember_many(["one", "two", "three"],
                          metadata=[{"i": 0}, {"i": 1}])
    assert mem.stats()["pending"] == 0           # nothing staged before the raise
    assert len(mem) == 0


def test_remember_many_takes_metadata_per_text_or_for_all(mem):
    ids = mem.remember_many(["runbook one", "runbook two"],
                            metadata=[{"kind": "runbook", "n": 1},
                                      {"kind": "runbook", "n": 2}])
    assert mem.get(ids[1]).metadata["n"] == 2
    shared = mem.remember_many(["chat one", "chat two"], metadata={"kind": "chat"})
    assert {mem.get(fid).metadata["kind"] for fid in shared} == {"chat"}
    assert len(mem.recall("runbook", k=5, where={"kind": "runbook"})) == 2


def test_remember_many_inside_a_batch_still_writes_one_segment(mem):
    with mem.batch("ingest"):
        mem.remember_many(["one", "two"])
        mem.remember_many(["three", "four"])
        mem.remember("five")
    assert mem.stats()["segments_this_epoch"] == 1
    assert len(mem) == 5
    assert mem.history()[-1]["note"] == "base: ingest"


def test_remember_many_takes_vectors_the_caller_already_has(bucket):
    """``embeddings=`` skips the embedder entirely -- and is still checked."""
    mem, embedder = batching_layer(bucket)
    vectors = [np.ones(256) * i for i in (1.0, 2.0)]
    ids = mem.remember_many(["one", "two"], embeddings=vectors)
    assert embedder.calls == 0
    assert len(ids) == 2
    with pytest.raises(ValueError):
        mem.remember_many(["one", "two", "three"], embeddings=vectors)


def test_remember_many_respects_the_layers_commit_policy(bucket, embedder):
    """It group-commits like ``batch()`` but does not overrule the layer: a
    deferred layer stays deferred, and a ``flush_every`` layer gets one
    segment for the call, not one per threshold."""
    deferred = MemoryLayer(bucket, space="team", agent="a", embedder=embedder,
                           dim=256, ttl=60, durable=False)
    deferred.remember_many(["one", "two", "three"])
    assert deferred.stats()["pending"] == 3
    assert deferred.stats()["segments_this_epoch"] == 0
    deferred.flush()
    assert deferred.stats()["pending"] == 0
    assert deferred.stats()["segments_this_epoch"] == 1

    every_ten = MemoryLayer(bucket, space="team", agent="b", embedder=embedder,
                            ttl=60, flush_every=10)
    every_ten.remember_many([f"memory number {i}" for i in range(50)])
    assert every_ten.stats()["segments_this_epoch"] == 1
    assert every_ten.stats()["pending"] == 0


def test_remember_many_rejects_a_string_where_a_list_belongs(mem):
    """Three texts and a three-character string is a length match by accident;
    zipping its characters as metadata is never what the caller meant."""
    with pytest.raises(ValueError):
        mem.remember_many(["one", "two", "three"], metadata="abc")
    assert mem.stats()["pending"] == 0


def test_cli_remember_routes_through_remember_many(tmp_path, capsys):
    """``remember one two three`` is the shape the CLI shows off: one call to
    the embedder, one segment, one id printed per text."""
    store = str(tmp_path / "agent-memory")
    assert main(["--store", store, "-a", "coder", "remember",
                 "one", "two", "three"]) == 0
    assert len(capsys.readouterr().out.split()) == 3

    reader = MemoryLayer(store, space="shared", agent="reader", ttl=60)
    assert len(reader._state(fresh=True).fragments) == 3
    assert len(reader.space.state.log) == 1      # three texts, one segment

