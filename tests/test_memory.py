"""The agent-facing API: remember, recall, forget, checkpoints, time travel,
metadata filtering, attachments."""

import time

import numpy as np
import pytest

from celluloid3 import MemoryLayer

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
