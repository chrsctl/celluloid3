"""The CLI, end to end against a local directory that behaves like a bucket."""

import json

import pytest

from celluloid3.__main__ import main


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "agent-memory")


def run(capsys, store, *argv):
    code = main(["--store", store, *argv])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return captured.out


def test_init_remember_recall(capsys, store):
    stats = json.loads(run(capsys, store, "init", "--dim", "128"))
    assert stats["dim"] == 128 and stats["fragments"] == 0

    run(capsys, store, "remember", "the ci pipeline breaks on pandas 2.3",
        "--meta", "kind=incident")
    out = run(capsys, store, "recall", "what breaks ci?", "-k", "1")
    assert "pandas" in out

    out = run(capsys, store, "recall", "anything", "-k", "5", "--where", "kind=other")
    assert out.strip() == ""


def test_many_memories_are_one_commit(capsys, store):
    run(capsys, store, "remember", "fact one", "fact two", "fact three")
    out = run(capsys, store, "log")
    assert len([line for line in out.splitlines() if line.strip()]) == 1
    assert "+3 -0" in out


def test_checkpoint_and_time_travel(capsys, store):
    run(capsys, store, "remember", "the original plan")
    run(capsys, store, "checkpoint", "v1")
    run(capsys, store, "remember", "the revised plan")
    assert "revised" in run(capsys, store, "recall", "plan", "-k", "2")
    assert "revised" not in run(capsys, store, "recall", "plan", "-k", "2", "--at", "v1")
    assert run(capsys, store, "checkpoints").strip() == "v1"


def test_forget_then_stats_and_spaces(capsys, store):
    run(capsys, store, "remember", "a fact worth forgetting")
    fid = run(capsys, store, "recall", "forgetting", "-k", "1").split()[1]
    run(capsys, store, "forget", fid)
    assert json.loads(run(capsys, store, "stats"))["fragments"] == 0
    assert run(capsys, store, "spaces").strip() == "shared"


def test_forgetting_something_absent_fails(capsys, store):
    main(["--store", store, "init"])
    assert main(["--store", store, "forget", "00" * 32]) == 1



def test_owner_record_is_readable(capsys, store):
    run(capsys, store, "remember", "a fact")
    record = json.loads(run(capsys, store, "owner"))
    assert record["owned"] is False        # the previous command handed it back
    assert record["epoch"] >= 1


def test_agents_share_a_space(capsys, store):
    """Two shells, two agents, one memory."""
    run(capsys, store, "-a", "planner", "remember", "the customer wants SSO")
    run(capsys, store, "-a", "coder", "remember", "the auth service has no OIDC")
    assert sorted(run(capsys, store, "-a", "coder", "agents").split()) == \
        ["(you)", "coder", "planner"]
    # the coder recalls what the planner learned, and knows who learned it
    out = run(capsys, store, "-a", "coder", "recall", "what does the customer want")
    assert "SSO" in out and "[planner]" in out
    # ...and can narrow to one agent
    assert "SSO" not in run(capsys, store, "-a", "coder", "recall",
                            "customer", "--by", "coder")


def test_separate_spaces_share_nothing(capsys, store):
    run(capsys, store, "-s", "private", "remember", "a private thought")
    run(capsys, store, "-s", "team", "remember", "a shared thought")
    assert sorted(run(capsys, store, "spaces").split()) == ["private", "team"]
    assert "private" not in run(capsys, store, "-s", "team", "recall", "thought")


def test_compact_and_gc(capsys, store):
    """Each CLI invocation is its own activation, and the base written at
    sequence zero is already a compaction -- so there is rarely anything for
    an explicit compact to fold."""
    run(capsys, store, "remember", *[f"fact {i}" for i in range(5)])
    assert "already one object" in run(capsys, store, "compact")
    assert "deleted" in run(capsys, store, "gc")
    assert json.loads(run(capsys, store, "stats"))["fragments"] == 5
