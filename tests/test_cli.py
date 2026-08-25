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


# -- wrong input fails loudly -----------------------------------------------

def test_a_typo_in_store_is_not_an_empty_memory(capsys, tmp_path):
    """Nothing found, nothing said, exit 0 -- and a new empty store left
    behind -- is indistinguishable from a memory that is simply empty."""
    missing = tmp_path / "typo"
    assert main(["--store", str(missing), "-a", "coder", "recall", "x"]) == 1
    err = capsys.readouterr().err
    assert str(missing) in err and "init" in err
    assert not missing.exists()                    # and it stayed missing

    assert main(["--store", str(missing), "spaces"]) == 1
    assert not missing.exists()


def test_the_commands_that_create_a_store_still_create_one(capsys, tmp_path):
    fresh = tmp_path / "fresh"
    assert main(["--store", str(fresh), "-a", "coder", "remember", "hello"]) == 0
    capsys.readouterr()
    assert (fresh / "celluloid3.json").exists()
    assert main(["--store", str(tmp_path / "other"), "init"]) == 0


def test_a_directory_without_a_store_in_it_reads_as_missing(capsys, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--store", str(empty), "recall", "x"]) == 1
    assert "no store at" in capsys.readouterr().err


def test_a_key_value_pair_without_an_equals_sign_says_so(capsys, store):
    run(capsys, store, "remember", "a fact", "--meta", "kind=note")
    assert main(["--store", store, "recall", "x", "--where", "badpair"]) == 1
    assert "expects KEY=VALUE" in capsys.readouterr().err
    assert main(["--store", store, "remember", "x", "--meta", "badpair"]) == 1
    assert "expects KEY=VALUE" in capsys.readouterr().err


def test_an_unknown_checkpoint_is_an_error_not_a_traceback(capsys, store):
    run(capsys, store, "remember", "a fact")
    assert main(["--store", store, "recall", "x", "--at", "nope"]) == 1
    assert "error:" in capsys.readouterr().err


def test_an_unknown_fragment_id_is_an_error_not_a_traceback(capsys, store):
    run(capsys, store, "remember", "a fact")
    assert main(["--store", store, "forget", "deadbeef"]) == 1
    assert "error:" in capsys.readouterr().err


# -- reading does not write -------------------------------------------------

def test_read_commands_leave_no_lane_behind(capsys, store):
    """`celluloid3 log` used to invent a `default` agent that never remembered
    anything: an owner record, a name in everyone's agents() forever, and an
    epoch advanced per run -- for a read."""
    from celluloid3 import FileObjectStore

    run(capsys, store, "-a", "planner", "remember", "the customer wants SSO")
    run(capsys, store, "-a", "coder", "remember", "the auth service has no OIDC")

    for argv in (["log"], ["stats"], ["agents"], ["recall", "SSO"],
                 ["checkpoints"], ["owner"], ["spaces"]):
        assert main(["--store", store, *argv]) == 0
    capsys.readouterr()

    lanes = FileObjectStore(store).list("spaces/shared/lanes/")
    assert not [key for key in lanes if "/default/" in key]
    # ...and `agents` with no --agent lists the two writers, not itself
    assert run(capsys, store, "agents").split() == ["coder", "planner"]
    # `owner` for a lane nobody claimed is null, not a record it just made
    assert json.loads(run(capsys, store, "owner")) is None


def test_a_read_command_does_not_advance_an_epoch(capsys, store):
    run(capsys, store, "-a", "planner", "remember", "a fact")
    before = json.loads(run(capsys, store, "-a", "planner", "stats"))
    for _ in range(3):
        run(capsys, store, "log")
        run(capsys, store, "recall", "fact")
    after = json.loads(run(capsys, store, "-a", "planner", "stats"))
    assert before["epoch"] is after["epoch"] is None      # read-only handles
    # ...and the writer's own lane record is untouched by all that reading
    assert json.loads(run(capsys, store, "-a", "planner", "owner"))["epoch"] == 1


def test_stats_does_not_invent_an_identity_it_does_not_have(capsys, store):
    run(capsys, store, "-a", "planner", "remember", "a fact")
    stats = json.loads(run(capsys, store, "-a", "planner", "stats"))
    assert stats["read_only"] is True
    assert stats["agent"] is None and stats["epoch"] is None
    assert stats["fragments"] == 1        # ...but it still reads everything
    assert stats["known_agents"] == 1     # the one writer, not itself as well
    assert stats["mine"] == 0 and stats["from_others"] == 1


def test_the_writing_commands_still_claim_a_lane(capsys, store):
    run(capsys, store, "-a", "planner", "remember", "a fact")
    assert json.loads(run(capsys, store, "-a", "planner", "owner"))["epoch"] == 1
    run(capsys, store, "-a", "planner", "checkpoint", "now")
    assert "now" in run(capsys, store, "checkpoints")
    assert "deleted" in run(capsys, store, "-a", "planner", "gc")

