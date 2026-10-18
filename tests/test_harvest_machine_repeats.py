"""Issue #319: single-turn transcripts a local hook re-runs verbatim.

A hook that summarises the previous session on a timer writes one single-turn
transcript per run. Neither the engine's replay markers nor the sub-agent
markers match that template, and `entrypoint` is "claude-desktop" in the
desktop app, like interactive sessions — so before this filter they were
harvested as user tasks and, because harvest walks newest first and stops at
`--max-sessions`, they filled the cap and pushed real sessions out of the
night entirely.

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.harvest import harvest
from skillopt_sleep.mine import heuristic_mine
from skillopt_sleep.types import SessionDigest

# The shape from the report: one identical header, then the run's own payload.
SUMMARY_HEADER = (
    "Below is a conversation log from a coding session. Create a summary to "
    "help the next session quickly understand the context.\n## Prioritize\n"
    "- decisions\n- open items\n\nLOG:\n"
)


class _HarvestFixture:
    """Writes Claude Code shaped transcripts under a temporary projects dir."""

    def __init__(self, tmp: str, project: str | None = None):
        self.transcripts = os.path.join(tmp, "projects")
        self.project = project or os.path.join(tmp, "proj")
        os.makedirs(self.project, exist_ok=True)

    def session(self, session_id: str, turns, mtime: float) -> str:
        folder = os.path.join(self.transcripts, "proj")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, session_id + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, (role, text) in enumerate(turns):
                f.write(json.dumps({
                    "type": role,
                    "cwd": self.project,
                    "sessionId": session_id,
                    "timestamp": f"2026-01-01T10:{i:02d}:00Z",
                    "message": {"role": role, "content": text},
                }) + "\n")
        os.utime(path, (mtime, mtime))
        return path

    def summary_sessions(self, n: int, mtime: float = 2_000_000) -> None:
        """Single-turn sessions whose prompts differ only after the header."""
        for i in range(n):
            self.session(
                f"summary-{i}",
                [("user", SUMMARY_HEADER + f"turn {i}: user asked for item {i}\n"),
                 ("assistant", f"Summary {i}: ...")],
                mtime + i,
            )

    def real_sessions(self) -> None:
        self.session("real-1", [
            ("user", "Fix the failing date parser test in module B."),
            ("assistant", "Fixed."),
            ("user", "Thanks, now add a regression test."),
            ("assistant", "Added."),
        ], 1_000_000)
        self.session("real-2", [
            ("user", "Add a --dry-run flag to the export command."),
            ("assistant", "Added."),
        ], 1_000_100)


class TestMachineGeneratedRepeats(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("SKILLOPT_SLEEP_REPEAT_MIN_SESSIONS", None)

    def tearDown(self):
        self._env.stop()

    def test_summary_sessions_do_not_fill_the_session_cap(self):
        # The report's repro: three hourly summaries are newer than two real
        # sessions, so with limit=3 the summaries used to take every slot.
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.real_sessions()
            fx.summary_sessions(3)

            capped = harvest(fx.transcripts, scope="invoked",
                             invoked_project=fx.project, limit=3)
            uncapped = harvest(fx.transcripts, scope="invoked",
                               invoked_project=fx.project)

        self.assertEqual(sorted(d.session_id for d in capped),
                         ["real-1", "real-2"])
        self.assertEqual(sorted(d.session_id for d in uncapped),
                         ["real-1", "real-2"])

    def test_the_mined_intents_are_the_real_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.real_sessions()
            fx.summary_sessions(3)
            capped = harvest(fx.transcripts, scope="invoked",
                             invoked_project=fx.project, limit=3)
            mined = heuristic_mine(capped)

        self.assertTrue(mined)
        self.assertEqual(sorted({s for t in mined for s in t.source_sessions}),
                         ["real-1", "real-2"])
        for task in mined:
            self.assertFalse(task.intent.startswith(SUMMARY_HEADER[:40]))

    def test_short_repeated_one_shot_prompts_are_kept(self):
        # The false-positive guard: a real one-shot prompt is usually short
        # and repeats *entirely*, so it never reaches the prefix rule.
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            for i in range(4):
                fx.session(f"ask-{i}",
                           [("user", "Run the tests and fix what is broken."),
                            ("assistant", "Done.")],
                           1_000_000 + i)
            kept = harvest(fx.transcripts, scope="invoked",
                           invoked_project=fx.project)

        self.assertEqual(len(kept), 4)

    def test_multi_turn_session_sharing_the_prefix_is_kept(self):
        # Only single-turn sessions can be a re-run template; a follow-up
        # user turn makes the session interactive.
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.summary_sessions(3)
            fx.session("interactive", [
                ("user", SUMMARY_HEADER + "written by hand this time\n"),
                ("assistant", "ok"),
                ("user", "no, summarise the other file too"),
                ("assistant", "done"),
            ], 2_000_100)
            kept = harvest(fx.transcripts, scope="invoked",
                           invoked_project=fx.project)

        self.assertEqual([d.session_id for d in kept], ["interactive"])

    def test_threshold_follows_the_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.summary_sessions(2)

            with mock.patch.dict(os.environ,
                                 {"SKILLOPT_SLEEP_REPEAT_MIN_SESSIONS": "2"}):
                dropped = harvest(fx.transcripts, scope="invoked",
                                  invoked_project=fx.project)
            with mock.patch.dict(os.environ,
                                 {"SKILLOPT_SLEEP_REPEAT_MIN_SESSIONS": "4"}):
                kept = harvest(fx.transcripts, scope="invoked",
                               invoked_project=fx.project)
            with mock.patch.dict(os.environ,
                                 {"SKILLOPT_SLEEP_REPEAT_MIN_SESSIONS": "0"}):
                disabled = harvest(fx.transcripts, scope="invoked",
                                   invoked_project=fx.project)

        self.assertEqual(dropped, [])
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(disabled), 2)

    def test_non_integer_threshold_falls_back_to_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.summary_sessions(3)
            with mock.patch.dict(os.environ,
                                 {"SKILLOPT_SLEEP_REPEAT_MIN_SESSIONS": "many"}):
                kept = harvest(fx.transcripts, scope="invoked",
                               invoked_project=fx.project)
        self.assertEqual(kept, [])

    def test_prefix_is_normalized_and_length_gated(self):
        from skillopt_sleep.harvest import (
            _REPEAT_PREFIX_CHARS,
            _machine_generated_repeats,
            _repeat_prefix,
        )

        def digest(session_id, prompt, n_user=1):
            return SessionDigest(session_id=session_id, project="/p",
                                 user_prompts=[prompt], n_user_turns=n_user)

        long_a = digest("a", SUMMARY_HEADER.upper() + "payload one")
        long_b = digest("b", SUMMARY_HEADER.lower() + "payload two")
        short = digest("c", "same one-liner", n_user=1)
        two_turn = digest("d", SUMMARY_HEADER + "payload", n_user=2)

        self.assertEqual(_repeat_prefix(digest("short", "x" * (
            _REPEAT_PREFIX_CHARS - 1))), "")
        self.assertEqual(_repeat_prefix(short), "")
        self.assertEqual(_repeat_prefix(two_turn), "")
        # Same text, different case and wrapping -> one prefix.
        self.assertEqual(_repeat_prefix(long_a), _repeat_prefix(long_b))
        self.assertEqual(
            _machine_generated_repeats([long_a, long_b], 2), {"a", "b"})
        self.assertEqual(
            _machine_generated_repeats([long_a, long_b, short, two_turn], 2),
            {"a", "b"})

    def test_stats_report_what_was_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.real_sessions()
            fx.summary_sessions(3)
            stats: dict = {}
            capped = harvest(fx.transcripts, scope="invoked",
                             invoked_project=fx.project, limit=3, stats=stats)

        self.assertEqual(stats, {
            "candidates": 5,
            "machine_generated_repeats": 3,
            "skipped_by_cap": 0,
            "sessions": 2,
        })
        self.assertEqual(len(capped), stats["sessions"])

    def test_stats_count_sessions_past_the_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.real_sessions()
            fx.summary_sessions(3)
            stats: dict = {}
            harvest(fx.transcripts, scope="invoked",
                    invoked_project=fx.project, limit=1, stats=stats)

        self.assertEqual(stats["machine_generated_repeats"], 3)
        self.assertEqual(stats["skipped_by_cap"], 1)
        self.assertEqual(stats["sessions"], 1)

    def test_harvest_note_is_empty_when_nothing_was_filtered(self):
        from skillopt_sleep.harvest import format_harvest_note

        self.assertEqual(format_harvest_note({}), "")
        self.assertEqual(format_harvest_note(None), "")
        self.assertEqual(
            format_harvest_note({"machine_generated_repeats": 0,
                                 "skipped_by_cap": 0}), "")
        self.assertEqual(
            format_harvest_note({"machine_generated_repeats": 3,
                                 "skipped_by_cap": 0}),
            " (skipped: 3 machine-generated)")
        self.assertEqual(
            format_harvest_note({"machine_generated_repeats": 3,
                                 "skipped_by_cap": 2}),
            " (skipped: 3 machine-generated, 2 past the session cap)")


class TestHarvestForConfigStats(unittest.TestCase):
    def test_claude_source_reports_stats_through_the_router(self):
        import argparse

        from skillopt_sleep.__main__ import _add_common, _cfg_from_args
        from skillopt_sleep.harvest_sources import harvest_for_config

        with tempfile.TemporaryDirectory() as tmp:
            fx = _HarvestFixture(tmp)
            fx.real_sessions()
            fx.summary_sessions(3)
            parser = argparse.ArgumentParser()
            _add_common(parser)
            args = parser.parse_args([
                "--project", fx.project,
                "--source", "claude",
                "--claude-home", tmp,
            ])
            cfg = _cfg_from_args(args)
            stats: dict = {}
            digests = harvest_for_config(cfg, limit=3, stats=stats)

        self.assertEqual(sorted(d.session_id for d in digests),
                         ["real-1", "real-2"])
        self.assertEqual(stats["machine_generated_repeats"], 3)


if __name__ == "__main__":
    unittest.main()
