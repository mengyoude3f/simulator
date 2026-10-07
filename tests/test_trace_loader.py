"""Loading request specs from an external jsonl trace."""

from __future__ import annotations

import os

import pytest

from pd_throughput_simulator_v5.request import Stage
from pd_throughput_simulator_v5.scenario import (
    CapacityConfig,
    Scenario,
    WorkloadConfig,
)
from pd_throughput_simulator_v5.simulator import PDSimulator
from pd_throughput_simulator_v5.trace_loader import load_trace_requests

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_MINI = os.path.join(_FIXTURES, "mini_trace.jsonl")
_MINI_FLAT = os.path.join(_FIXTURES, "mini_trace_flat.jsonl")
_REAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # tests → pkg → infra
    "trace",
    "traces_2026-07-09-16-52-48.jsonl",
)


def test_nested_metadata_trace():
    reqs = load_trace_requests(_MINI)
    assert len(reqs) == 3

    # arrivals rebased so the earliest is 0
    assert [r.arrival_time_s for r in reqs] == pytest.approx([0.0, 2.5, 5.0])

    a, b, c = reqs
    assert (a.request_id, a.input_tokens, a.output_tokens, a.cached_tokens) == ("r-a", 1000, 10, 200)
    assert a.prefill_work_tokens == 800
    # line 2 has no cached_tokens → defaults to 0
    assert (b.request_id, b.cached_tokens) == ("r-b", 0)
    # line 3 has no request_id → fallback prefix; cached (900) clamped to
    # input - 1 (799), because SGLang caps a prefix hit at input_len - 1 so
    # prefill always recomputes at least one token (schedule_batch.py:1296-1301)
    assert c.request_id.startswith("req-")
    assert c.cached_tokens == 799
    assert c.prefill_work_tokens == 1  # never a zero-token prefill


def test_flat_fields_and_numeric_arrival():
    reqs = load_trace_requests(_MINI_FLAT)
    assert len(reqs) == 2
    assert [r.arrival_time_s for r in reqs] == pytest.approx([0.0, 2.0])  # rebased from 10,12
    assert (reqs[0].input_tokens, reqs[0].output_tokens, reqs[0].cached_tokens) == (300, 8, 50)
    assert reqs[1].cached_tokens == 0


def test_limit():
    assert len(load_trace_requests(_MINI, limit=2)) == 2


def test_trace_drives_simulator_via_config():
    scenario = Scenario(
        capacity=CapacityConfig(prefill_kv_pages=10_000, decode_kv_pages=10_000),
        workload=WorkloadConfig(trace_path=_MINI),
    )
    states = PDSimulator(scenario).run()
    assert len(states) == 3
    assert all(Stage.REQUEST_END in s.stage_times_s for s in states)
    for s in states:
        assert s.generated_tokens == s.spec.output_tokens
        assert s.prefill_done_tokens == s.spec.prefill_work_tokens


@pytest.mark.skipif(not os.path.exists(_REAL), reason="real trace file not present")
def test_real_trace_loads_and_runs():
    reqs = load_trace_requests(_REAL, limit=20)
    assert len(reqs) == 20
    assert reqs[0].arrival_time_s == 0.0
    for r in reqs:
        assert r.input_tokens > 0
        assert r.output_tokens >= 1
        assert 0 <= r.cached_tokens <= r.input_tokens
    scenario = Scenario(
        capacity=CapacityConfig(
            prefill_kv_pages=100_000, decode_kv_pages=300_000, chunked_prefill_size=8192
        ),
    )
    states = PDSimulator(scenario).run(reqs)
    assert all(Stage.REQUEST_END in s.stage_times_s for s in states)
