"""Overlap scheduler loop: back-to-back forwards, no CPU gap (FINDINGS §一)."""

from __future__ import annotations

import pytest

from pd_throughput_simulator_v5.cost_model import SimpleAnalyticCostModel
from pd_throughput_simulator_v5.request import RequestSpec, Stage
from pd_throughput_simulator_v5.scenario import CapacityConfig, Scenario, WorkloadConfig
from pd_throughput_simulator_v5.simulator import PDSimulator

_COST = SimpleAnalyticCostModel()


def test_decode_forwards_are_contiguous():
    # A single request's decode iterations run back-to-back: the inter-token
    # gap equals exactly one decode forward (overlap hides CPU bookkeeping).
    sim = PDSimulator(Scenario(workload=WorkloadConfig(num_requests=1)))
    req = RequestSpec("r0", 0.0, input_tokens=128, output_tokens=8)
    st = sim.run([req])[0]

    expected = _COST.decode_iter_s(1)
    gaps = [
        st.token_times_s[i] - st.token_times_s[i - 1]
        for i in range(1, len(st.token_times_s))
    ]
    assert len(gaps) == 7
    for gap in gaps:
        assert gap == pytest.approx(expected, rel=1e-9)


def test_prefill_forwards_are_contiguous():
    # One request per forward (chunk budget == prompt); three requests arriving
    # together must forward back-to-back with no idle gap between them.
    cap = CapacityConfig(chunked_prefill_size=256, prefill_kv_pages=10_000, decode_kv_pages=10_000)
    sim = PDSimulator(Scenario(capacity=cap, workload=WorkloadConfig(num_requests=3)))
    reqs = [RequestSpec(f"r{i}", 0.0, input_tokens=256, output_tokens=2) for i in range(3)]
    states = sim.run(reqs)

    forward_times = sorted(s.stage_times_s[Stage.PREFILL_FORWARD] for s in states)
    forward_dur = _COST.prefill_forward_s(256, 1)
    diffs = [forward_times[i] - forward_times[i - 1] for i in range(1, 3)]
    for diff in diffs:
        assert diff == pytest.approx(forward_dur, rel=1e-9)


def test_all_requests_finish_under_overlap():
    sim = PDSimulator(Scenario(workload=WorkloadConfig(num_requests=16)))
    states = sim.run()
    assert all(Stage.REQUEST_END in s.stage_times_s for s in states)
