"""End-to-end multi-request run: completion, token counts, KV conservation."""

from __future__ import annotations

from pd_throughput_simulator_v5.metrics import build_report
from pd_throughput_simulator_v5.request import Stage
from pd_throughput_simulator_v5.scenario import (
    CapacityConfig,
    ClusterConfig,
    Scenario,
    WorkloadConfig,
    default_demo_scenario,
)
from pd_throughput_simulator_v5.simulator import PDSimulator


def test_demo_scenario_completes_and_conserves_kv():
    scenario = default_demo_scenario()
    sim = PDSimulator(scenario)
    states = sim.run()

    # every request finishes and emits exactly output_tokens tokens
    assert len(states) == scenario.workload.num_requests
    for s in states:
        assert Stage.REQUEST_END in s.stage_times_s
        assert s.generated_tokens == s.spec.output_tokens
        assert s.prefill_done_tokens == s.spec.prefill_work_tokens

    # KV pages, req slots and metadata buffers are fully returned at the end
    for p in sim.prefill_nodes:
        assert p.free_kv_pages == scenario.capacity.prefill_kv_pages
        assert p.free_metadata == scenario.capacity.prefill_metadata_buffers
        assert not p.inflight_queue and not p.waiting_queue and not p.bootstrap_queue
    for d in sim.decode_nodes:
        assert d.free_kv_pages == scenario.capacity.decode_kv_pages
        assert d.free_req_slots == scenario.capacity.decode_req_slots
        assert d.free_metadata == scenario.capacity.decode_metadata_buffers
        assert not d.running and not d.waiting_queue and not d.transfer_queue


def test_report_has_positive_latencies_and_throughput():
    scenario = default_demo_scenario()
    sim = PDSimulator(scenario)
    states = sim.run()
    report = build_report(states, scenario.metrics)
    assert report.num_completed == scenario.workload.num_requests
    assert report.ttft_s.p50 is not None and report.ttft_s.p50 > 0
    assert report.e2e_s.p50 is not None and report.e2e_s.p50 > 0
    assert report.request_throughput_rps > 0
    assert report.output_token_throughput_tps > 0


def test_multi_replica_round_robin_balances():
    scenario = Scenario(
        cluster=ClusterConfig(num_prefill_replicas=2, num_decode_replicas=2),
        capacity=CapacityConfig(prefill_kv_pages=10_000, decode_kv_pages=10_000),
        workload=WorkloadConfig(num_requests=20, arrival_rate_rps=500.0),
    )
    sim = PDSimulator(scenario)
    states = sim.run()
    assert all(Stage.REQUEST_END in s.stage_times_s for s in states)
    p_assignments = {s.prefill_replica_id for s in states}
    d_assignments = {s.decode_replica_id for s in states}
    assert p_assignments == {"p0", "p1"}
    assert d_assignments == {"d0", "d1"}


def test_tight_prefill_kv_backpressure_makes_progress():
    # Only one prefill request fits in KV at a time (16 pages / 16 per req).
    # Must serialize and complete without livelock (the 5M-event guard would
    # otherwise trip).
    scenario = Scenario(
        capacity=CapacityConfig(
            page_size=16,
            prefill_kv_pages=16,  # exactly one 256-token request
            decode_kv_pages=100_000,
            chunked_prefill_size=2048,
        ),
        workload=WorkloadConfig(
            num_requests=8, arrival_rate_rps=1000.0, input_tokens=256, output_tokens=3
        ),
    )
    sim = PDSimulator(scenario)
    states = sim.run()
    assert all(Stage.REQUEST_END in s.stage_times_s for s in states)
    for p in sim.prefill_nodes:
        assert p.free_kv_pages == scenario.capacity.prefill_kv_pages
