"""Multi-resource bandwidth contention: P egress, D ingress, cluster fabric."""

from __future__ import annotations

import pytest

from pd_throughput_simulator_v5.cost_model import SimpleAnalyticCostModel
from pd_throughput_simulator_v5.network import NetworkFabric
from pd_throughput_simulator_v5.request import RequestSpec, Stage
from pd_throughput_simulator_v5.scenario import (
    CapacityConfig,
    ClusterConfig,
    Scenario,
    WorkloadConfig,
)
from pd_throughput_simulator_v5.sim_clock import DiscreteEventLoop
from pd_throughput_simulator_v5.simulator import PDSimulator

# Wide fabric so it is not the bottleneck unless a test says so.
_NIC = 1.0e9
_COST = SimpleAnalyticCostModel(
    egress_bandwidth_bytes_s=_NIC,
    ingress_bandwidth_bytes_s=_NIC,
    fabric_bandwidth_bytes_s=1.0e12,
)


def _fabric(cost=_COST):
    loop = DiscreteEventLoop()
    return loop, NetworkFabric(loop, cost)


def _run(loop, fabric, flows):
    """flows: list of (rid, src, dst, bytes). Returns {rid: completion_time}."""
    done: dict[str, float] = {}
    for rid, src, dst, nbytes in flows:
        fabric.submit(
            rid, src=src, dst=dst, total_bytes=nbytes,
            on_done=(lambda r=rid: done.__setitem__(r, loop.clock.now_s)),
        )
    loop.run()
    return done


def test_single_flow_gets_full_nic():
    loop, fab = _fabric()
    done = _run(loop, fab, [("a", "p0", "d0", int(_NIC))])
    assert done["a"] == pytest.approx(1.0)  # 1 GB / 1 GB/s


def test_two_flows_on_same_egress_share_it():
    # Same prefill → its egress NIC is the bottleneck: each gets NIC/2.
    loop, fab = _fabric()
    done = _run(loop, fab, [
        ("a", "p0", "d0", int(_NIC)),
        ("b", "p0", "d1", int(_NIC)),
    ])
    assert done["a"] == pytest.approx(2.0)
    assert done["b"] == pytest.approx(2.0)


def test_two_flows_on_same_ingress_share_it():
    # Different prefills → same decode: the D ingress NIC is the bottleneck
    # (incast).  This is the newly added decode-side contention.
    loop, fab = _fabric()
    done = _run(loop, fab, [
        ("a", "p0", "d0", int(_NIC)),
        ("b", "p1", "d0", int(_NIC)),
    ])
    assert done["a"] == pytest.approx(2.0)
    assert done["b"] == pytest.approx(2.0)


def test_disjoint_nics_do_not_contend():
    # p0→d0 and p1→d1 share no NIC and the fabric is wide → both run at full NIC.
    loop, fab = _fabric()
    done = _run(loop, fab, [
        ("a", "p0", "d0", int(_NIC)),
        ("b", "p1", "d1", int(_NIC)),
    ])
    assert done["a"] == pytest.approx(1.0)
    assert done["b"] == pytest.approx(1.0)


def test_fabric_cap_throttles_disjoint_flows():
    # Same disjoint pairs, but the cluster switch only has 1 NIC worth of
    # capacity → the two flows must split it (max-min fair), taking 2s each.
    cost = SimpleAnalyticCostModel(
        egress_bandwidth_bytes_s=_NIC,
        ingress_bandwidth_bytes_s=_NIC,
        fabric_bandwidth_bytes_s=_NIC,
    )
    loop, fab = _fabric(cost)
    done = _run(loop, fab, [
        ("a", "p0", "d0", int(_NIC)),
        ("b", "p1", "d1", int(_NIC)),
    ])
    assert done["a"] == pytest.approx(2.0)
    assert done["b"] == pytest.approx(2.0)


def test_max_min_fair_reallocates_unused_capacity():
    # a and b share p0's egress (NIC/2 each).  c is alone on p1→d1 and is only
    # limited by the fabric, which has 2*NIC: after a,b take NIC/2 each the
    # fabric still has NIC to spare, so c must get a full NIC, not a 1/3 share.
    cost = SimpleAnalyticCostModel(
        egress_bandwidth_bytes_s=_NIC,
        ingress_bandwidth_bytes_s=_NIC,
        fabric_bandwidth_bytes_s=2 * _NIC,
    )
    loop, fab = _fabric(cost)
    fab.submit("a", src="p0", dst="d0", total_bytes=int(_NIC), on_done=lambda: None)
    fab.submit("b", src="p0", dst="d1", total_bytes=int(_NIC), on_done=lambda: None)
    fab.submit("c", src="p1", dst="d2", total_bytes=int(_NIC), on_done=lambda: None)
    assert fab.rate_of("a") == pytest.approx(_NIC / 2)
    assert fab.rate_of("b") == pytest.approx(_NIC / 2)
    assert fab.rate_of("c") == pytest.approx(_NIC)  # not throttled to NIC/3


def test_staggered_flows_recompute_share():
    loop, fab = _fabric()
    done: dict[str, float] = {}
    fab.submit("a", src="p0", dst="d0", total_bytes=int(_NIC),
               on_done=lambda: done.__setitem__("a", loop.clock.now_s))
    loop.schedule_after(
        0.5, "submit_b",
        lambda e: fab.submit("b", src="p0", dst="d1", total_bytes=int(_NIC),
                             on_done=lambda: done.__setitem__("b", loop.clock.now_s)),
    )
    loop.run()
    # a alone on [0,0.5] (half done), then shares → finishes at 1.5;
    # b gets half over [0.5,1.5], then full → finishes at 2.0.
    assert done["a"] == pytest.approx(1.5)
    assert done["b"] == pytest.approx(2.0)


def test_congestion_alpha_derates_capacity():
    # alpha=1 → two concurrent flows on a NIC see capacity/2 total, so each gets
    # capacity/4 and a 1-NIC-byte transfer takes 4s instead of 2s.
    cost = SimpleAnalyticCostModel(
        egress_bandwidth_bytes_s=_NIC,
        ingress_bandwidth_bytes_s=1.0e12,
        fabric_bandwidth_bytes_s=1.0e12,
        congestion_alpha=1.0,
    )
    loop, fab = _fabric(cost)
    done = _run(loop, fab, [
        ("a", "p0", "d0", int(_NIC)),
        ("b", "p0", "d1", int(_NIC)),
    ])
    assert done["a"] == pytest.approx(4.0)
    assert done["b"] == pytest.approx(4.0)
    # A single flow is unaffected by alpha (n <= 1 → no derate).
    loop2, fab2 = _fabric(cost)
    done2 = _run(loop2, fab2, [("a", "p0", "d0", int(_NIC))])
    assert done2["a"] == pytest.approx(1.0)


def test_zero_capacity_fails_fast():
    # A misconfigured (zero-capacity) resource must raise, not silently hang
    # with unfinished requests.
    cost = SimpleAnalyticCostModel(egress_bandwidth_bytes_s=0.0)
    loop, fab = _fabric(cost)
    with pytest.raises(ValueError, match="zero rate"):
        fab.submit("a", src="p0", dst="d0", total_bytes=1000, on_done=lambda: None)


def test_end_to_end_with_contention_completes():
    scenario = Scenario(
        cluster=ClusterConfig(num_prefill_replicas=2, num_decode_replicas=1),
        capacity=CapacityConfig(prefill_kv_pages=100_000, decode_kv_pages=100_000,
                                chunked_prefill_size=100_000),
        workload=WorkloadConfig(num_requests=8, arrival_rate_rps=1e6,
                                input_tokens=256, output_tokens=2),
        cost_model=SimpleAnalyticCostModel(congestion_alpha=0.5),
    )
    sim = PDSimulator(scenario)
    states = sim.run()
    assert all(Stage.REQUEST_END in s.stage_times_s for s in states)
    assert sim.fabric.num_active == 0  # fabric fully drains
