"""Chunked prefill splitting (FINDINGS §一 chunked prefill)."""

from __future__ import annotations

from pd_throughput_simulator_v5.cost_model import SimpleAnalyticCostModel
from pd_throughput_simulator_v5.kv_transfer import KVTransferSession
from pd_throughput_simulator_v5.network import NetworkFabric
from pd_throughput_simulator_v5.prefill_node import PrefillNode
from pd_throughput_simulator_v5.request import RequestSpec, RequestState, Stage
from pd_throughput_simulator_v5.scenario import CapacityConfig, Scenario, WorkloadConfig
from pd_throughput_simulator_v5.sim_clock import DiscreteEventLoop
from pd_throughput_simulator_v5.simulator import PDSimulator


def _bootstrapped_node(cap: CapacityConfig):
    loop = DiscreteEventLoop()
    cost = SimpleAnalyticCostModel()
    sessions: dict[str, KVTransferSession] = {}

    def get_session(rid: str) -> KVTransferSession:
        return sessions[rid]

    node = PrefillNode(
        replica_id="p0",
        loop=loop,
        cost=cost,
        capacity=cap,
        get_session=get_session,
        wake_decode=lambda req: None,
        fabric=NetworkFabric(loop, cost),
    )
    return node, sessions


def _drive_chunks(node, req_state):
    node.poll(0.0)
    chunks = []
    while node._has_forward_work():
        batch = node._select_forward_batch(0.0)
        assert batch.size == 1
        item = batch.items[0]
        chunks.append((item.tokens, item.finishes_prefill))
    return chunks


def test_long_prompt_is_split_page_aligned():
    cap = CapacityConfig(page_size=16, chunked_prefill_size=128, prefill_kv_pages=10_000)
    node, sessions = _bootstrapped_node(cap)
    req = RequestState(RequestSpec("r0", 0.0, input_tokens=500, output_tokens=4))
    sessions["r0"] = KVTransferSession("r0")
    sessions["r0"].d_receiver_init()
    sessions["r0"].d_send_metadata()  # prefill bootstrap done
    node.admit(req, 0.0)

    chunks = _drive_chunks(node, req)
    assert chunks == [(128, False), (128, False), (128, False), (116, True)]
    assert sum(c[0] for c in chunks) == 500
    assert [c[1] for c in chunks].count(True) == 1  # only the last chunk finishes
    assert req.prefill_done_tokens == 500


def test_chunk_consuming_the_remainder_is_marked_final():
    # The leftover budget (3 tokens) is smaller than a page, so the one-page
    # floor rounds *up* past it and swallows the second request's whole
    # 10-token remainder.  That chunk finishes the prefill and must be flagged
    # final: otherwise the request becomes a chunked_req with nothing left to
    # do, costing a spurious zero-token forward and a zero-byte KV flow.
    cap = CapacityConfig(page_size=16, chunked_prefill_size=2048, prefill_kv_pages=10_000)
    node, sessions = _bootstrapped_node(cap)
    for rid, inp, cached in (("big", 2045, 0), ("small", 11, 1)):
        sessions[rid] = KVTransferSession(rid)
        sessions[rid].d_receiver_init()
        sessions[rid].d_send_metadata()
        node.admit(
            RequestState(
                RequestSpec(rid, 0.0, input_tokens=inp, output_tokens=2, cached_tokens=cached)
            ),
            0.0,
        )

    batch = node._select_forward_batch(0.0)

    assert [(i.req.spec.request_id, i.tokens, i.finishes_prefill) for i in batch.items] == [
        ("big", 2045, True),
        ("small", 10, True),
    ]
    assert node.chunked_req is None  # nothing left half-done


def test_short_prompt_runs_whole():
    cap = CapacityConfig(page_size=16, chunked_prefill_size=2048, prefill_kv_pages=10_000)
    node, sessions = _bootstrapped_node(cap)
    req = RequestState(RequestSpec("r0", 0.0, input_tokens=100, output_tokens=4))
    sessions["r0"] = KVTransferSession("r0")
    sessions["r0"].d_receiver_init()
    sessions["r0"].d_send_metadata()
    node.admit(req, 0.0)

    chunks = _drive_chunks(node, req)
    assert chunks == [(100, True)]


def test_chunked_request_completes_end_to_end():
    scenario = Scenario(
        capacity=CapacityConfig(page_size=16, chunked_prefill_size=64, prefill_kv_pages=10_000, decode_kv_pages=10_000),
        workload=WorkloadConfig(num_requests=1),
    )
    sim = PDSimulator(scenario)
    req = RequestSpec("r0", 0.0, input_tokens=300, output_tokens=5)
    st = sim.run([req])[0]
    assert Stage.REQUEST_END in st.stage_times_s
    assert st.generated_tokens == 5
    assert st.prefill_done_tokens == 300


def test_kv_streams_per_chunk_and_overlaps_compute():
    # 1024-token prompt at chunk=256 → 4 chunks → 4 separate KV sends, the
    # first of which flies long before the last chunk's forward finishes.
    scenario = Scenario(
        capacity=CapacityConfig(page_size=16, chunked_prefill_size=256,
                                prefill_kv_pages=100_000, decode_kv_pages=100_000),
        workload=WorkloadConfig(num_requests=1),
    )
    sim = PDSimulator(scenario)
    st = sim.run([RequestSpec("r0", 0.0, input_tokens=1024, output_tokens=2)])[0]

    sends = [t for s, t in st.history if s == Stage.KV_CHUNK_SEND]
    assert len(sends) == 4  # one send per chunk, not one per request
    assert sends == sorted(sends)
    # streaming: the first chunk is already in flight while later chunks compute
    assert sends[0] < st.stage_times_s[Stage.PREFILL_FORWARD]

    session = sim.sessions["r0"]
    assert session.chunks_sent == 4
    assert session.chunks_done == 4
    # byte conservation: the whole prompt KV reaches decode exactly once
    assert session.transfer_bytes == scenario.cost_model.kv_transfer_bytes(1024)
    assert st.kv_send_idx == 1024


def test_cached_prefix_ships_with_the_first_chunk():
    # P recomputes only the uncached suffix, but the *full* prompt KV must
    # reach decode; the cached prefix rides along with the first chunk.
    scenario = Scenario(
        capacity=CapacityConfig(page_size=16, chunked_prefill_size=256,
                                prefill_kv_pages=100_000, decode_kv_pages=100_000),
        workload=WorkloadConfig(num_requests=1),
    )
    sim = PDSimulator(scenario)
    st = sim.run([RequestSpec("r0", 0.0, input_tokens=1024, cached_tokens=768,
                              output_tokens=2)])[0]
    assert st.spec.prefill_work_tokens == 256  # only the suffix is computed
    assert st.kv_send_idx == 1024  # but the whole prompt KV was pushed
    assert sim.sessions["r0"].transfer_bytes == scenario.cost_model.kv_transfer_bytes(1024)
