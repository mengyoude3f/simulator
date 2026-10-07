"""Admission gates and capacity fail-fast (FINDINGS §二 P#2 / D#2, D#4).

Pins the source-alignment rules that are easy to get wrong because they only
bite at the edges:

* a prefix hit is capped at ``input_len - 1`` so prefill always recomputes at
  least one token (``schedule_batch.py:1296-1301``);
* an over-capacity request is rejected at queue-add time rather than queued
  forever (``prefill.py:317-325`` / ``decode.py:634-654``);
* every *resource* gate in ``pop_preallocated`` is a ``break`` (head-of-line
  blocking), while a still-handshaking request is a ``continue``
  (``decode.py:950-957``, ``:1020-1024``).
"""

from __future__ import annotations

import pytest

from pd_throughput_simulator_v5.cost_model import SimpleAnalyticCostModel
from pd_throughput_simulator_v5.decode_node import DecodeNode
from pd_throughput_simulator_v5.kv_transfer import KVTransferSession
from pd_throughput_simulator_v5.request import RequestSpec, RequestState
from pd_throughput_simulator_v5.scenario import CapacityConfig, Scenario, WorkloadConfig
from pd_throughput_simulator_v5.sim_clock import DiscreteEventLoop
from pd_throughput_simulator_v5.simulator import PDSimulator


def _decode_node(cap: CapacityConfig):
    loop = DiscreteEventLoop()
    sessions: dict[str, KVTransferSession] = {}
    node = DecodeNode(
        replica_id="d0",
        loop=loop,
        cost=SimpleAnalyticCostModel(),
        capacity=cap,
        get_session=lambda rid: sessions[rid],
        wake_prefill=lambda req: None,
        on_request_end=lambda req, now: None,
    )
    return node, sessions


def _admit_all(node, sessions, specs):
    for spec in specs:
        sessions[spec.request_id] = KVTransferSession(spec.request_id)
        node.admit(RequestState(spec), 0.0)


# --- prefix hit is capped at input_len - 1 --------------------------------


def test_full_cache_hit_is_not_representable():
    # A 100%-cached prompt would mean a zero-token prefill forward with no
    # logits to sample the first output token from; SGLang prevents it by
    # capping the radix match at input_len - 1.
    with pytest.raises(ValueError, match="cached_tokens"):
        RequestSpec("r", 0.0, input_tokens=100, output_tokens=1, cached_tokens=100)
    spec = RequestSpec("r", 0.0, input_tokens=100, output_tokens=1, cached_tokens=99)
    assert spec.prefill_work_tokens == 1


# --- over-capacity requests fail fast instead of hanging -------------------


def test_oversized_prefill_request_fails_fast():
    scenario = Scenario(
        capacity=CapacityConfig(
            page_size=16,
            prefill_kv_pages=4,  # 64 tokens — a 1000-token prompt can never fit
            decode_kv_pages=1_000_000,
            chunked_prefill_size=2048,
        ),
        workload=WorkloadConfig(num_requests=1),
    )
    with pytest.raises(ValueError, match="prefill needs"):
        PDSimulator(scenario).run(
            [RequestSpec("r0", 0.0, input_tokens=1000, output_tokens=2)]
        )


def test_oversized_decode_request_fails_fast():
    scenario = Scenario(
        capacity=CapacityConfig(
            page_size=16,
            prefill_kv_pages=1_000_000,
            decode_kv_pages=4,
            chunked_prefill_size=2048,
        ),
        workload=WorkloadConfig(num_requests=1),
    )
    with pytest.raises(ValueError, match="decode prealloc needs"):
        PDSimulator(scenario).run(
            [RequestSpec("r0", 0.0, input_tokens=1000, output_tokens=2)]
        )


# --- pop_preallocated: break on resources, continue on handshake -----------


def test_decode_prealloc_is_head_of_line_blocking():
    # Budget is exactly r0 (158 pages) + r2 (39 pages).  r0 is admitted, r1
    # (158) no longer fits — and r2, which *would* fit in the 39 left over,
    # must not overtake it.
    cap = CapacityConfig(
        page_size=16,
        decode_kv_pages=197,
        decode_reserved_tokens=512,
        prefill_kv_pages=1_000_000,
    )
    node, sessions = _decode_node(cap)
    _admit_all(
        node,
        sessions,
        [
            RequestSpec("r0", 0.0, input_tokens=2000, output_tokens=2),
            RequestSpec("r1", 0.0, input_tokens=2000, output_tokens=2),
            RequestSpec("r2", 0.0, input_tokens=100, output_tokens=2),
        ],
    )
    for session in sessions.values():
        session.d_receiver_init()  # all three have handshaked

    node.poll(0.0)

    assert [r.spec.request_id for r in node.transfer_queue] == ["r0"]
    assert [r.spec.request_id for r in node.prealloc_queue] == ["r1", "r2"]
    assert node.free_kv_pages == 39  # room for r2, deliberately left unused


def test_unhandshaked_request_does_not_block_the_queue():
    # The handshake gate is the one gate SGLang skips past rather than breaks
    # on, so a later request whose receiver.init already landed goes first.
    cap = CapacityConfig(
        page_size=16, decode_kv_pages=1_000_000, prefill_kv_pages=1_000_000
    )
    node, sessions = _decode_node(cap)
    _admit_all(
        node,
        sessions,
        [
            RequestSpec("r0", 0.0, input_tokens=100, output_tokens=2),
            RequestSpec("r1", 0.0, input_tokens=100, output_tokens=2),
            RequestSpec("r2", 0.0, input_tokens=100, output_tokens=2),
        ],
    )
    sessions["r1"].d_receiver_init()  # only r1 has handshaked

    node.poll(0.0)

    assert [r.spec.request_id for r in node.transfer_queue] == ["r1"]
    assert [r.spec.request_id for r in node.prealloc_queue] == ["r0", "r2"]
