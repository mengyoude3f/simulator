"""Cross-line handoff causality (FINDINGS §三)."""

from __future__ import annotations

from pd_throughput_simulator_v5.request import RequestSpec, Stage
from pd_throughput_simulator_v5.scenario import Scenario, WorkloadConfig
from pd_throughput_simulator_v5.simulator import PDSimulator


def _run_single():
    sim = PDSimulator(Scenario(workload=WorkloadConfig(num_requests=1)))
    req = RequestSpec("r0", 0.0, input_tokens=256, output_tokens=8)
    return sim.run([req])[0]


def test_decode_send_metadata_triggers_prefill_bootstrap_done():
    st = _run_single()
    t = st.stage_times_s
    # DECODE_BOOTSTRAP == decode did send_metadata; it must not happen after
    # prefill leaves the bootstrap queue.
    assert t[Stage.DECODE_BOOTSTRAP] <= t[Stage.PREFILL_BOOTSTRAP]


def test_transfer_success_gates_decode_transferred():
    st = _run_single()
    t = st.stage_times_s
    # decode can only enter its waiting queue once prefill's KV transfer landed.
    assert t[Stage.DECODE_TRANSFERRED] >= t[Stage.PREFILL_TRANSFER_KV_CACHE]


def test_first_token_is_sampled_by_prefill_before_decode_observes_it():
    st = _run_single()
    t = st.stage_times_s
    # prefill samples the first token at PREFILL_FORWARD; decode replays it later.
    assert t[Stage.PREFILL_FORWARD] <= t[Stage.FIRST_TOKEN]
    assert t[Stage.DECODE_TRANSFERRED] <= t[Stage.FIRST_TOKEN]


def test_full_end_to_end_order():
    st = _run_single()
    t = st.stage_times_s
    assert (
        t[Stage.REQUEST_ARRIVAL]
        <= t[Stage.DECODE_PREPARE]
        <= t[Stage.DECODE_BOOTSTRAP]
        <= t[Stage.PREFILL_WAITING]
        <= t[Stage.PREFILL_FORWARD]
        <= t[Stage.KV_TRANSFER_START]
        <= t[Stage.PREFILL_TRANSFER_KV_CACHE]
        <= t[Stage.DECODE_TRANSFERRED]
        <= t[Stage.FIRST_TOKEN]
        <= t[Stage.REQUEST_END]
    )
