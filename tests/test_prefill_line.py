"""Single-request prefill-line lifecycle order (FINDINGS §二 PREFILL table)."""

from __future__ import annotations

from pd_throughput_simulator_v5.request import PLinePhase, RequestSpec, Stage
from pd_throughput_simulator_v5.scenario import Scenario, WorkloadConfig
from pd_throughput_simulator_v5.simulator import PDSimulator


def _run_single() -> "RequestState":
    scenario = Scenario(workload=WorkloadConfig(num_requests=1))
    sim = PDSimulator(scenario)
    req = RequestSpec("r0", 0.0, input_tokens=256, output_tokens=8)
    states = sim.run([req])
    return states[0]


def test_prefill_stage_order():
    st = _run_single()
    t = st.stage_times_s
    order = [
        Stage.PREFILL_PREPARE,
        Stage.PREFILL_BOOTSTRAP,
        Stage.PREFILL_WAITING,
        Stage.PREFILL_FORWARD,
        Stage.KV_TRANSFER_START,
        Stage.PREFILL_TRANSFER_KV_CACHE,
        Stage.PREFILL_COMPLETE,
    ]
    for stage in order:
        assert stage in t, f"missing {stage}"
    times = [t[s] for s in order]
    assert times == sorted(times), f"prefill stages out of order: {times}"
    assert st.p_phase == PLinePhase.DONE


def test_kv_transfer_end_after_start():
    st = _run_single()
    t = st.stage_times_s
    assert t[Stage.PREFILL_TRANSFER_KV_CACHE] >= t[Stage.KV_TRANSFER_START]
