"""Single-request decode-line lifecycle order (FINDINGS §二 DECODE table)."""

from __future__ import annotations

from pd_throughput_simulator_v5.request import DLinePhase, RequestSpec, Stage
from pd_throughput_simulator_v5.scenario import Scenario, WorkloadConfig
from pd_throughput_simulator_v5.simulator import PDSimulator


def _run_single(output_tokens: int = 8):
    sim = PDSimulator(Scenario(workload=WorkloadConfig(num_requests=1)))
    req = RequestSpec("r0", 0.0, input_tokens=256, output_tokens=output_tokens)
    return sim.run([req])[0]


def test_decode_stage_order():
    st = _run_single()
    t = st.stage_times_s
    order = [
        Stage.DECODE_PREPARE,
        Stage.DECODE_BOOTSTRAP,
        Stage.DECODE_TRANSFERRED,
        Stage.DECODE_WAITING,
        Stage.DECODE_FAKE_OUTPUT,
        Stage.FIRST_TOKEN,
        Stage.REQUEST_END,
    ]
    for stage in order:
        assert stage in t, f"missing {stage}"
    times = [t[s] for s in order]
    assert times == sorted(times), f"decode stages out of order: {times}"
    assert st.d_phase == DLinePhase.DONE


def test_first_token_is_the_prebuilt_token():
    st = _run_single(output_tokens=8)
    t = st.stage_times_s
    # prebuilt fake output and the first token happen together (prefill-sampled).
    assert t[Stage.DECODE_FAKE_OUTPUT] == t[Stage.FIRST_TOKEN]
    assert st.generated_tokens == 8


def test_single_output_token_quick_finishes_without_decode_loop():
    st = _run_single(output_tokens=1)
    assert st.generated_tokens == 1
    # the lone token is the prefill-sampled one; no decode iteration needed.
    assert st.stage_times_s[Stage.FIRST_TOKEN] == st.stage_times_s[Stage.REQUEST_END]
