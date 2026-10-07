"""Metrics computation over hand-built request states."""

from __future__ import annotations

import pytest

from pd_throughput_simulator_v5.metrics import build_report
from pd_throughput_simulator_v5.request import RequestSpec, RequestState, Stage
from pd_throughput_simulator_v5.scenario import MetricsConfig


def _state(rid: str, arrival: float, token_times: list[float]) -> RequestState:
    st = RequestState(RequestSpec(rid, arrival, input_tokens=8, output_tokens=len(token_times)))
    st.mark(Stage.REQUEST_ARRIVAL, arrival)
    for i, t in enumerate(token_times):
        st.generated_tokens += 1
        st.token_times_s.append(t)
        st.mark(Stage.TOKEN_EMITTED, t)
        if i == 0:
            st.mark(Stage.FIRST_TOKEN, t)
    st.mark(Stage.REQUEST_END, token_times[-1])
    return st


def test_ttft_e2e_and_throughput():
    # two requests, first token 1s after arrival, then two more tokens 0.5s apart
    s0 = _state("r0", 0.0, [1.0, 1.5, 2.0])
    s1 = _state("r1", 0.0, [1.0, 1.5, 2.0])
    report = build_report([s0, s1], MetricsConfig(window_start_s=0.0, window_end_s=None))

    assert report.num_completed == 2
    assert report.ttft_s.p50 == pytest.approx(1.0)
    assert report.e2e_s.p50 == pytest.approx(2.0)
    assert report.tbt_s.p50 == pytest.approx(0.5)
    # window is [0, 2.0]; degenerate? end == max end time 2.0, so window (0,2.0)
    # both complete at 2.0 which is the exclusive edge -> handled by degenerate rule
    assert report.request_throughput_rps > 0


def test_window_excludes_out_of_range_completions():
    s0 = _state("r0", 0.0, [1.0])
    s1 = _state("r1", 0.0, [5.0])
    report = build_report([s0, s1], MetricsConfig(window_start_s=0.0, window_end_s=2.0))
    # only r0 completes within [0, 2.0)
    assert report.ttft_s.count == 1
    assert report.request_throughput_rps == pytest.approx(1 / 2.0)
