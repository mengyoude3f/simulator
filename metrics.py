"""Latency and throughput metrics over completed request states."""

from __future__ import annotations

from dataclasses import dataclass

from .request import RequestState, Stage
from .scenario import MetricsConfig


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class LatencyStats:
    count: int
    mean: float | None
    p50: float | None
    p90: float | None
    p99: float | None

    @classmethod
    def of(cls, values: list[float]) -> "LatencyStats":
        return cls(
            count=len(values),
            mean=(sum(values) / len(values)) if values else None,
            p50=_percentile(values, 0.50),
            p90=_percentile(values, 0.90),
            p99=_percentile(values, 0.99),
        )


@dataclass
class SimulationReport:
    num_requests: int
    num_completed: int
    window_start_s: float
    window_end_s: float
    ttft_s: LatencyStats
    tbt_s: LatencyStats
    e2e_s: LatencyStats
    request_throughput_rps: float
    output_token_throughput_tps: float


def _ttft(req: RequestState) -> float | None:
    first = req.stage_times_s.get(Stage.FIRST_TOKEN)
    arrival = req.stage_times_s.get(Stage.REQUEST_ARRIVAL)
    if first is None or arrival is None:
        return None
    return first - arrival


def _e2e(req: RequestState) -> float | None:
    end = req.stage_times_s.get(Stage.REQUEST_END)
    arrival = req.stage_times_s.get(Stage.REQUEST_ARRIVAL)
    if end is None or arrival is None:
        return None
    return end - arrival


def _tbts(req: RequestState) -> list[float]:
    ts = req.token_times_s
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]


def build_report(states: list[RequestState], cfg: MetricsConfig) -> SimulationReport:
    completed = [s for s in states if Stage.REQUEST_END in s.stage_times_s]

    all_end_times = [s.stage_times_s[Stage.REQUEST_END] for s in completed]
    window_start = cfg.window_start_s
    auto_end = cfg.window_end_s is None
    window_end = (
        (max(all_end_times) if all_end_times else 0.0)
        if auto_end
        else cfg.window_end_s
    )

    def in_window(t: float | None) -> bool:
        if t is None:
            return False
        if window_end <= window_start:  # degenerate window: keep all completions
            return True
        if auto_end:  # window fitted to the run; include the last completion
            return window_start <= t <= window_end
        return window_start <= t < window_end

    windowed = [s for s in completed if in_window(s.stage_times_s.get(Stage.REQUEST_END))]

    ttft = [v for s in windowed if (v := _ttft(s)) is not None]
    e2e = [v for s in windowed if (v := _e2e(s)) is not None]
    tbt: list[float] = []
    for s in windowed:
        tbt.extend(_tbts(s))

    duration = max(window_end - window_start, 1e-12)
    output_tokens = sum(s.generated_tokens for s in windowed)

    return SimulationReport(
        num_requests=len(states),
        num_completed=len(completed),
        window_start_s=window_start,
        window_end_s=window_end,
        ttft_s=LatencyStats.of(ttft),
        tbt_s=LatencyStats.of(tbt),
        e2e_s=LatencyStats.of(e2e),
        request_throughput_rps=len(windowed) / duration,
        output_token_throughput_tps=output_tokens / duration,
    )


def format_report(report: SimulationReport) -> str:
    def fmt(x: float | None, scale: float = 1e3, unit: str = "ms") -> str:
        return "n/a" if x is None else f"{x * scale:.3f}{unit}"

    lines = [
        "=== PD Throughput Simulation Report ===",
        f"requests: {report.num_requests}  completed: {report.num_completed}",
        f"window: [{report.window_start_s:.4f}, {report.window_end_s:.4f}) s",
        f"TTFT  p50/p90/p99: {fmt(report.ttft_s.p50)}/{fmt(report.ttft_s.p90)}/{fmt(report.ttft_s.p99)}",
        f"TBT   p50/p90/p99: {fmt(report.tbt_s.p50)}/{fmt(report.tbt_s.p90)}/{fmt(report.tbt_s.p99)}",
        f"E2E   p50/p90/p99: {fmt(report.e2e_s.p50)}/{fmt(report.e2e_s.p90)}/{fmt(report.e2e_s.p99)}",
        f"request throughput: {report.request_throughput_rps:.3f} req/s",
        f"output token throughput: {report.output_token_throughput_tps:.3f} tok/s",
    ]
    return "\n".join(lines)
