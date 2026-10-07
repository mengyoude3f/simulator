#!/usr/bin/env python
"""本目录下可直接运行的示例。

    cd pd_throughput_simulator_v5
    python run_example.py          # 回放 example_trace.jsonl（真实生产 trace 前 60 条）
    python run_example.py compare  # 受控对照：只换 workload，其余配置相同
    python run_example.py demo     # 跑内置合成负载 default_demo_scenario()

脚本自行把仓库父目录加入 sys.path，并按自身位置解析 trace 路径，
因此无论从哪个工作目录调用都能跑。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# 包位于父目录之下，需把父目录加入 sys.path 才能 import 本包
sys.path.insert(0, os.path.dirname(_HERE))

from pd_throughput_simulator_v5 import (  # noqa: E402
    CapacityConfig,
    ClusterConfig,
    PDSimulator,
    Scenario,
    SimpleAnalyticCostModel,
    WorkloadConfig,
    build_report,
    default_demo_scenario,
    format_report,
    load_trace_requests,
)

TRACE = os.path.join(_HERE, "example_trace.jsonl")


def build_scenario() -> Scenario:
    return Scenario(
        cluster=ClusterConfig(num_prefill_replicas=1, num_decode_replicas=2),
        capacity=CapacityConfig(
            page_size=16,
            chunked_prefill_size=8192,  # 每次 prefill forward 的 token 预算
            prefill_kv_pages=500_000,  # P 侧 KV 容量（页）
            decode_kv_pages=900_000,  # D 侧 KV 容量（页）
            max_running_requests=128,  # D 侧 continuous batching 并发上限
        ),
        workload=WorkloadConfig(trace_path=TRACE),
        cost_model=SimpleAnalyticCostModel(congestion_alpha=0.3),
    )


def print_input_summary() -> None:
    import statistics

    reqs = load_trace_requests(TRACE)
    ins = [r.input_tokens for r in reqs]
    outs = [r.output_tokens for r in reqs]
    cached = [r.cached_tokens for r in reqs]
    work = [r.prefill_work_tokens for r in reqs]

    def q(v: list[int]) -> str:
        return f"min={min(v):<7} p50={int(statistics.median(v)):<7} max={max(v)}"

    print("=== 输入：example_trace.jsonl ===")
    print(f"请求数: {len(reqs)}")
    print(f"  input_tokens         {q(ins)}")
    print(f"  output_tokens        {q(outs)}")
    print(f"  cached_tokens        {q(cached)}")
    print(f"  prefill_work_tokens  {q(work)}   (= input - cached)")
    ratio = statistics.median(c / i for c, i in zip(cached, ins))
    print(f"  cached 占比 p50       {ratio * 100:.1f}%")
    print(f"  arrival_time_s       0.000 ~ {max(r.arrival_time_s for r in reqs):.3f}")
    print(f"  重复 id 去重          {sum('#dup' in r.request_id for r in reqs)} 条")
    print()


def run_controlled_comparison() -> None:
    """受控对照：cluster / capacity / cost_model 完全相同，只换 workload。

    A 真实 trace（异构形状 + 真实到达时刻）
    B 合成：形状固定为 trace 均值，泊松到达同均值速率  → 隔离"形状异构性"
    C trace 形状 + 完全均匀到达                        → 隔离"到达模式"
    """
    import statistics

    from pd_throughput_simulator_v5 import RequestSpec

    cluster = ClusterConfig(num_prefill_replicas=1, num_decode_replicas=2)
    capacity = CapacityConfig(
        page_size=16,
        chunked_prefill_size=8192,
        prefill_kv_pages=500_000,
        decode_kv_pages=900_000,
        max_running_requests=128,
    )
    cost = SimpleAnalyticCostModel(congestion_alpha=0.3)

    reqs = load_trace_requests(TRACE)
    n = len(reqs)
    span = max(r.arrival_time_s for r in reqs)
    rate = n / span
    mean_in = int(statistics.mean(r.input_tokens for r in reqs))
    mean_out = int(statistics.mean(r.output_tokens for r in reqs))
    mean_cached = int(statistics.mean(r.cached_tokens for r in reqs))

    print("=== 受控对照：只换 workload，其余配置完全相同 ===")
    print(f"共同配置: 1P2D, chunk=8192, KV 500k/900k 页, alpha=0.3")
    print(f"trace 均值: in={mean_in} out={mean_out} cached={mean_cached} 到达率={rate:.4f}/s")
    print()

    def report(scenario: Scenario, explicit_reqs, tag: str) -> None:
        states = PDSimulator(scenario).run(explicit_reqs)
        r = build_report(states, scenario.metrics)
        ms = lambda x: f"{x * 1e3:.1f}"  # noqa: E731
        print(
            f"{tag:24s} TTFT p50={ms(r.ttft_s.p50):>8}ms p99={ms(r.ttft_s.p99):>8}ms"
            f"  p99/p50={r.ttft_s.p99 / r.ttft_s.p50:5.1f}x"
            f"  E2E p50={ms(r.e2e_s.p50):>8}ms  tok={r.output_token_throughput_tps:6.1f}/s"
        )

    report(
        Scenario(cluster=cluster, capacity=capacity, cost_model=cost,
                 workload=WorkloadConfig(trace_path=TRACE)),
        None, "A 真实trace",
    )
    report(
        Scenario(cluster=cluster, capacity=capacity, cost_model=cost,
                 workload=WorkloadConfig(num_requests=n, arrival_process="poisson",
                                         arrival_rate_rps=rate, input_tokens=mean_in,
                                         output_tokens=mean_out, cached_tokens=mean_cached,
                                         seed=1234)),
        None, "B 合成(均值形状)",
    )
    uniform = [
        RequestSpec(f"u{i}", i * (span / n), input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens, cached_tokens=r.cached_tokens)
        for i, r in enumerate(reqs)
    ]
    report(
        Scenario(cluster=cluster, capacity=capacity, cost_model=cost,
                 workload=WorkloadConfig(num_requests=n)),
        uniform, "C trace形状+均匀到达",
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "compare":
        run_controlled_comparison()
        return

    if mode == "demo":
        scenario = default_demo_scenario()
        w = scenario.workload
        print("=== 输入：内置合成负载 default_demo_scenario() ===")
        print(f"请求数: {w.num_requests}  到达: {w.arrival_process} @ {w.arrival_rate_rps}/s")
        print(f"每请求: input={w.input_tokens} output={w.output_tokens} cached={w.cached_tokens}")
        print()
        states = PDSimulator(scenario).run()
        print(format_report(build_report(states, scenario.metrics)))
        return

    print_input_summary()
    scenario = build_scenario()
    sim = PDSimulator(scenario)
    states = sim.run()
    print(format_report(build_report(states, scenario.metrics)))
    total_chunks = sum(sim.sessions[s.spec.request_id].chunks_sent for s in states)
    total_bytes = sum(sim.sessions[s.spec.request_id].transfer_bytes for s in states)
    print(f"KV chunk 流总数: {total_chunks}")
    print(f"KV 传输总量: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
