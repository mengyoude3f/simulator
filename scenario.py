#硬件配置参数
"""Python-dataclass scenario configuration (no JSON loader, no CLI).

A ``Scenario`` fully describes a run: cluster shape, capacity limits, the cost
model, the workload, and the metrics measurement window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cost_model import CostModel, SimpleAnalyticCostModel


@dataclass(frozen=True)
class ClusterConfig:#集群配置
    num_prefill_replicas: int = 1
    num_decode_replicas: int = 1

    def __post_init__(self) -> None:
        if self.num_prefill_replicas < 1 or self.num_decode_replicas < 1:
            raise ValueError("need at least one prefill and one decode replica")


@dataclass(frozen=True)
class CapacityConfig:
    page_size: int = 16
    # Per-replica KV capacity, in pages.
    prefill_kv_pages: int = 4096
    decode_kv_pages: int = 4096
    # Prefill scheduler.
    chunked_prefill_size: int = 2048  # per-forward token budget
    prefill_metadata_buffers: int = 64  # bootstrap → waiting gate #p 侧握手用总缓存
    # Decode scheduler.
    max_running_requests: int = 256
    decode_req_slots: int = 256
    decode_metadata_buffers: int = 256#d 侧握手用总缓存
    decode_reserved_tokens: int = 512  # == SGLang num_reserved_decode_tokens (server_args.py:2550) #decode_prealloc 每请求预留余量

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if self.chunked_prefill_size < 1:
            raise ValueError("chunked_prefill_size must be >= 1 (chunking always on)")

    def pages_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.page_size - 1) // self.page_size#“//”为向下取整，先加 self.page_size - 1，再 // self.page_size，即为向上取整


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int = 128
    arrival_process: str = "poisson"  # "poisson" | "fixed_interval" #到达过程
    arrival_rate_rps: float = 200.0
    input_tokens: int = 512
    output_tokens: int = 128
    cached_tokens: int = 0
    seed: int = 0
    request_id_prefix: str = "req"
    # External request trace (jsonl).  When set, per-request input/output/cached
    # tokens and arrival times come from the file; the generated fields above are
    # ignored.  ``trace_limit`` caps how many records are read (None = all).
    trace_path: str | None = None#从哪里读请求
    trace_limit: int | None = None#最多读多少条

    def __post_init__(self) -> None:
        if self.arrival_process not in ("poisson", "fixed_interval"):
            raise ValueError(f"unknown arrival_process: {self.arrival_process}")
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if self.arrival_rate_rps <= 0:
            raise ValueError("arrival_rate_rps must be > 0")


@dataclass(frozen=True)
class MetricsConfig:#统计窗口
    # Half-open measurement window [start, end); ``None`` end = until last event.
    window_start_s: float = 0.0
    window_end_s: float | None = None


@dataclass(frozen=True)
class Scenario:#上述所有的打包，总配置
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    cost_model: CostModel = field(default_factory=SimpleAnalyticCostModel)
    # Guard against runaway simulations.
    max_events: int | None = 5_000_000


def default_demo_scenario() -> Scenario:#默认配置
    """A small, self-consistent 1P1D scenario used by the demo and e2e test."""
    return Scenario(
        cluster=ClusterConfig(num_prefill_replicas=1, num_decode_replicas=1),
        capacity=CapacityConfig(
            page_size=16,
            prefill_kv_pages=8192,
            decode_kv_pages=8192,
            chunked_prefill_size=2048,
            max_running_requests=128,
        ),
        workload=WorkloadConfig(
            num_requests=64,
            arrival_process="poisson",
            arrival_rate_rps=120.0,
            input_tokens=1024,
            output_tokens=64,
            cached_tokens=0,
            seed=1234,
        ),
        metrics=MetricsConfig(window_start_s=0.0, window_end_s=None),
        cost_model=SimpleAnalyticCostModel(),
    )
#finish review 2026.9.16
#finish review version-“允许 input/output/cached tokens 各异的 request 到达” 2026.9.18