"""PD disaggregation throughput simulator (v5 rewrite).

A lean discrete-event model of SGLang PD-disaggregated serving whose P-line and
D-line state machines follow ``SGLANG_SOURCE_FINDINGS.md`` exactly.  Only the
overlap scheduler loop and chunked prefill are modelled.
"""

from __future__ import annotations

from .cost_model import CostModel, SimpleAnalyticCostModel
from .kv_transfer import KVPoll, KVTransferSession
from .metrics import LatencyStats, SimulationReport, build_report, format_report
from .network import NetworkFabric
from .request import (
    DLinePhase,
    PLinePhase,
    RequestSpec,
    RequestState,
    Stage,
)
from .scenario import (
    CapacityConfig,
    ClusterConfig,
    MetricsConfig,
    Scenario,
    WorkloadConfig,
    default_demo_scenario,
)
from .simulator import PDSimulator
from .trace_loader import load_trace_requests
from .workload import build_requests

__all__ = [
    "CostModel",
    "SimpleAnalyticCostModel",
    "KVPoll",
    "KVTransferSession",
    "NetworkFabric",
    "LatencyStats",
    "SimulationReport",
    "build_report",
    "format_report",
    "RequestSpec",
    "RequestState",
    "Stage",
    "PLinePhase",
    "DLinePhase",
    "CapacityConfig",
    "ClusterConfig",
    "MetricsConfig",
    "Scenario",
    "WorkloadConfig",
    "default_demo_scenario",
    "PDSimulator",
    "build_requests",
    "load_trace_requests",
]
