#KV 传输的多级共享带宽模型（P 出口 / D 入口 / 集群 fabric，max-min fair）
"""Multi-resource shared-bandwidth fabric for KV transfers.

One *flow* = one request's KV data phase.  It simultaneously traverses three
shared resources:

  * ``egress``  — the sending prefill replica's NIC (one per prefill replica);
  * ``ingress`` — the receiving decode replica's NIC (one per decode replica);
  * ``fabric``  — the cluster-wide switching capacity shared by all flows.

Rates are assigned by **max-min fair** (progressive water-filling) allocation,
so a flow throttled at one resource releases the capacity it cannot use back to
the flows sharing its other resources.  A single central allocator is required
because a flow's rate is the *joint* result of all three constraints; separate
per-NIC links could not see each other's limits.

Each resource's *effective capacity* is queried from the cost model at that
resource's own concurrency (``*_capacity_bytes_s(num_flows)``), which is how
non-linear congestion (incast collapse, congestion-control backoff, retransmits)
enters the model.  With ``congestion_alpha = 0`` capacity is a plain ceiling.

The fabric models only the shared *data* phase; the fixed per-transfer setup
latency (bootstrap + route) is applied by the caller before ``submit``.

Note: SGLang itself performs no bandwidth arbitration — it hands a transfer to
the RDMA/transport layer.  This module is an explicit physical approximation and
is fully pluggable through the cost model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .cost_model import CostModel
from .sim_clock import DiscreteEventLoop, Event

_EPS_BYTES = 1e-6
_REL_TOL = 1e-12#流速相对误差参数，用于判等与最小流速流速相同的流，冻结这些流

# Resource keys are (kind, owner_id); the fabric is a single global resource.
_FABRIC = ("fabric", "")#查询集群流量的键值（P/D 格式分别为"egress", flow.src)/("ingress", flow.dst)）


@dataclass
class _Flow:#一条在传的流（= 一个请求的 KV 数据相）
    request_id: str
    src: str  # prefill replica id → its egress NIC
    dst: str  # decode replica id → its ingress NIC
    remaining: float#剩余字节（截至 _last_update_s 的快照）
    on_done: Callable[[], None]


class NetworkFabric:#全局唯一，统一分配三级带宽
    def __init__(self, loop: DiscreteEventLoop, cost: CostModel) -> None:
        self.loop = loop#des 引擎
        self.cost = cost
        self._flows: dict[str, _Flow] = {}
        # Per-flow rates in force since ``_last_update_s`` (recomputed whenever
        # the set of flows changes).
        self._rates: dict[str, float] = {}#各流当前速率（区间内恒定）
        self._last_update_s = 0.0#记账基准时刻
        self._epoch = 0#事件代号，用于作废旧事件

    @property
    def num_active(self) -> int:
        return len(self._flows)

    def active_on_egress(self, src: str) -> int:
        return sum(1 for f in self._flows.values() if f.src == src)

    def active_on_ingress(self, dst: str) -> int:
        return sum(1 for f in self._flows.values() if f.dst == dst)

    def rate_of(self, request_id: str) -> float:#查询某条流实施速率，若没有该流返回 0
        return self._rates.get(request_id, 0.0)

    # --- flow lifecycle ----------------------------------------------------
    def submit(
        self,
        request_id: str,
        src: str,
        dst: str,
        total_bytes: int,
        on_done: Callable[[], None],
    ) -> None:
        now = self.loop.clock.now_s
        self._advance(now)  # settle the past at the old rates #推进到此刻，各流减少剩余传输量
        self._flows[request_id] = _Flow(
            request_id=request_id,
            src=src,
            dst=dst,
            remaining=float(total_bytes),
            on_done=on_done,
        )
        self._reschedule()  # re-allocate with the new flow present

    # --- fluid bookkeeping -------------------------------------------------
    def _advance(self, now: float) -> None:#把时间"流"过去，按各流自己的速率扣减
        dt = now - self._last_update_s
        if dt > 0 and self._flows:
            for rid, flow in self._flows.items():
                flow.remaining -= self._rates.get(rid, 0.0) * dt
        self._last_update_s = now

    def _resources_of(self, flow: _Flow) -> tuple:#一条流占用的三个资源
        return (("egress", flow.src), ("ingress", flow.dst), _FABRIC)

    def _capacities(self) -> dict[tuple, float]:#P/D/集群实时总流量
        """Effective capacity of every resource, at its own concurrency."""
        egress_n = Counter(f.src for f in self._flows.values())#Counter：元素为键值对（名称，出现次数）的字典；这里统计各 P/D 出现的次数
        ingress_n = Counter(f.dst for f in self._flows.values())
        caps: dict[tuple, float] = {}
        for src, n in egress_n.items():
            caps[("egress", src)] = self.cost.egress_capacity_bytes_s(n)
        for dst, n in ingress_n.items():
            caps[("ingress", dst)] = self.cost.ingress_capacity_bytes_s(n)
        caps[_FABRIC] = self.cost.fabric_capacity_bytes_s(len(self._flows))
        return caps

    def _compute_rates(self) -> dict[str, float]:#每部分流量被占用该资源的所有流均分，找到最小流量，去除对应流，剩下的流重新计算，直到流空
        """Max-min fair allocation by progressive water-filling.

        Each round: give every still-unassigned flow the smallest per-resource
        fair share it faces; the global minimum of those is the next bottleneck
        level, and every flow sitting at that level is frozen there.  Freezing at
        least one flow per round guarantees termination.
        """
        if not self._flows:
            return {}
        remaining_cap = self._capacities()
        pending = dict(self._flows)
        rates: dict[str, float] = {}

        while pending:
            counts: Counter = Counter()
            for flow in pending.values():
                for res in self._resources_of(flow):
                    counts[res] += 1
            share = {
                res: (max(0.0, remaining_cap[res]) / n if n > 0 else float("inf"))
                for res, n in counts.items()
            }
            candidate = {
                rid: min(share[res] for res in self._resources_of(flow))
                for rid, flow in pending.items()
            }
            level = min(candidate.values())
            frozen = [rid for rid, v in candidate.items() if v <= level * (1 + _REL_TOL)]
            if not frozen:  # defensive: float noise
                frozen = [min(candidate, key=lambda r: candidate[r])]#在 candidate 中选出使 candidate[r] min 的 r
            for rid in frozen:
                flow = pending.pop(rid)
                rates[rid] = level
                for res in self._resources_of(flow):
                    remaining_cap[res] -= level
        return rates

    def _reschedule(self) -> None:#重算分配 + 排下一个完成事件（相对当前时刻）
        # Bump the epoch so any previously scheduled completion becomes stale.
        self._epoch += 1
        self._rates = self._compute_rates()
        if not self._flows:
            return
        etas = [#各正在传的流传完耗时
            max(0.0, f.remaining) / self._rates[rid]
            for rid, f in self._flows.items()
            if self._rates.get(rid, 0.0) > 0.0
        ]
        if not etas:
            # Every flow has zero rate → some resource was configured with zero
            # (or negative) capacity, so nothing can ever complete.  Fail fast
            # instead of silently hanging with unfinished requests.
            raise ValueError(
                "NetworkFabric: all active flows have zero rate; check that "
                "egress/ingress/fabric capacities are positive "
                f"(active flows: {len(self._flows)})"
            )
        self.loop.schedule_after(#最早的流传完事件发生时处理边界事件
            min(etas), "kv_fabric_complete", self._on_complete, "", {"epoch": self._epoch}
        )

    def _on_complete(self, event: Event) -> None:
        if event.payload["epoch"] != self._epoch:
            return  # superseded by a submit/completion since this was scheduled
        now = self.loop.clock.now_s
        self._advance(now)
        done = [rid for rid, f in self._flows.items() if f.remaining <= _EPS_BYTES]
        if not done and self._flows:
            # Floating-point drift can leave the just-finished flow a hair above
            # the byte epsilon; a non-stale completion always means the smallest
            # flow is done, so force-remove it to guarantee forward progress
            # (otherwise a sub-ULP eta reschedules at the same timestamp forever).
            done = [min(self._flows, key=lambda r: self._flows[r].remaining)]
        callbacks = [self._flows.pop(rid).on_done for rid in done]
        self._reschedule()  # re-allocate among the survivors
        for cb in callbacks:
            cb()  # may kick new work → its own submit/reschedule
#finish review version-“添加 D/集群带宽竞争 + P/D/集群并发增高导致总流量非线性衰减” 2026.9.20