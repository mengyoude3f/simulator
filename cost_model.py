#状态机耗时/实时带宽接口定义及默认简单实现
"""Pluggable cost model with a simple analytic default.

The state machine's *timing* comes entirely from a ``CostModel``.  The default
``SimpleAnalyticCostModel`` produces self-consistent throughput/latency numbers
from linear coefficients; a calibrated external profile can subclass or replace
it without touching the scheduler logic.

KV-transfer timing is split into pluggable pieces so bandwidth *contention* can
be modelled by ``network.NetworkFabric``:
  * ``transfer_latency_s`` — fixed per-transfer setup (bootstrap + route), does
    not compete for bandwidth;
  * ``egress_/ingress_/fabric_capacity_bytes_s(n)`` — the effective *total*
    capacity of each shared resource when ``n`` flows use it.  How that capacity
    is divided among flows is decided by the fabric (max-min fair), not here;
    override these for a different congestion curve.

None of these coefficients are measured H200 / GLM-5.2 values — they are
explicit simulation inputs supplied via ``scenario.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CostModel(Protocol):#定义接口：含这八个方法即可
    def prefill_forward_s(self, num_tokens: int, batch_size: int) -> float: ...

    def decode_iter_s(self, batch_size: int) -> float: ...#单次迭代耗时

    def kv_transfer_bytes(self, num_kv_tokens: int) -> int: ...

    def transfer_latency_s(self) -> float: ...#固定前缀：bootstrap+route，不争带宽 #固定耗时

    # 三级共享资源的"有效总容量"（不是每流速率）；怎么在流之间分配由
    # NetworkFabric 的 max-min fair 决定。参数是该资源上的并发流数，
    # 便于建模拥塞导致的非线性容量衰减。
    def egress_capacity_bytes_s(self, num_flows: int) -> float: ...#单个 P replica 出口网卡 #单个 P 出口实时总带宽

    def ingress_capacity_bytes_s(self, num_flows: int) -> float: ...#单个 D replica 入口网卡 #单个 D 入口实时总带宽

    def fabric_capacity_bytes_s(self, num_flows: int) -> float: ...#集群 fabric 总带宽 #集群实时总带宽

    def bootstrap_handshake_s(self) -> float: ...


@dataclass(frozen=True)
class SimpleAnalyticCostModel:#默认简单实现，全部为线性函数
    """Linear defaults; every term is an explicit input.

    prefill_forward_s = prefill_base_s + prefill_per_token_s * tokens
                        + prefill_per_req_s * batch_size
    decode_iter_s     = decode_base_s + decode_per_req_s * batch_size
    kv bytes          = num_kv_tokens * kv_bytes_per_token
    transfer_latency  = bootstrap_latency_s + route_latency_s
    capacity(n)       = base_bandwidth / (1 + congestion_alpha * (n - 1))#P/D/集群实时总流量
    """

    prefill_base_s: float = 1.0e-3
    prefill_per_token_s: float = 5.0e-5
    prefill_per_req_s: float = 2.0e-4

    decode_base_s: float = 5.0e-4
    decode_per_req_s: float = 1.0e-4

    kv_bytes_per_token: int = 4096
    bootstrap_latency_s: float = 1.0e-4
    route_latency_s: float = 2.0e-5

    # 三级共享带宽（字节/秒）。egress/ingress 是"每个 replica 一块网卡"的容量；
    # fabric 是整个集群共享的交换容量。
    egress_bandwidth_bytes_s: float = 25.0e9  # 200 Gbit/s NIC per prefill replica
    ingress_bandwidth_bytes_s: float = 25.0e9  # 200 Gbit/s NIC per decode replica
    fabric_bandwidth_bytes_s: float = 400.0e9  # cluster-wide switching capacity

    # 非线性拥塞系数：>0 时有效容量随并发流数衰减（incast、拥塞控制退避、
    # 重传等的粗粒度近似）。0 = 纯线性上限（默认，不衰减）。
    congestion_alpha: float = 0.0#随并发增加，P/D/集群总流量上限非线性衰减参数，暂未启用

    def prefill_forward_s(self, num_tokens: int, batch_size: int) -> float:
        return (
            self.prefill_base_s
            + self.prefill_per_token_s * num_tokens
            + self.prefill_per_req_s * batch_size
        )

    def decode_iter_s(self, batch_size: int) -> float:
        return self.decode_base_s + self.decode_per_req_s * batch_size

    def kv_transfer_bytes(self, num_kv_tokens: int) -> int:
        return num_kv_tokens * self.kv_bytes_per_token

    def transfer_latency_s(self) -> float:
        return self.bootstrap_latency_s + self.route_latency_s

    def _derate(self, capacity: float, num_flows: int) -> float:#随并发增加，P/D/集群总流量上限非线性衰减模拟，暂未启用
        # Non-linear congestion: effective capacity shrinks as concurrency grows.
        if self.congestion_alpha <= 0.0 or num_flows <= 1:
            return capacity
        return capacity / (1.0 + self.congestion_alpha * (num_flows - 1))

    def egress_capacity_bytes_s(self, num_flows: int) -> float:
        return self._derate(self.egress_bandwidth_bytes_s, num_flows)

    def ingress_capacity_bytes_s(self, num_flows: int) -> float:
        return self._derate(self.ingress_bandwidth_bytes_s, num_flows)

    def fabric_capacity_bytes_s(self, num_flows: int) -> float:
        return self._derate(self.fabric_bandwidth_bytes_s, num_flows)

    def bootstrap_handshake_s(self) -> float:
        return self.bootstrap_latency_s
#finish review 2026.9.16
#finish review version-“考虑 kv_transfer 中带宽争夺” 2026.9.18
#finish review version-“添加 D/集群带宽竞争 + P/D/集群并发增高导致总流量非线性衰减” 2026.9.20