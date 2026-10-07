#总模拟器的装配
"""Top-level PD simulator: router + replicas + DES driver.

Wires a set of prefill and decode replicas to the discrete-event loop, routes
each arriving request to one prefill replica and one decode replica (round
robin), and shares a per-request ``KVTransferSession`` between the paired lines.
"""

from __future__ import annotations

from .decode_node import DecodeNode
from .kv_transfer import KVTransferSession
from .network import NetworkFabric
from .prefill_node import PrefillNode
from .request import RequestSpec, RequestState, Stage
from .scenario import Scenario
from .sim_clock import DiscreteEventLoop, Event
from .workload import build_requests


class PDSimulator:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario#硬件配置参数
        self.loop = DiscreteEventLoop()
        self.sessions: dict[str, KVTransferSession] = {}#SGLang 的 bootstrap_room，每个 request 各一个
        self.states: dict[str, RequestState] = {}
        self.completed: list[RequestState] = []

        cost = scenario.cost_model
        cap = scenario.capacity

        # One shared fabric for the whole cluster: it owns every prefill egress
        # NIC, every decode ingress NIC, and the cluster-wide switch capacity,
        # so a flow's rate is the joint result of all three.
        self.fabric = NetworkFabric(self.loop, cost)

        self.prefill_nodes = [
            PrefillNode(
                replica_id=f"p{i}",
                loop=self.loop,
                cost=cost,
                capacity=cap,
                get_session=self._session,
                wake_decode=self._wake_decode_for,
                fabric=self.fabric,
            )
            for i in range(scenario.cluster.num_prefill_replicas)
        ]
        self.decode_nodes = [
            DecodeNode(
                replica_id=f"d{i}",
                loop=self.loop,
                cost=cost,
                capacity=cap,
                get_session=self._session,
                wake_prefill=self._wake_prefill_for,
                on_request_end=self._on_request_end,
            )
            for i in range(scenario.cluster.num_decode_replicas)
        ]
        self._p_by_id = {n.replica_id: n for n in self.prefill_nodes}
        self._d_by_id = {n.replica_id: n for n in self.decode_nodes}

    # --- session / wake plumbing ------------------------------------------
    def _session(self, request_id: str) -> KVTransferSession:
        return self.sessions[request_id]

    def _wake_decode_for(self, req: RequestState) -> None:#prefill 叫醒 decode
        self._d_by_id[req.decode_replica_id].wake(self.loop.clock.now_s)

    def _wake_prefill_for(self, req: RequestState) -> None:#decode 叫醒 prefill
        self._p_by_id[req.prefill_replica_id].wake(self.loop.clock.now_s)

    def _on_request_end(self, req: RequestState, now: float) -> None:
        self.completed.append(req)

    # --- run ---------------------------------------------------------------
    def run(self, requests: list[RequestSpec] | None = None) -> list[RequestState]:
        if requests is None:
            requests = build_requests(self.scenario.workload)#压力规模
        for idx, spec in enumerate(requests):
            if spec.request_id in self.states:
                # Per-request state and the KV session are keyed by id; a
                # duplicate would make two requests share one transfer session.
                raise ValueError(f"duplicate request_id: {spec.request_id!r}")
            state = RequestState(spec=spec)
            self.states[spec.request_id] = state
            self.sessions[spec.request_id] = KVTransferSession(spec.request_id)
            self.loop.schedule_at(
                spec.arrival_time_s,
                "request_arrival",
                self._on_arrival,
                spec.request_id,
                {"state": state, "index": idx},
            )
        self.loop.run(max_events=self.scenario.max_events)
        return list(self.states.values())

    def _on_arrival(self, event: Event) -> None:#请求到达：1.轮询分配给 P / D；2.先 D 后 P（同源码，D 要先预分配）
        now = self.loop.clock.now_s
        state: RequestState = event.payload["state"]
        idx: int = event.payload["index"]
        state.mark(Stage.REQUEST_ARRIVAL, now)
        # Router: round-robin across prefill and decode replicas.  In PD the
        # request is sent to both lines in parallel.
        p = self.prefill_nodes[idx % len(self.prefill_nodes)]
        d = self.decode_nodes[idx % len(self.decode_nodes)]
        d.admit(state, now)  # decode preallocates first (source-order handshake)
        p.admit(state, now)
#finish review 2026.9.17
#finish review version-“添加 D/集群带宽竞争 + P/D/集群并发增高导致总流量非线性衰减” 2026.9.20（这个文件逻辑没变）