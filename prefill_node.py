"""Prefill node (KV sender) state machine.

Implements the 9-stage PREFILL lifecycle from ``SGLANG_SOURCE_FINDINGS.md`` §二,
driven by the overlap scheduler loop (§一) and chunked prefill.

Queues: ``bootstrap_queue`` → ``waiting_queue`` → ``inflight_queue``.

Overlap loop per timed iteration: **select current forward batch → launch it
(schedule forward_done at now+dur) → commit the previous batch's results**.  So
the current batch is selected on state that does not yet reflect the previous
batch's forward result (the one-step staleness), while forwards run back-to-back
with no CPU gap.  Chunk *range* bookkeeping advances at select time (like
``set_extend_range``); only the result-dependent actions (first-token sampling
and KV send kickoff) are deferred to commit.

KV is streamed **per chunk**: each committed chunk pushes its own slice of the
prompt KV (cursor ``req.kv_send_idx``, == SGLang's ``req.start_send_idx``), so
earlier chunks fly while later ones are still computing.  The request only
reaches KV ``SUCCESS`` once the final chunk's transfer lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .cost_model import CostModel
from .kv_transfer import KVTransferSession
from .network import NetworkFabric
from .request import PLinePhase, RequestState, Stage
from .scenario import CapacityConfig
from .sim_clock import DiscreteEventLoop, Event

_FORWARD_DONE = "prefill_forward_done"


@dataclass
class _WorkItem:#某请求某轮工作
    req: RequestState
    tokens: int#这一轮该请求算了多少
    finishes_prefill: bool#这一轮是否为该请求最后一轮
    # Absolute prompt position covered after this chunk (== SGLang's
    # ``req.tmp_end_idx``).  Captured at *select* time because
    # ``prefill_done_tokens`` has already advanced past this chunk by the time
    # the batch is committed.
    send_end_idx: int = 0#本轮 forward 后，到达的 kv 绝对下表（cached_tokens + generated_tokens）


@dataclass
class _Batch:
    items: list[_WorkItem] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:#本轮总 tokens
        return sum(i.tokens for i in self.items)

    @property
    def size(self) -> int:#本轮请求数
        return len(self.items)


class PrefillNode:
    def __init__(
        self,
        replica_id: str,
        loop: DiscreteEventLoop,#离散事件模拟（DES）引擎
        cost: CostModel,#时间成本模型
        capacity: CapacityConfig,
        get_session: Callable[[str], KVTransferSession],#取哪份 bootstrap_room
        wake_decode: Callable[[RequestState], None],#配对的 D replica
        fabric: NetworkFabric,#全集群共享的带宽 fabric
    ) -> None:
        self.replica_id = replica_id
        self.loop = loop
        self.cost = cost
        self.cap = capacity
        self.get_session = get_session
        self.wake_decode = wake_decode

        self.bootstrap_queue: list[RequestState] = []#等待 D 握手
        self.waiting_queue: list[RequestState] = []#已握手，等待前向传播
        self.inflight_queue: list[RequestState] = []#前向传播完成，等待 kv_transfer 完成
        self.chunked_req: RequestState | None = None#已预留 kv，已从 waiting_queue 中取出，但没完事的请求；至多一个（SGLang 就这么写的）

        self.free_kv_pages = capacity.prefill_kv_pages
        self.free_metadata = capacity.prefill_metadata_buffers#握手用缓存
        self._alloc_pages: dict[str, int] = {}#每个请求占了多少页

        self._busy = False#是否正在前向传播
        self._boundary_pending = False#是否已安排边界

        # Shared bandwidth fabric: this replica's egress NIC, the target decode
        # replica's ingress NIC, and the cluster switch all throttle the flow.
        self.fabric = fabric

    # --- external entry ----------------------------------------------------
    def admit(self, req: RequestState, now: float) -> None:#接受请求，等待 D 侧握手
        """Router hands a request to this prefill replica (request_arrival →
        PREFILL_PREPARE / bootstrap queue entry)."""
        # SGLang rejects an over-capacity request at create_sender time
        # (``_check_if_req_exceed_kv_capacity``, prefill.py:317-325) and aborts
        # it with HTTP 400.  We have no abort terminal state, so fail fast:
        # otherwise the request would sit in the waiting queue forever and the
        # run would silently report fewer completions than requests.
        need = self.cap.pages_for_tokens(req.spec.prefill_work_tokens)
        if need > self.cap.prefill_kv_pages:
            raise ValueError(
                f"{req.spec.request_id}: prefill needs {need} KV pages but the "
                f"replica only has {self.cap.prefill_kv_pages}; SGLang would "
                f"abort this request (prefill.py:317)"
            )
        req.prefill_replica_id = self.replica_id
        req.p_phase = PLinePhase.BOOTSTRAP_QUEUE
        req.mark(Stage.PREFILL_PREPARE, now)
        self.bootstrap_queue.append(req)
        self.wake(now)

    def wake(self, now: float) -> None:#状态推动统一接口
        self.poll(now)#推进所有能推进的
        if not self._busy and not self._boundary_pending and self._has_launchable_work():
            self._boundary_pending = True
            self.loop.schedule_at(#在某绝对时刻排一个边界
                now, _FORWARD_DONE, self._on_boundary, self.replica_id, {"batch": None}
            )

    # --- zero-cost bookkeeping (polled every wake / boundary) --------------
    def poll(self, now: float) -> None:#边界处不耗时记账
        self._pop_bootstrapped(now)
        self._process_inflight(now)

    def _pop_bootstrapped(self, now: float) -> None:#（如 D 侧握手）从等待握手推进到准备 forward，暂不分配 kv
        # bootstrap_queue → waiting_queue: sender reached WAITING_FOR_INPUT
        # (decode already sent its dst metadata) AND a metadata buffer is free.
        still_waiting: list[RequestState] = []
        for req in self.bootstrap_queue:
            session = self.get_session(req.spec.request_id)
            if session.p_bootstrap_done and self.free_metadata > 0:
                self.free_metadata -= 1
                req.mark(Stage.PREFILL_BOOTSTRAP, now)
                req.p_phase = PLinePhase.WAITING_QUEUE
                self.waiting_queue.append(req)
            else:
                still_waiting.append(req)
        self.bootstrap_queue = still_waiting

    def _process_inflight(self, now: float) -> None:#（如 kv_transfer 完成）从等待传输完成推进到完成，释放资源
        # inflight_queue: on KVPoll.Success release KV + complete.
        remaining: list[RequestState] = []
        for req in self.inflight_queue:
            session = self.get_session(req.spec.request_id)
            if session.p_transfer_succeeded:
                req.mark(Stage.PREFILL_TRANSFER_KV_CACHE, now)
                self._release_kv(req)
                self.free_metadata += 1
                req.mark(Stage.PREFILL_COMPLETE, now)
                req.p_phase = PLinePhase.DONE
            else:
                remaining.append(req)
        self.inflight_queue = remaining

    def _has_forward_work(self) -> bool:#有没有活干
        return self.chunked_req is not None or len(self.waiting_queue) > 0

    def _has_launchable_work(self) -> bool:#有没有能干的活（kv 够不够），fcfs
        # A waiting request is only launchable once its KV can actually be
        # reserved; otherwise scheduling a forward iteration would select
        # nothing and (if used to self-reschedule) spin at this timestamp.
        if self.chunked_req is not None:#chunked_req 已经分配 kv，无需考虑 kv 不足
            return True
        if self.waiting_queue:#只看队首，fcfs
            head = self.waiting_queue[0]
            need = self.cap.pages_for_tokens(head.spec.prefill_work_tokens)
            return need <= self.free_kv_pages
        return False

    # --- forward loop ------------------------------------------------------
    def _on_boundary(self, event: Event) -> None:
        now = self.loop.clock.now_s
        self._boundary_pending = False
        self._busy = False
        finished: _Batch | None = event.payload.get("batch")#获取刚跑完的一轮

        self.poll(now)#1.零耗时记账
        current = self._select_forward_batch(now)#2.选择下一轮
        if finished is not None:
            self._commit_forward_batch(finished, now)#3.提交上一轮
            self.poll(now)

        if current.size > 0:#4.启动下一轮
            dur = self.cost.prefill_forward_s(current.total_tokens, current.size)
            self._busy = True
            self.loop.schedule_after(
                dur, _FORWARD_DONE, self._on_boundary, self.replica_id, {"batch": current}
            )
        elif self._has_launchable_work():#如没有选出下一轮，在上一轮提交后，再次检查有没有可以跑的请求，避免空转
            # Commit/poll may have exposed work the stale select missed; run an
            # immediate follow-up iteration (same timestamp) to pick it up.
            self._boundary_pending = True
            self.loop.schedule_at(
                now, _FORWARD_DONE, self._on_boundary, self.replica_id, {"batch": None}
            )

    def _select_forward_batch(self, now: float) -> _Batch:
        """PrefillAdder: fill a chunked-prefill token budget from the in-flight
        chunked req first, then new requests from the waiting queue."""
        batch = _Batch()
        budget = self.cap.chunked_prefill_size

        # 1) continue the in-flight chunked request (KV already reserved).
        if self.chunked_req is not None:#优先跑跑到一半的
            budget = self._add_chunk(self.chunked_req, budget, batch)

        # 2) admit new requests until the token budget or KV pool is exhausted.
        while budget > 0 and self.waiting_queue:#还有余量，再从队首拉新请求
            req = self.waiting_queue[0]
            need_pages = self.cap.pages_for_tokens(req.spec.prefill_work_tokens)
            if need_pages > self.free_kv_pages:
                break  # backpressure: cannot reserve KV for this request yet
            self.free_kv_pages -= need_pages
            self._alloc_pages[req.spec.request_id] = need_pages
            self.waiting_queue.pop(0)
            req.mark(Stage.PREFILL_WAITING, now)
            req.p_phase = PLinePhase.FORWARDING
            budget = self._add_chunk(req, budget, batch)
            if self.chunked_req is not None:#没跑完优先继续跑，而非拉新（没有余量了，拉新无意义）
                break  # a request became the new chunked req; stop admitting

        return batch

    def _add_chunk(self, req: RequestState, budget: int, batch: _Batch) -> int:#prefill 一个 chunk
        remaining = req.prefill_remaining_tokens
        if remaining <= budget:
            take = remaining
            finishes = True
        else:
            # page-aligned truncated chunk, at least one page.
            page = self.cap.page_size
            take = max(page, (budget // page) * page)#以整页为单位，至少一页
            take = min(take, remaining)
            # The one-page floor can still swallow everything left (leftover
            # budget < page_size and remaining < page_size), in which case this
            # *is* the final chunk.  Deciding on `take` rather than on the
            # branch avoids a spurious extra zero-token forward iteration.
            finishes = take >= remaining
        # advance range bookkeeping at select time (== set_extend_range).
        req.prefill_done_tokens += take
        # Absolute prompt position this chunk covers: the cached prefix is
        # already resident, so the first chunk's send also carries it.
        send_end_idx = req.spec.cached_tokens + req.prefill_done_tokens
        batch.items.append(
            _WorkItem(
                req=req,
                tokens=take,
                finishes_prefill=finishes,
                send_end_idx=send_end_idx,
            )
        )
        if finishes:
            self.chunked_req = None
        else:
            self.chunked_req = req
        return budget - take

    def _commit_forward_batch(self, batch: _Batch, now: float) -> None:#推进 forward 完的 chunk，推进请求到 transfer 阶段；若是末块，则请求打点（transfer），标志着最后一个 chunk 开始发送
        for item in batch.items:
            req = item.req
            if req.p_phase != PLinePhase.FORWARDING:
                continue  # already handled / retracted
            if item.finishes_prefill:#所有 chunk 都 forward 完了才会触发
                req.mark(Stage.PREFILL_FORWARD, now)
                session = self.get_session(req.spec.request_id)
                session.p_mark_forward_done()  # first token sampled here #首 token 采样
                req.p_phase = PLinePhase.TRANSFERRING
                self.inflight_queue.append(req)
                # Request-level "entered the transfer queue" marker; matches
                # SGLang's set_prefill_transfer_queue_entry_time (prefill.py:696),
                # which fires right after the *final* send_kv_chunk.
                req.mark(Stage.KV_TRANSFER_START, now)
            # Every finished chunk streams its own slice of the prompt KV, so
            # earlier chunks fly while later ones are still being computed
            # (== SGLang send_kv_chunk / req.start_send_idx).
            self._send_chunk(req, item, now)#不管所有 chunk 是否都 forward 完，每个 chunk 都要发送

    def _send_chunk(self, req: RequestState, item: _WorkItem, now: float) -> None:#发送单个 chunk
        session = self.get_session(req.spec.request_id)
        if not session.p_can_send():
            # Bootstrap not finalized yet: keep computing without sending; the
            # cursor stays put so a later chunk covers the accumulated range
            # (== SGLang prefill.py:1002 "still bootstrapping").
            return
        start_idx = req.kv_send_idx
        end_idx = item.send_end_idx
        num_kv_tokens = max(0, end_idx - start_idx)
        if num_kv_tokens == 0 and not item.finishes_prefill:
            return  # nothing new to push (== should_send_kv_chunk num_pages > 0)
        req.kv_send_idx = end_idx
        transfer_bytes = self.cost.kv_transfer_bytes(num_kv_tokens)
        session.p_start_chunk_send(transfer_bytes, is_last=item.finishes_prefill)#是否完成 forward 决定了是否为 last chunk
        req.mark(Stage.KV_CHUNK_SEND, now)
        # Fixed setup latency (bootstrap+route) is serial; then the shared-
        # bandwidth data phase enters the fabric (contends on both NICs + switch).
        flow_id = f"{req.spec.request_id}#c{session.chunks_sent}"
        self.loop.schedule_after(#此处仅为固定耗时
            self.cost.transfer_latency_s(),
            "kv_data_enter",
            self._enter_link,
            req.spec.request_id,
            {"req": req, "bytes": transfer_bytes, "flow_id": flow_id},
        )

    def _enter_link(self, event: Event) -> None:#把单个 chunk 所在的流从固定耗时推进到带宽争夺
        req: RequestState = event.payload["req"]
        transfer_bytes: int = event.payload["bytes"]
        flow_id: str = event.payload["flow_id"]
        self.fabric.submit(
            flow_id,  # 每个 chunk 一条独立的流
            src=self.replica_id,  # 本 P 的出口网卡
            dst=req.decode_replica_id,  # 目标 D 的入口网卡
            total_bytes=transfer_bytes,
            on_done=lambda: self._finish_chunk_transfer(req),#lambda：传完后回调
        )

    def _finish_chunk_transfer(self, req: RequestState) -> None:#单个 chunk
        now = self.loop.clock.now_s
        session = self.get_session(req.spec.request_id)
        if not session.complete_chunk_transfer():
            return  # more chunks still outstanding; request not done yet
        self.wake(now)  # prefill inflight poll → completion
        self.wake_decode(req)  # decode transfer-queue poll → Success

    # --- KV accounting -----------------------------------------------------
    def _release_kv(self, req: RequestState) -> None:
        pages = self._alloc_pages.pop(req.spec.request_id, 0)
        self.free_kv_pages += pages
#finish review 2026.9.17
#finish review version-“考虑 kv_transfer 中带宽争夺” 2026.9.18
#finish review version-“添加 D/集群带宽竞争 + P/D/集群并发增高导致总流量非线性衰减” 2026.9.20（这个文件逻辑没变）
#finish review version-"kv_transfer 由逐 request 改为逐 chunk" 2026.9.20