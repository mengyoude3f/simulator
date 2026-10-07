"""Decode node (KV receiver) state machine.

Implements the 9-stage DECODE lifecycle from ``SGLANG_SOURCE_FINDINGS.md`` §二.
The decode node receives the request *before* KV arrives: it preallocates its
own KV slots, tells prefill where to write (``send_metadata``), waits for the
transfer, then replays the prefill-sampled first token and runs the decode loop.

Queues: ``prealloc_queue`` → ``transfer_queue`` → ``waiting_queue`` → running.

The queue transitions and PREBUILT admission are zero-cost bookkeeping (``poll``).
Only the autoregressive decode iteration is timed, driven by the overlap loop:
**select the running batch → launch decode forward → commit the previous
iteration's token emissions**.  Over-selection (a request finished by the
previous, not-yet-committed iteration) is tolerated as wasted work.
"""

from __future__ import annotations

from typing import Callable

from .cost_model import CostModel
from .kv_transfer import KVTransferSession
from .request import DLinePhase, RequestState, Stage
from .scenario import CapacityConfig
from .sim_clock import DiscreteEventLoop, Event

_DECODE_DONE = "decode_iter_done"


class DecodeNode:
    def __init__(
        self,
        replica_id: str,
        loop: DiscreteEventLoop,#离散事件仿真引擎
        cost: CostModel,#耗时模型
        capacity: CapacityConfig,
        get_session: Callable[[str], KVTransferSession],
        wake_prefill: Callable[[RequestState], None],
        on_request_end: Callable[[RequestState, float], None],
    ) -> None:
        self.replica_id = replica_id
        self.loop = loop
        self.cost = cost
        self.cap = capacity
        self.get_session = get_session
        self.wake_prefill = wake_prefill
        self.on_request_end = on_request_end

        self.prealloc_queue: list[RequestState] = []#预分配等（prealloc 完成后）握手
        self.transfer_queue: list[RequestState] = []#已握手等 kv 过来
        self.waiting_queue: list[RequestState] = []#kv 已过来等 forward
        self.running: list[RequestState] = []#正在 forward

        self.free_kv_pages = capacity.decode_kv_pages
        self.free_req_slots = capacity.decode_req_slots#剩余并发空间
        self.free_metadata = capacity.decode_metadata_buffers#用于握手的缓存
        self._alloc_pages: dict[str, int] = {}

        self._busy = False
        self._boundary_pending = False

    # --- external entry ----------------------------------------------------
    def admit(self, req: RequestState, now: float) -> None:#接受新请求，设置握手完成事件
        """Router hands a request to this decode replica (request_arrival →
        DECODE_PREPARE / prealloc queue entry).  A handshake timer models
        ``receiver.init`` completing (KVPoll → WAITING_FOR_INPUT)."""
        # Same fail-fast as the prefill side: SGLang aborts an over-capacity
        # request when it is added to the prealloc queue (decode.py:634-654).
        need = self._decode_pages(req)
        if need > self.cap.decode_kv_pages:
            raise ValueError(
                f"{req.spec.request_id}: decode prealloc needs {need} KV pages "
                f"but the replica only has {self.cap.decode_kv_pages}; SGLang "
                f"would abort this request (decode.py:634)"
            )
        req.decode_replica_id = self.replica_id
        req.d_phase = DLinePhase.PREALLOC_QUEUE
        req.mark(Stage.DECODE_PREPARE, now)
        self.prealloc_queue.append(req)
        handshake_s = self.cost.bootstrap_handshake_s()
        self.loop.schedule_after(
            handshake_s,
            "decode_handshake_done",
            self._on_handshake_done,
            req.spec.request_id,
            {"req": req},
        )

    def _on_handshake_done(self, event: Event) -> None:#到握手事件，推进到处理边际事件
        now = self.loop.clock.now_s
        req: RequestState = event.payload["req"]
        session = self.get_session(req.spec.request_id)
        session.d_receiver_init()
        self.wake(now)

    def wake(self, now: float) -> None:#通用边际事件处理接口
        self.poll(now)
        if not self._busy and not self._boundary_pending and len(self.running) > 0:#decode 不卡资源，无需判断资源够不够
            self._boundary_pending = True
            self.loop.schedule_at(
                now, _DECODE_DONE, self._on_boundary, self.replica_id, {"batch": None}
            )

    # --- zero-cost bookkeeping --------------------------------------------
    def poll(self, now: float) -> None:
        self._pop_preallocated(now)
        self._pop_transferred(now)
        self._admit_prebuilt(now)

    def _pop_preallocated(self, now: float) -> None:#在1.握手完成；2.有空并发位；3.有空对接槽；4.kv 足够的情况下，1.进行 prealloc；2.通信，推动 P 侧推进；3.自己推进到等待 kv_transfer 阶段
        # prealloc_queue → transfer_queue: handshake done AND req slot +
        # metadata idx + KV pages available → _pre_alloc + send_metadata.
        still: list[RequestState] = []
        queue = self.prealloc_queue
        for idx, req in enumerate(queue):
            session = self.get_session(req.spec.request_id)
            if not session.d_handshake_done:
                # Still handshaking: skipped over, does *not* block the queue
                # (SGLang decode.py:950-951 `continue`).
                still.append(req)
                continue
            need_pages = self._decode_pages(req)#input_tokens + output_tokens + reserved
            if (
                self.free_req_slots <= 0
                or self.free_metadata <= 0
                or need_pages > self.free_kv_pages
            ):
                # Resource exhaustion blocks the queue *head* — every one of
                # SGLang's resource gates is a `break`, not a `continue`
                # (decode.py:953-957 slots/metadata, :1020-1024 KV budget), so a
                # later, smaller request must not jump ahead of a blocked one.
                still.extend(queue[idx:])
                break
            self.free_req_slots -= 1
            self.free_metadata -= 1
            self.free_kv_pages -= need_pages
            self._alloc_pages[req.spec.request_id] = need_pages
            session.d_send_metadata()  # flips prefill sender → WAITING_FOR_INPUT
            req.mark(Stage.DECODE_BOOTSTRAP, now)
            req.d_phase = DLinePhase.TRANSFER_QUEUE
            self.transfer_queue.append(req)
            self.wake_prefill(req)  # prefill bootstrap can now advance
        self.prealloc_queue = still

    def _pop_transferred(self, now: float) -> None:#kv_transfer 完成后，从等待 kv_transfer 阶段推进到等待前向传播阶段
        # transfer_queue → waiting_queue: KVPoll.Success (KV fully written).
        still: list[RequestState] = []
        for req in self.transfer_queue:
            session = self.get_session(req.spec.request_id)
            if session.d_transfer_succeeded:
                req.mark(Stage.DECODE_TRANSFERRED, now)
                req.d_phase = DLinePhase.WAITING_QUEUE
                self.waiting_queue.append(req)
            else:
                still.append(req)
        self.transfer_queue = still

    def _admit_prebuilt(self, now: float) -> None:#在有并发空位且有等待请求时，将队首请求首 token 回放，若还有其他 token 则进入并发槽
        # waiting_queue → PREBUILT fake batch (no prefill compute): the prefill-
        # sampled first token is replayed as the first decode bonus token.
        while self.waiting_queue and len(self.running) < self._batch_slot_limit():
            req = self.waiting_queue.pop(0)
            req.mark(Stage.DECODE_WAITING, now)
            req.mark(Stage.DECODE_FAKE_OUTPUT, now)
            req.d_phase = DLinePhase.PREBUILT
            req.emit_token(now)  # first token (from prefill), FIRST_TOKEN recorded
            if req.decode_remaining_tokens > 0:
                req.d_phase = DLinePhase.DECODING
                self.running.append(req)
            else:
                self._complete(req, now)  # quick finish (output_tokens == 1)

    def _batch_slot_limit(self) -> int:
        return min(self.cap.max_running_requests, self.cap.decode_req_slots)

    # --- decode forward loop ----------------------------------------------
    def _on_boundary(self, event: Event) -> None:#循环到终止
        now = self.loop.clock.now_s
        self._boundary_pending = False
        self._busy = False
        finished: list[RequestState] | None = event.payload.get("batch")

        self.poll(now)#1.零时耗记账，处理上一批
        current = list(self.running)  # selected before committing previous iter#2.选择下一批
        if finished is not None:
            self._commit_decode_iter(finished, now)#3.提交上一批
            self.poll(now)

        if current:#4.启动下一批
            dur = self.cost.decode_iter_s(len(current))
            self._busy = True
            self.loop.schedule_after(
                dur, _DECODE_DONE, self._on_boundary, self.replica_id, {"batch": current}
            )
        elif len(self.running) > 0:
            self._boundary_pending = True
            self.loop.schedule_at(
                now, _DECODE_DONE, self._on_boundary, self.replica_id, {"batch": None}
            )

    def _commit_decode_iter(self, batch: list[RequestState], now: float) -> None:#单次 decode 迭代处理
        for req in batch:
            if req.d_phase != DLinePhase.DECODING:
                continue  # already completed by an earlier iteration (wasted work)
            if req.decode_remaining_tokens <= 0:
                continue
            req.emit_token(now)
            if req.decode_remaining_tokens <= 0:
                self.running.remove(req)
                self._complete(req, now)

    # --- completion / accounting ------------------------------------------
    def _complete(self, req: RequestState, now: float) -> None:
        self._release(req)
        req.d_phase = DLinePhase.DONE
        req.mark(Stage.REQUEST_END, now)
        self.on_request_end(req, now)

    def _release(self, req: RequestState) -> None:
        pages = self._alloc_pages.pop(req.spec.request_id, 0)
        self.free_kv_pages += pages
        self.free_req_slots += 1
        self.free_metadata += 1

    def _decode_pages(self, req: RequestState) -> int:
        # DIVERGENCE (deliberate): SGLang reserves ``prompt + 512`` at prealloc
        # (``_pre_alloc_fill_len`` decode.py:619-632 + ``num_reserved_decode_tokens``)
        # and grows/retracts as decode proceeds.  We do not model retraction, so
        # we reserve the *whole* output budget up front instead — strictly
        # conservative, and it cannot deadlock mid-generation.
        tokens = req.spec.input_tokens + req.spec.output_tokens + self.cap.decode_reserved_tokens
        return self.cap.pages_for_tokens(tokens)
#finish review 2026.9.17