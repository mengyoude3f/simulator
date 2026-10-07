"""KVPoll state machine and the per-request P/D transfer session.

This is the cross-node coupling described in ``SGLANG_SOURCE_FINDINGS.md`` §二
(KVPoll) and §三 (handshake causality).  One ``KVTransferSession`` per request
(the ``bootstrap_room``) is shared by the prefill sender and the decode
receiver.  States advance monotonically via ``max`` (except ``FAILED`` which is
forced), exactly like SGLang's per-room ``request_status``.

Only the default *strict* (non-optimistic) path is modelled: the prefill line
cannot leave the bootstrap queue until the decode line has sent its destination
metadata (``p_state == WAITING_FOR_INPUT``).

KV is pushed **per chunk**, mirroring SGLang's ``send_kv_chunk`` /
``req.start_send_idx`` cursor (``prefill.py:1051-1180``): each finished prefill
chunk streams its own slice of the prompt KV while later chunks are still being
computed.  The session only reaches ``SUCCESS`` once the *final* chunk has been
submitted and every outstanding chunk transfer has landed.
"""

from __future__ import annotations

from enum import IntEnum


class KVPoll(IntEnum):#编码规则同 SGLang
    FAILED = 0
    BOOTSTRAPPING = 1
    WAITING_FOR_INPUT = 2
    TRANSFERRING = 3
    SUCCESS = 4


class KVTransferSession:#SGLang 的 bootstrap_room，每个 request 各一个
    """Shared handshake + transfer state for one request across P and D."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        # Sender (prefill) and receiver (decode) each track their own poll state.
        self.p_state = KVPoll.BOOTSTRAPPING
        self.d_state = KVPoll.BOOTSTRAPPING
        # Handshake / rendezvous flags.
        self.d_metadata_sent = False
        self.p_forward_done = False
        self.transfer_started = False
        # Per-chunk transfer accounting: SUCCESS requires the final chunk to
        # have been submitted AND every outstanding chunk to have landed.
        self.chunks_sent = 0#已发送几个 chunk
        self.chunks_done = 0#已落地几个 chunk
        self.last_chunk_sent = False#末块是否已发送
        # Payload accounting (cumulative over all chunks).
        self.transfer_bytes = 0

    # --- monotonic helpers -------------------------------------------------
    @staticmethod
    def _advance(current: KVPoll, target: KVPoll) -> KVPoll:#failed 粘滞：failed 则返回 failed；否则只向前不后退；
        if target == KVPoll.FAILED:
            return KVPoll.FAILED
        if current == KVPoll.FAILED:
            return KVPoll.FAILED
        return KVPoll(max(int(current), int(target)))

    # --- decode-side transitions ------------------------------------------
    def d_receiver_init(self) -> None:#D 侧和 bootstrap server 握手完成
        """Decode handshake with the bootstrap server completes."""
        self.d_state = self._advance(self.d_state, KVPoll.WAITING_FOR_INPUT)

    def d_send_metadata(self) -> None:#D 预分配好，把目标地址告诉 P
        """Decode preallocated slots and told prefill *where* to write KV.

        This is the necessary+sufficient trigger for the prefill line to leave
        its bootstrap queue (flips the sender to WAITING_FOR_INPUT).
        """
        if self.d_state < KVPoll.WAITING_FOR_INPUT:
            raise ValueError("cannot send metadata before decode handshake")
        self.d_metadata_sent = True
        self.p_state = self._advance(self.p_state, KVPoll.WAITING_FOR_INPUT)

    # --- prefill-side transitions -----------------------------------------
    def p_mark_forward_done(self) -> None:#最后一个 chunk 的 forward 完成（首 token 在此采样）
        self.p_forward_done = True

    def p_can_send(self) -> bool:
        """A prefill *chunk* may be pushed as soon as decode's destination
        metadata has arrived.

        Note this does NOT require the whole prefill to be finished: SGLang
        streams each finished chunk while later chunks are still computing
        (``prefill.py:755`` / ``:1010``).  The per-chunk forward result is
        implicitly ready because the send is issued from the commit path.
        """
        return self.d_metadata_sent

    def p_start_chunk_send(self, transfer_bytes: int, is_last: bool) -> None:#发一个 chunk 的 kv #某个 chunk 的发送
        if not self.p_can_send():
            raise ValueError("KV send preconditions not met")
        if self.last_chunk_sent:
            raise ValueError("final chunk was already sent")
        self.transfer_started = True
        self.chunks_sent += 1
        self.transfer_bytes += transfer_bytes
        if is_last:
            self.last_chunk_sent = True
        self.p_state = self._advance(self.p_state, KVPoll.TRANSFERRING)
        self.d_state = self._advance(self.d_state, KVPoll.TRANSFERRING)

    def complete_chunk_transfer(self) -> bool:#所有 chunk 是否发送并落地
        """One chunk's physical transfer landed.

        Returns True iff this completes the request (final chunk submitted and
        no outstanding chunks left) — only then do both sides reach SUCCESS.
        """
        if not self.transfer_started:
            raise ValueError("transfer never started")
        self.chunks_done += 1
        if self.last_chunk_sent and self.chunks_done == self.chunks_sent:
            self.p_state = self._advance(self.p_state, KVPoll.SUCCESS)
            self.d_state = self._advance(self.d_state, KVPoll.SUCCESS)
            return True
        return False

    # --- gates read by the nodes ------------------------------------------
    @property
    def p_bootstrap_done(self) -> bool:
        return self.p_state >= KVPoll.WAITING_FOR_INPUT

    @property
    def d_handshake_done(self) -> bool:
        return self.d_state >= KVPoll.WAITING_FOR_INPUT

    @property
    def p_transfer_succeeded(self) -> bool:
        return self.p_state == KVPoll.SUCCESS

    @property
    def d_transfer_succeeded(self) -> bool:
        return self.d_state == KVPoll.SUCCESS

#finish review 2026.9.16
#finish review version-"kv_transfer 由逐 request 改为逐 chunk" 2026.9.20