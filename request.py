"""Request specs, per-line phase state machines, and stage/event names.

Stage names mirror the SGLang ``SchedulerReqTimeStats`` lifecycle documented in
``SGLANG_SOURCE_FINDINGS.md`` §四.  The two request "lines" (P line on the
prefill node, D line on the decode node) each advance through their own phase
enum; both are driven from a single ``RequestState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):#对应 SGLang 的 RequestStage，请求的生命周期阶段
    """Lifecycle stage markers (== SGLang RequestStage / setter names)."""

    REQUEST_ARRIVAL = "request_arrival"

    # Prefill node (KV sender)
    PREFILL_PREPARE = "prefill_prepare"  # → bootstrap queue
    PREFILL_BOOTSTRAP = "prefill_bootstrap"  # bootstrap done → waiting queue
    PREFILL_WAITING = "prefill_waiting"  # scheduled into extend batch
    PREFILL_FORWARD = "prefill_forward"  # last chunk forward done
    KV_CHUNK_SEND = "kv_chunk_send"  # a chunk's KV started flying (first = earliest) #每一块 chunk 发送（从 P transfer 到 D）时打点
    KV_TRANSFER_START = "kv_transfer_start"#最后一个 chunk 发送时打点（SGLang 逻辑）；但状态机在第一块 chunk 发送时就已进入 TRANSFERRING 状态
    PREFILL_TRANSFER_KV_CACHE = "prefill_transfer_kv_cache"  # transfer done
    PREFILL_COMPLETE = "prefill_complete"

    # Decode node (KV receiver)
    DECODE_PREPARE = "decode_prepare"  # → prealloc queue
    DECODE_BOOTSTRAP = "decode_bootstrap"  # prealloc + send_metadata → transfer queue
    DECODE_TRANSFERRED = "decode_transferred"  # KV Success → waiting queue
    DECODE_WAITING = "decode_waiting"  # scheduled into prebuilt batch
    DECODE_FAKE_OUTPUT = "decode_fake_output"  # prebuilt fake forward, first token
    FIRST_TOKEN = "first_token"
    TOKEN_EMITTED = "token_emitted"
    REQUEST_END = "request_end"


class PLinePhase(str, Enum):
    ARRIVAL = "p_arrival"
    BOOTSTRAP_QUEUE = "p_bootstrap_queue"
    WAITING_QUEUE = "p_waiting_queue"
    FORWARDING = "p_forwarding"
    TRANSFERRING = "p_transferring"
    DONE = "p_done"


class DLinePhase(str, Enum):
    ARRIVAL = "d_arrival"
    PREALLOC_QUEUE = "d_prealloc_queue"
    TRANSFER_QUEUE = "d_transfer_queue"
    WAITING_QUEUE = "d_waiting_queue"
    PREBUILT = "d_prebuilt"
    DECODING = "d_decoding"
    DONE = "d_done"


@dataclass(frozen=True)
class RequestSpec:#不可变请求描述
    """Immutable workload description of one request."""

    request_id: str
    arrival_time_s: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0

    def __post_init__(self) -> None:#校验：首 token 采样的存在，output_tokens >= 1，下游 emit_tokens 不崩
        if self.input_tokens <= 0:
            raise ValueError(f"{self.request_id}: input_tokens must be > 0")
        if self.output_tokens < 1:
            # The first output token is always sampled by the prefill node, so
            # every modelled request produces at least one token.
            raise ValueError(f"{self.request_id}: output_tokens must be >= 1")
        if not 0 <= self.cached_tokens < self.input_tokens:
            # SGLang caps a radix-cache hit at ``input_len - 1``
            # (``_compute_max_prefix_len``, schedule_batch.py:1296-1301: "the
            # matched length is at most 1 less than the input length to enable
            # logprob computation"), so prefill always recomputes >= 1 token and
            # can sample the first output token.  A fully-cached prompt is
            # therefore not representable.
            raise ValueError(
                f"{self.request_id}: need 0 <= cached_tokens < input_tokens "
                f"(got {self.cached_tokens} / {self.input_tokens}); SGLang caps "
                f"a prefix hit at input_len - 1"
            )

    @property
    def prefill_work_tokens(self) -> int:#uncached_tokens
        """Tokens the prefill node actually computes (prompt minus cache hit)."""
        return self.input_tokens - self.cached_tokens


@dataclass
class RequestState:#请求的运行状态
    """Mutable runtime state carrying both lines of one request."""

    spec: RequestSpec
    p_phase: PLinePhase = PLinePhase.ARRIVAL
    d_phase: DLinePhase = DLinePhase.ARRIVAL

    prefill_replica_id: str = ""
    decode_replica_id: str = ""

    # Prefill chunking progress.
    prefill_done_tokens: int = 0
    # KV send cursor: absolute prompt position already pushed to decode
    # (== SGLang's ``req.start_send_idx``).  Chunks stream out incrementally.
    kv_send_idx: int = 0#发送到第几个 token（顺序：cached_tokens -> input_tokens）
    # Decode progress (first token comes from prefill sampling, replayed on D).
    generated_tokens: int = 0

    # Timestamps: first occurrence of each stage, plus full ordered history.
    stage_times_s: dict[Stage, float] = field(default_factory=dict)#每个 stage 首次时间
    history: list[tuple[Stage, float]] = field(default_factory=list)
    token_times_s: list[float] = field(default_factory=list)

    def mark(self, stage: Stage, time_s: float) -> None:
        self.stage_times_s.setdefault(stage, time_s)#如 dict 已有 stage ：直接返回 dict[stage]；否则，dict[stage] = time_s，并返回time_s
        self.history.append((stage, time_s))

    @property
    def prefill_remaining_tokens(self) -> int:
        return max(self.spec.prefill_work_tokens - self.prefill_done_tokens, 0)

    @property
    def prefill_finished(self) -> bool:
        return self.prefill_remaining_tokens == 0

    @property
    def decode_remaining_tokens(self) -> int:
        return max(self.spec.output_tokens - self.generated_tokens, 0)

    def emit_token(self, time_s: float) -> None:
        if self.decode_remaining_tokens <= 0:
            raise ValueError(f"{self.spec.request_id}: no remaining output token")
        self.generated_tokens += 1
        self.token_times_s.append(time_s)
        self.mark(Stage.TOKEN_EMITTED, time_s)
        if self.generated_tokens == 1:
            self.mark(Stage.FIRST_TOKEN, time_s)
#finish review 2026.9.16
#finish review version-"kv_transfer 由逐 request 改为逐 chunk" 2026.9.20