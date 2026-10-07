#从外部 jsonl trace 读取每请求的 input/output/cached tokens 与到达时间
"""Load request specs from an external jsonl trace.

The reference format (one JSON object per line) nests token counts under a
``metadata`` object and carries an ISO-8601 arrival timestamp, e.g.::

    {"request_arrival_timestamp": "2026-07-01T00:00:11.210319270Z",
     "request_id": "…",
     "metadata": {"input_token_sequence_length": 59790,
                  "output_token_sequence_length": 398,
                  "cached_tokens": 58944, …}}

Only these fields are consumed; everything else is ignored, and each field has
a few fallback names so partial / differently-named traces still load.  Arrival
timestamps are rebased so the earliest request starts at t=0.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

from .request import RequestSpec

_INPUT_KEYS = ("input_token_sequence_length", "input_len", "input_tokens")
_OUTPUT_KEYS = ("output_token_sequence_length", "output_len", "output_tokens")
_CACHED_KEYS = ("cached_tokens", "cached_len")
_ARRIVAL_KEYS = ("request_arrival_timestamp", "arrival_time_s", "arrival")
_ID_KEYS = ("request_id", "id")


def _pick(record: dict, meta: dict, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
        if k in record and record[k] is not None:
            return record[k]
    return None


def _parse_arrival(value: Any) -> float:
    """Return absolute seconds from a numeric or ISO-8601 (nanosecond) value."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1]
        if "." in s:
            main, frac = s.split(".", 1)
            frac_s = float("0." + frac)  # tolerates nanosecond (9-digit) precision
        else:
            main, frac_s = s, 0.0
        dt = datetime.strptime(main, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp() + frac_s
    raise ValueError(f"unrecognized arrival timestamp: {value!r}")


def load_trace_requests(
    path: str,
    *,
    limit: int | None = None,
    request_id_prefix: str = "req",
    metadata_key: str = "metadata",
) -> list[RequestSpec]:
    raw: list[tuple[str, Any, int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(raw) >= limit:
                break
            record = json.loads(line)
            meta = record.get(metadata_key)
            meta = meta if isinstance(meta, dict) else {}

            inp = _pick(record, meta, _INPUT_KEYS)
            out = _pick(record, meta, _OUTPUT_KEYS)
            if inp is None or out is None:
                raise ValueError(f"{path}:{lineno}: missing input/output token length")
            cached = _pick(record, meta, _CACHED_KEYS)
            cached = 0 if cached is None else int(cached)
            arrival = _pick(record, meta, _ARRIVAL_KEYS)
            rid = _pick(record, meta, _ID_KEYS) or f"{request_id_prefix}-{len(raw)}"
            raw.append((str(rid), arrival, int(inp), int(out), cached))

    if not raw:
        raise ValueError(f"{path}: no records found")

    parsed = [
        (rid, (_parse_arrival(a) if a is not None else None), inp, out, cached)
        for rid, a, inp, out, cached in raw
    ]
    known = [a for _, a, *_ in parsed if a is not None]
    base = min(known) if known else 0.0

    requests: list[RequestSpec] = []
    seen_ids: dict[str, int] = {}
    for rid, arrival, inp, out, cached in parsed:
        arrival_rel = max(0.0, arrival - base) if arrival is not None else 0.0
        # A zero-output trace row still yields the prefill-sampled first token,
        # so treat exactly 0 as 1; other invalid values fall through to
        # RequestSpec validation (fail fast).
        if out == 0:
            out = 1
        # Production traces can repeat a request_id across rows (retries, or a
        # session logged more than once).  The simulator keys per-request state
        # and the KV session by id, so duplicates must be disambiguated or two
        # distinct requests would silently share one transfer session.
        if rid in seen_ids:
            seen_ids[rid] += 1
            rid = f"{rid}#dup{seen_ids[rid]}"
        else:
            seen_ids[rid] = 0
        requests.append(
            RequestSpec(
                request_id=rid,
                arrival_time_s=arrival_rel,
                input_tokens=inp,
                output_tokens=out,
                # 0 <= cached <= input - 1: SGLang caps a prefix hit at
                # ``input_len - 1`` so prefill always recomputes >= 1 token
                # (schedule_batch.py:1296-1301).
                cached_tokens=max(0, min(cached, inp - 1)),
            )
        )
    return requests
