"""Workload generation: arrival process + request shape.

Two sources:
  * generated — fixed request shape + fixed-interval / seeded-Poisson arrivals;
  * trace — per-request input/output/cached tokens and arrivals loaded from an
    external jsonl (see ``trace_loader``), selected via ``WorkloadConfig.trace_path``.
"""

from __future__ import annotations

import random

from .request import RequestSpec
from .scenario import WorkloadConfig
from .trace_loader import load_trace_requests


def build_requests(cfg: WorkloadConfig) -> list[RequestSpec]:
    if cfg.trace_path is not None:
        return load_trace_requests(
            cfg.trace_path,
            limit=cfg.trace_limit,
            request_id_prefix=cfg.request_id_prefix,
        )

    rng = random.Random(cfg.seed)
    requests: list[RequestSpec] = []
    t = 0.0
    for i in range(cfg.num_requests):
        if cfg.arrival_process == "fixed_interval":
            interval = 1.0 / cfg.arrival_rate_rps
        else:  # poisson
            interval = rng.expovariate(cfg.arrival_rate_rps)
        if i == 0:
            arrival = 0.0
        else:
            t += interval
            arrival = t
        output = max(1, cfg.output_tokens)  # first token comes from prefill
        requests.append(
            RequestSpec(
                request_id=f"{cfg.request_id_prefix}-{i}",
                arrival_time_s=arrival,
                input_tokens=cfg.input_tokens,
                output_tokens=output,
                cached_tokens=cfg.cached_tokens,
            )
        )
    return requests
