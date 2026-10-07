#离散事件仿真引擎
"""Discrete-event simulation core: monotonic clock + stable priority event queue.

The engine is deliberately policy-free.  It only orders callbacks in time.  All
PD-disaggregation logic lives in the prefill / decode nodes that register
callbacks here.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable

# Two events at the same timestamp are ordered by insertion sequence, so the
# simulation is fully deterministic regardless of heap tie-breaking.
_EPSILON_S = 1e-12


@dataclass(order=True)
class Event:
    time_s: float
    sequence: int
    kind: str = field(compare=False)
    request_id: str = field(default="", compare=False)
    callback: Callable[["Event"], None] | None = field(default=None, compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class Clock:
    """A monotonic, never-decreasing simulation clock."""

    def __init__(self) -> None:
        self._now_s = 0.0

    @property
    def now_s(self) -> float:
        return self._now_s

    def advance_to(self, time_s: float) -> None:
        if time_s + _EPSILON_S < self._now_s:
            raise ValueError(
                f"clock cannot go backwards: now={self._now_s} target={time_s}"
            )
        self._now_s = max(self._now_s, time_s)


class EventQueue:
    """Stable min-heap of pending events keyed by (time, insertion order)."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._sequence = 0

    def __len__(self) -> int:
        return len(self._heap)

    def schedule_at(
        self,
        time_s: float,
        kind: str,
        callback: Callable[[Event], None],
        request_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(
            time_s=time_s,
            sequence=self._sequence,
            kind=kind,
            request_id=request_id,
            callback=callback,
            payload=payload or {},
        )
        self._sequence += 1
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> Event:
        return heapq.heappop(self._heap)


class DiscreteEventLoop:#把 clock 和 EventQueue 绑在一起
    """Owns the clock and event queue and runs events to completion."""

    def __init__(self) -> None:
        self.clock = Clock()
        self.queue = EventQueue()

    def schedule_after(#相对现在延迟某段时间安排事件
        self,
        delay_s: float,
        kind: str,
        callback: Callable[[Event], None],
        request_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> Event:
        if delay_s < -_EPSILON_S:
            raise ValueError(f"delay must be non-negative, got {delay_s}")
        return self.queue.schedule_at(
            self.clock.now_s + max(0.0, delay_s), kind, callback, request_id, payload
        )

    def schedule_at(#在某绝对时刻安排事件
        self,
        time_s: float,
        kind: str,
        callback: Callable[[Event], None],
        request_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> Event:
        return self.queue.schedule_at(time_s, kind, callback, request_id, payload)

    def run(self, max_events: int | None = None) -> int:#不断跑，直到完事/处理事件量达到最大值
        """Drain the queue.  Returns the number of events processed."""
        processed = 0
        while len(self.queue) > 0:
            if max_events is not None and processed >= max_events:
                raise RuntimeError(
                    f"event budget of {max_events} exhausted; possible livelock"
                )
            event = self.queue.pop()
            self.clock.advance_to(event.time_s)
            if event.callback is not None:
                event.callback(event)
            processed += 1
        return processed
#finish review 2026.9.16