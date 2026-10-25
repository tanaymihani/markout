"""Clocks for the paper contest. The bot and the paper exchange take any `() -> datetime`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class SimClock:
    """Simulated time that runs `speed` times faster than real time."""

    def __init__(self, speed: float = 1.0, start: datetime | None = None):
        self.speed = speed
        self.t0 = datetime.now(UTC)
        self.start = start or self.t0

    def __call__(self) -> datetime:
        return self.start + (datetime.now(UTC) - self.t0) * self.speed


class ManualClock:
    """Time that moves only when told to (the offline demo, the tests, the hosted desk)."""

    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, dt: timedelta) -> None:
        self.t += dt
