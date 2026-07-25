from __future__ import annotations

from datetime import datetime, timedelta

from synth.generator.domain.entities import TradingSession


class SessionClock:
    def __init__(self, session: TradingSession, steps: int) -> None:
        self.session = session
        self.steps = steps
        self.start = datetime.fromisoformat(f"{session.trade_date}T{session.open_time}")
        self.end = datetime.fromisoformat(f"{session.trade_date}T{session.close_time}")
        self.step_size = (self.end - self.start) / max(steps, 1)

    def iter_timestamps(self) -> list[datetime]:
        return [self.start + self.step_size * offset for offset in range(self.steps)]

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp <= self.end
