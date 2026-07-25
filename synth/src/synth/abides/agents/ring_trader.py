"""RingTraderAgent — circular wash-trading ring inside ABIDES.

Model: k members are arranged in a directed cycle. On each rotation the
Coordinator schedules per-member actions that alternate buy/sell roles
around the ring (even-indexed members sell, odd-indexed members buy on
this rotation; on the next rotation roles swap). Inventory rotates around
the cycle while overall ownership rolls over with each rotation. This is
the classical circular-trading topology Indian market surveillance has
chased since the 1990s.

Why this is different from a clique: a clique pushes price; a ring rotates
inventory without net price impact. Surveillance detects them differently —
rings show up in the directed-graph structure (cycles) much more than in
return correlation. So Rung-1 (Pearson) doesn't catch them; Rung-4
(graph neural network with directional edge features) should.

Inherits from `abides_markets.agents.trading_agent.TradingAgent` (JPMC
fork, snake_case API). Same module-level import fallback as
``collusive_clique.py`` keeps the file host-importable.

Lessons baked in from CollusiveCliqueAgent's STEP 3 debugging:
  * ``wakeup`` returns None implicitly (non-None triggers the gym escape
    hatch in ``kernel.runner`` line 380).
  * ``get_wake_frequency`` exists so ``TradingAgent.receive_message`` can
    schedule the market-open wakeup.
  * Action times are session-relative ms; agent converts using
    ``coordinator.reference_time_ns`` (set by the config harness to
    ``mkt_open``).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


try:
    from abides_core import Message, NanosecondTime
    from abides_core.utils import str_to_ns
    from abides_markets.agents.trading_agent import TradingAgent
    from abides_markets.messages.query import QuerySpreadResponseMsg
    from abides_markets.orders import Side
    _ABIDES_AVAILABLE = True
except ImportError:        # pragma: no cover — host-side import path
    Message = object       # type: ignore
    NanosecondTime = int   # type: ignore
    QuerySpreadResponseMsg = object  # type: ignore

    class TradingAgent:    # type: ignore
        def __init__(self, *_, **__) -> None:
            raise RuntimeError(
                "abides_markets not installed; RingTraderAgent can only be "
                "instantiated inside the abides-synth container. The "
                "RingAction / RingCoordinator dataclasses remain usable."
            )

    class Side:            # type: ignore
        BID = "BID"
        ASK = "ASK"

    def str_to_ns(_: str) -> int:  # type: ignore
        return 0

    _ABIDES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coordination dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RingAction:
    """One member's instruction within a rotation cycle.

    The Coordinator pre-schedules a long list of these — typically several
    per member per rotation, alternating buy/sell roles around the cycle.
    Each member filters by ``member_idx == self.ring_idx`` on wakeup.
    """

    action_id: str
    fire_at_ms: int        # session-relative ms when this member fires
    window_ms: int         # within-window jitter the member samples
    member_idx: int        # 0..len(member_ids)-1 — which ring slot
    side: str              # "buy" | "sell"
    base_size: int = 100


@dataclass
class RingCoordinator:
    """Shared coordination state for a circular-trading ring.

    Same off-bus pattern as ``CliqueCoordinator`` — members read decisions
    from shared in-process state, NOT via ABIDES messages, so the trade
    tape reflects coordination without leaking it through the simulator's
    message stream.
    """

    ring_id: str
    member_ids: list[str] = field(default_factory=list)
    actions: list[RingAction] = field(default_factory=list)
    stagger_ms: int = 150          # per-member spacing within a rotation
    size_jitter: float = 0.20      # ±20% on order size
    jitter_ms: int = 80            # per-member temporal noise on top of window
    rotation_window_ms: int = 60_000  # nominal time per full circulation
    scenario_id: str = "scenario_ring"
    scenario_label: str = "ring_alpha"
    scenario_type: str = "circular_trading_ring"
    output_dir: Optional[Path] = None
    reference_time_ns: int = 0
    _labels_flushed: bool = field(default=False, init=False, repr=False)

    # -------------- topology helpers --------------
    def register_member(self, trader_id: str) -> Optional[int]:
        if trader_id in self.member_ids:
            return self.member_ids.index(trader_id)
        self.member_ids.append(trader_id)
        return len(self.member_ids) - 1

    def index_of(self, trader_id: str) -> Optional[int]:
        return self.member_ids.index(trader_id) if trader_id in self.member_ids else None

    def next_of(self, trader_id: str) -> Optional[str]:
        idx = self.index_of(trader_id)
        if idx is None or not self.member_ids:
            return None
        return self.member_ids[(idx + 1) % len(self.member_ids)]

    def prev_of(self, trader_id: str) -> Optional[str]:
        idx = self.index_of(trader_id)
        if idx is None or not self.member_ids:
            return None
        return self.member_ids[(idx - 1) % len(self.member_ids)]

    # -------------- action lookup --------------
    def action_at(self, member_idx: int, current_ms: int) -> Optional[RingAction]:
        """Return any unfired action for this member whose window is open now."""
        for a in self.actions:
            if a.member_idx != member_idx:
                continue
            if a.fire_at_ms <= current_ms <= a.fire_at_ms + a.window_ms:
                return a
        return None

    def next_action_after(self, member_idx: int, current_ms: int) -> Optional[RingAction]:
        upcoming = [a for a in self.actions
                    if a.member_idx == member_idx
                    and a.fire_at_ms + a.window_ms > current_ms]
        return min(upcoming, key=lambda a: a.fire_at_ms) if upcoming else None

    # -------------- label sidecar --------------
    def flush_labels(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        if self._labels_flushed:
            return None
        target_dir = output_dir or self.output_dir
        if target_dir is None:
            logger.warning(
                "RingCoordinator %s: no output_dir set; "
                "manipulator_labels.csv not written.", self.ring_id,
            )
            return None
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "manipulator_labels.csv"
        file_exists = path.is_file()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["trader_id", "scenario_id",
                                "scenario_label", "scenario_type"],
            )
            if not file_exists:
                writer.writeheader()
            for tid in self.member_ids:
                writer.writerow({
                    "trader_id":      tid,
                    "scenario_id":    self.scenario_id,
                    "scenario_label": self.scenario_label,
                    "scenario_type":  self.scenario_type,
                })
        self._labels_flushed = True
        return path


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class RingTraderAgent(TradingAgent):
    """A TradingAgent member of one RingCoordinator.

    Same lifecycle and state machine as CollusiveCliqueAgent, but the
    Coordinator's actions are per-member rather than per-clique-action.
    """

    @staticmethod
    def msa_trader_id(abides_agent_id: int) -> str:
        return f"trader_{int(abides_agent_id):05d}"

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        coordinator: RingCoordinator,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = True,
        wake_up_freq: "NanosecondTime" = None,  # type: ignore
    ) -> None:
        if not _ABIDES_AVAILABLE:
            super().__init__(id, name, type, random_state, starting_cash, log_orders)
        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.coordinator = coordinator
        self.wake_up_freq: NanosecondTime = (
            wake_up_freq if wake_up_freq is not None else str_to_ns("1s")
        )
        self.state: str = "AWAITING_WAKEUP"
        self._fired_action_ids: set[str] = set()
        self._pending_action: Optional[RingAction] = None
        # Register with the Coordinator and remember our slot in the cycle.
        self.ring_idx: int = coordinator.register_member(
            self.msa_trader_id(id)
        ) or 0

    def manipulator_label(self) -> tuple[str, str, str]:
        return (
            self.coordinator.scenario_id,
            self.coordinator.scenario_label,
            self.coordinator.scenario_type,
        )

    def get_wake_frequency(self) -> "NanosecondTime":
        return self.wake_up_freq

    # ABIDES lifecycle ---------------------------------------------------
    def kernel_starting(self, start_time: "NanosecondTime") -> None:
        super().kernel_starting(start_time)
        self.set_wakeup(start_time + str_to_ns("100ms"))

    def kernel_stopping(self) -> None:
        try:
            self.coordinator.flush_labels()
        except Exception as exc:  # noqa: BLE001
            logger.exception("RingCoordinator flush_labels failed: %s", exc)
        super().kernel_stopping()

    def wakeup(self, current_time: "NanosecondTime") -> None:
        # MUST return None — see CollusiveCliqueAgent for the trap.
        can_trade = super().wakeup(current_time)
        if not can_trade:
            self.set_wakeup(current_time + self.wake_up_freq)
            return

        current_ms = int((current_time - self.coordinator.reference_time_ns)
                         // 1_000_000)
        action = self.coordinator.action_at(self.ring_idx, current_ms)

        if action is not None and action.action_id not in self._fired_action_ids:
            self._pending_action = action
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        upcoming = self.coordinator.next_action_after(self.ring_idx, current_ms)
        if upcoming is None:
            self.set_wakeup(current_time + str_to_ns("60s"))
            return

        rng = self.random_state if self.random_state is not None else np.random
        window_offset_ms = int(rng.uniform(0, max(upcoming.window_ms, 1)))
        jitter_offset_ms = int(rng.uniform(-self.coordinator.jitter_ms,
                                            +self.coordinator.jitter_ms))
        fire_at_ms = max(current_ms,
                         upcoming.fire_at_ms + window_offset_ms + jitter_offset_ms)
        delta_ms = max(1, fire_at_ms - current_ms)
        self.set_wakeup(current_time + delta_ms * 1_000_000)

    def receive_message(
        self,
        current_time: "NanosecondTime",
        sender_id: int,
        message: "Message",
    ) -> None:
        super().receive_message(current_time, sender_id, message)
        if (
            self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            self._place_ring_order(bid, ask)
            self.state = "AWAITING_WAKEUP"
            self.set_wakeup(current_time + str_to_ns("500ms"))

    # ------------------------------------------------------------------
    def _place_ring_order(self, bid: int, ask: int) -> None:
        action = self._pending_action
        if action is None:
            return
        if not bid or not ask:
            logger.debug("Ring %s member %d: no spread (bid=%s ask=%s)",
                         self.coordinator.ring_id, self.id, bid, ask)
            return

        rng = self.random_state if self.random_state is not None else np.random
        size_mult = 1.0 + float(rng.uniform(-self.coordinator.size_jitter,
                                             +self.coordinator.size_jitter))
        quantity = max(1, int(round(action.base_size * size_mult)))

        if action.side == "buy":
            self.place_limit_order(self.symbol, quantity, Side.BID, ask)
        elif action.side == "sell":
            self.place_limit_order(self.symbol, quantity, Side.ASK, bid)
        else:
            logger.warning("Ring %s action %s has unknown side %r; skipped.",
                           self.coordinator.ring_id, action.action_id, action.side)
            return

        self._fired_action_ids.add(action.action_id)
        self._pending_action = None


# ---------------------------------------------------------------------------
# Coordinator factory
# ---------------------------------------------------------------------------
def schedule_rotations(
    coordinator: RingCoordinator,
    *,
    n_members: int,
    n_rotations: int,
    session_ms: int,
    base_size: int = 150,
    window_ms: int = 300,
) -> None:
    """Populate ``coordinator.actions`` with ``n_rotations`` ring cycles.

    On rotation r, member k acts as SELLER if (k + r) is even, otherwise
    BUYER. Members stagger by ``coordinator.stagger_ms`` within a rotation.
    Across rotations, the global spacing is ``rotation_step_ms = session_ms / n_rotations``.
    Each per-member ``RingAction`` has its own ``window_ms`` for further jitter.
    """
    if n_members <= 0 or n_rotations <= 0:
        return
    rotation_step_ms = max(session_ms // (n_rotations + 1), 60_000)
    for r in range(n_rotations):
        rotation_start_ms = (r + 1) * rotation_step_ms
        for k in range(n_members):
            fire_at = rotation_start_ms + k * coordinator.stagger_ms
            side = "sell" if (k + r) % 2 == 0 else "buy"
            coordinator.actions.append(RingAction(
                action_id=f"r{r}_m{k}",
                fire_at_ms=fire_at,
                window_ms=window_ms,
                member_idx=k,
                side=side,
                base_size=base_size,
            ))
