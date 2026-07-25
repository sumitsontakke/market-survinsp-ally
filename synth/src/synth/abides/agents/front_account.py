"""FrontAccountAgent — passive layer that obscures a ringleader's topology.

Model: a non-decision-making account that subscribes to a ringleader's
Coordinator (clique or ring). On each Coordinator action it submits an
order with a stochastic 200-2000ms delay, then after a brief hold offloads
the inventory back toward the ringleader via the open book — creating a
two-hop edge pattern (ringleader ↔ front ↔ ringleader) instead of a
direct ringleader↔ringleader edge.

This is the trader that, in real markets, lets a small inner ring appear
to be many unrelated parties when it's really a handful of ringleaders
behind layered accounts. Crucial for the dissertation: Rung-3's engineered
edge features struggle here because the manipulative edge is two hops
away; Rung-4's graph neural net should "see through" the layering by
aggregating neighbourhood structure.

Same JPMC-fork API lessons as collusive_clique.py — wakeup returns None,
get_wake_frequency exists, label sidecar via the ringleader Coordinator's
flush_labels (the front does NOT own its own labels; it shares the
ringleader's scenario tag).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


try:
    from abides_core import Message, NanosecondTime
    from abides_core.utils import str_to_ns
    from abides_markets.agents.trading_agent import TradingAgent
    from abides_markets.messages.query import QuerySpreadResponseMsg
    from abides_markets.orders import Side
    _ABIDES_AVAILABLE = True
except ImportError:        # pragma: no cover
    Message = object       # type: ignore
    NanosecondTime = int   # type: ignore
    QuerySpreadResponseMsg = object  # type: ignore

    class TradingAgent:    # type: ignore
        def __init__(self, *_, **__) -> None:
            raise RuntimeError(
                "abides_markets not installed; FrontAccountAgent can only be "
                "instantiated inside the abides-synth container."
            )

    class Side:            # type: ignore
        BID = "BID"
        ASK = "ASK"

    def str_to_ns(_: str) -> int:  # type: ignore
        return 0

    _ABIDES_AVAILABLE = False


# Import the Coordinator types lazily so this module doesn't force a
# circular import on package load. We type-hint with `Union` so either is
# acceptable; the duck-typed contract is just .action_at/.next_action_after
# and .flush_labels.
def _resolve_coordinator_types():
    from .collusive_clique import CliqueCoordinator
    from .ring_trader import RingCoordinator
    return CliqueCoordinator, RingCoordinator


# ---------------------------------------------------------------------------
# Front instruction
# ---------------------------------------------------------------------------
@dataclass
class _PendingOffload:
    """In-flight inventory the front needs to unload back toward the ringleader."""
    side: str               # "buy" or "sell" — direction of the OFFLOAD
    base_size: int
    earliest_fire_ms: int   # session-relative


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class FrontAccountAgent(TradingAgent):
    """Layered front for a clique or ring ringleader.

    Subscribes to ``ringleader_coordinator``. On wakeup, scans the
    Coordinator for any ringleader action whose window is open OR
    recently closed; if found, schedules a delayed mirror of that order
    (same side, similar size) at ``delay_ms`` after the action's
    fire_at_ms. After firing the mirror, schedules an opposite-side
    offload some ``offload_after_ms`` later to roll the inventory back.

    The two-hop pattern emerges in the trade tape: ringleader → front
    (first leg) followed by front → ringleader (offload). Net P&L on the
    front is small; volume profile is asynchronous to any single
    ringleader.
    """

    @staticmethod
    def msa_trader_id(abides_agent_id: int) -> str:
        return f"trader_{int(abides_agent_id):05d}"

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        ringleader_coordinator,                  # CliqueCoordinator | RingCoordinator
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = True,
        wake_up_freq: "NanosecondTime" = None,  # type: ignore
        delay_ms_low: int = 200,
        delay_ms_high: int = 2000,
        offload_after_ms_low: int = 800,
        offload_after_ms_high: int = 3000,
        size_jitter: float = 0.30,
    ) -> None:
        if not _ABIDES_AVAILABLE:
            super().__init__(id, name, type, random_state, starting_cash, log_orders)
        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.ringleader_coordinator = ringleader_coordinator
        self.wake_up_freq: NanosecondTime = (
            wake_up_freq if wake_up_freq is not None else str_to_ns("2s")
        )
        self.state: str = "AWAITING_WAKEUP"
        self.delay_ms_low = int(delay_ms_low)
        self.delay_ms_high = int(delay_ms_high)
        self.offload_after_ms_low = int(offload_after_ms_low)
        self.offload_after_ms_high = int(offload_after_ms_high)
        self.size_jitter = float(size_jitter)
        # We track which ringleader action_ids we've already mirrored so we
        # don't double-mirror the same coordinator instruction.
        self._mirrored_action_ids: set[str] = set()
        # Pending offloads (1 per mirrored action, fired after a delay).
        self._pending_offloads: list[_PendingOffload] = []
        # The order we're about to place when AWAITING_SPREAD resolves.
        self._pending_side: Optional[str] = None
        self._pending_base_size: int = 0
        # Register with the ringleader's coordinator so we share the label.
        if hasattr(ringleader_coordinator, "register_member"):
            ringleader_coordinator.register_member(self.msa_trader_id(id))

    def manipulator_label(self) -> tuple[str, str, str]:
        c = self.ringleader_coordinator
        return (c.scenario_id, c.scenario_label, c.scenario_type)

    def get_wake_frequency(self) -> "NanosecondTime":
        return self.wake_up_freq

    # ABIDES lifecycle ---------------------------------------------------
    def kernel_starting(self, start_time: "NanosecondTime") -> None:
        super().kernel_starting(start_time)
        self.set_wakeup(start_time + str_to_ns("100ms"))

    def kernel_stopping(self) -> None:
        # The ringleader coordinator owns the label sidecar — calling
        # flush_labels here too is idempotent and harmless.
        try:
            self.ringleader_coordinator.flush_labels()
        except Exception as exc:  # noqa: BLE001
            logger.exception("ringleader flush_labels failed: %s", exc)
        super().kernel_stopping()

    def wakeup(self, current_time: "NanosecondTime") -> None:
        # MUST return None — see CollusiveCliqueAgent for the trap.
        can_trade = super().wakeup(current_time)
        if not can_trade:
            self.set_wakeup(current_time + self.wake_up_freq)
            return

        current_ms = int((current_time - self.ringleader_coordinator.reference_time_ns)
                         // 1_000_000)

        # 1. Drain any pending offload whose earliest_fire_ms has passed.
        for offload in list(self._pending_offloads):
            if current_ms >= offload.earliest_fire_ms:
                self._pending_offloads.remove(offload)
                self._pending_side = offload.side
                self._pending_base_size = offload.base_size
                self.get_current_spread(self.symbol)
                self.state = "AWAITING_SPREAD"
                return  # fire one order per wakeup

        # 2. Look for any new ringleader action we haven't mirrored yet.
        new_action = self._find_unmirrored_action(current_ms)
        if new_action is not None:
            action_id, side, base_size = new_action
            self._mirrored_action_ids.add(action_id)
            # Schedule the mirror to fire after a stochastic delay rather
            # than firing immediately — this is what masks the temporal
            # correlation to the ringleader's wakeup.
            rng = self.random_state if self.random_state is not None else np.random
            delay_ms = int(rng.uniform(self.delay_ms_low, self.delay_ms_high))
            self.set_wakeup(current_time + delay_ms * 1_000_000)
            # Queue this side/size as a pending "offload" with reversed side
            # to fire AFTER the mirror. Trick: we use the same _pending_offloads
            # list, but mark the first one as the mirror itself (side as-is)
            # and queue a true offload (side reversed) for later.
            self._pending_offloads.append(_PendingOffload(
                side=side,
                base_size=base_size,
                earliest_fire_ms=current_ms + delay_ms,
            ))
            offload_delay_ms = delay_ms + int(rng.uniform(
                self.offload_after_ms_low, self.offload_after_ms_high,
            ))
            self._pending_offloads.append(_PendingOffload(
                side=("sell" if side == "buy" else "buy"),
                base_size=base_size,
                earliest_fire_ms=current_ms + offload_delay_ms,
            ))
            return

        # 3. Nothing to do — poll again later.
        self.set_wakeup(current_time + self.wake_up_freq)

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
            self._place_front_order(bid, ask)
            self.state = "AWAITING_WAKEUP"
            self.set_wakeup(current_time + str_to_ns("500ms"))

    # ------------------------------------------------------------------
    def _find_unmirrored_action(
        self, current_ms: int,
    ) -> Optional[tuple[str, str, int]]:
        """Return (action_id, side, base_size) for the most recent ringleader
        action we haven't yet mirrored. Works against both Clique and Ring
        coordinators (they expose ``.actions`` with compatible attributes).
        """
        c = self.ringleader_coordinator
        actions = getattr(c, "actions", None)
        if not actions:
            return None
        # Look back up to 5s past the close of each action's window.
        for a in actions:
            if a.action_id in self._mirrored_action_ids:
                continue
            window_end_ms = a.fire_at_ms + getattr(a, "window_ms", 0)
            if a.fire_at_ms <= current_ms <= window_end_ms + 5_000:
                # RingAction has member_idx + side; CliqueAction has side.
                return (a.action_id, a.side, a.base_size)
        return None

    def _place_front_order(self, bid: int, ask: int) -> None:
        side = self._pending_side
        if side is None:
            return
        if not bid or not ask:
            logger.debug("Front %d: no spread (bid=%s ask=%s)", self.id, bid, ask)
            self._pending_side = None
            return

        rng = self.random_state if self.random_state is not None else np.random
        size_mult = 1.0 + float(rng.uniform(-self.size_jitter, +self.size_jitter))
        quantity = max(1, int(round(self._pending_base_size * size_mult)))

        if side == "buy":
            self.place_limit_order(self.symbol, quantity, Side.BID, ask)
        elif side == "sell":
            self.place_limit_order(self.symbol, quantity, Side.ASK, bid)
        self._pending_side = None
        self._pending_base_size = 0
