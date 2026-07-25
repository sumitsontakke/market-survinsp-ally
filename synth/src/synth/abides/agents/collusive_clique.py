"""CollusiveCliqueAgent — coordinated price-pushing clique inside ABIDES.

Model: k members of a clique coordinate (via shared in-process state, NOT
exchange messages) to push price in a chosen direction. Each member submits
same-side orders within a short jittered window; the price impact compounds.

Defensibility note for the dissertation: this models the trade pattern of
collusion, not the communication channel. Real colluders coordinate
off-exchange (phone, chat); the surveillance signal lives in the resulting
trade tape, which is what we detect.

Why a Coordinator class: ABIDES agents communicate via the kernel's message
bus. Real colluders don't. CliqueCoordinator holds shared decisions
(direction, firing window, target side) that all clique members read off-bus.
Each agent samples within the action's window and adds jitter so the
synchrony is not trivially detectable by Rung-1 (Pearson correlation).

Inherits from `abides_markets.agents.trading_agent.TradingAgent` (JPMC fork,
snake_case API). The module-level import is wrapped so the file remains
importable on hosts without ABIDES installed — the agent class falls back
to a stub that raises a clear error when instantiated, while the dataclasses
(CliqueAction, CliqueCoordinator) and the label-flush helper remain usable.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional ABIDES imports — present in the abides-synth container, absent on
# the host. We resolve at module-load time but degrade gracefully so the
# adapter test suite (which never touches the agent class) still imports
# this file without crashing.
# ---------------------------------------------------------------------------
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

    class TradingAgent:    # type: ignore — minimal stub for host-side imports
        def __init__(self, *_, **__) -> None:
            raise RuntimeError(
                "abides_markets not installed; CollusiveCliqueAgent can only "
                "be instantiated inside the abides-synth container. The "
                "CliqueAction / CliqueCoordinator dataclasses remain usable."
            )

    class Side:            # type: ignore
        BID = "BID"
        ASK = "ASK"

    def str_to_ns(_: str) -> int:  # type: ignore
        return 0

    _ABIDES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coordination dataclasses (host-importable, no ABIDES dependency)
# ---------------------------------------------------------------------------
@dataclass
class CliqueAction:
    """One coordinated push event from the Coordinator to its members.

    Each member samples a firing time uniformly in
    ``[fire_at_ms, fire_at_ms + window_ms]`` and a size with ±size_jitter
    around ``base_size`` so the trade tape doesn't show a perfectly
    synchronous burst (which would be trivially detectable by Rung-1
    Pearson correlation).
    """

    action_id: str         # unique within a Coordinator, for fired-tracking
    fire_at_ms: int        # session-relative ms when the firing window opens
    window_ms: int         # member firings sampled within [fire_at, fire_at+window]
    side: str              # "buy" | "sell"
    target_pct_move: float = 0.0       # informational only; not used to set price
    base_size: int = 100               # base order size before per-member jitter


@dataclass
class CliqueCoordinator:
    """Shared coordination state for all members of one clique.

    NOT an ABIDES agent. NOT registered with the kernel. Lives in-process
    so members can read shared decisions without using the message bus.

    The ``output_dir`` field is set by the run-config harness (Phase D);
    the first member's ``kernel_stopping`` calls ``flush_labels(output_dir)``
    to write the ``manipulator_labels.csv`` sidecar the adapter consumes.
    """

    clique_id: str
    member_ids: list[str] = field(default_factory=list)
    actions: list[CliqueAction] = field(default_factory=list)
    size_jitter: float = 0.25     # ±25% on order size
    jitter_ms: int = 250          # per-member temporal jitter on top of window
    scenario_id: str = "scenario_clique"
    scenario_label: str = "clique_alpha"
    scenario_type: str = "collusive_clique"
    output_dir: Optional[Path] = None
    # Anchor for translating ABIDES absolute nanosecond times into the
    # session-relative milliseconds used by CliqueAction.fire_at_ms. The
    # config harness sets this to mkt_open (in ABIDES ns). If left at 0,
    # session-relative ms == absolute ms (only useful for unit tests).
    reference_time_ns: int = 0
    _labels_flushed: bool = field(default=False, init=False, repr=False)

    def register_member(self, trader_id: str) -> None:
        if trader_id not in self.member_ids:
            self.member_ids.append(trader_id)

    def action_at(self, current_ms: int) -> Optional[CliqueAction]:
        """Return the (first) action whose firing window covers current_ms."""
        for a in self.actions:
            if a.fire_at_ms <= current_ms <= a.fire_at_ms + a.window_ms:
                return a
        return None

    def next_action_after(self, current_ms: int) -> Optional[CliqueAction]:
        """Return the earliest action whose window has not yet closed."""
        upcoming = [a for a in self.actions
                    if a.fire_at_ms + a.window_ms > current_ms]
        return min(upcoming, key=lambda a: a.fire_at_ms) if upcoming else None

    def flush_labels(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Idempotently write the manipulator_labels.csv sidecar.

        Called by the first member's kernel_stopping. Subsequent calls
        return ``None`` without rewriting. If ``output_dir`` is provided
        it overrides ``self.output_dir`` for this call.
        """
        if self._labels_flushed:
            return None
        target_dir = output_dir or self.output_dir
        if target_dir is None:
            logger.warning(
                "CliqueCoordinator %s: no output_dir set; "
                "manipulator_labels.csv not written.", self.clique_id,
            )
            return None
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "manipulator_labels.csv"
        # Append mode so multiple Coordinators in the same run merge into
        # one CSV. Each Coordinator writes only its own members.
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
# The agent itself
# ---------------------------------------------------------------------------
class CollusiveCliqueAgent(TradingAgent):
    """A TradingAgent member of one CliqueCoordinator.

    Lifecycle:
      __init__         — register with the Coordinator (off-bus).
      kernel_starting  — schedule first wakeup at the next action's fire window.
      wakeup           — if in an action window, request current spread.
                         Otherwise reschedule for the next pending action.
      receive_message  — when the spread comes back (QuerySpreadResponseMsg),
                         place a limit order on the action's side, mark the
                         action as fired, schedule next wakeup.
      kernel_stopping  — first member to stop flushes the Coordinator's
                         manipulator_labels.csv.
    """

    # MSA convention: trader_id = "trader_{abides_agent_id:05d}". The
    # adapter expects this format in both manipulator_labels.csv and the
    # orders/trades logs.
    @staticmethod
    def msa_trader_id(abides_agent_id: int) -> str:
        return f"trader_{int(abides_agent_id):05d}"

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        coordinator: CliqueCoordinator,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = True,
        wake_up_freq: "NanosecondTime" = None,  # type: ignore
    ) -> None:
        if not _ABIDES_AVAILABLE:
            # Re-raise from the stub TradingAgent so the message is clear.
            super().__init__(id, name, type, random_state, starting_cash, log_orders)
        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.coordinator = coordinator
        # Default poll cadence: 1s. We override per-action with explicit
        # set_wakeup() calls anyway, but this is the fallback if no actions
        # are pending and we just want to stay alive.
        self.wake_up_freq: NanosecondTime = (
            wake_up_freq if wake_up_freq is not None else str_to_ns("1s")
        )
        # State machine for receive_message dispatch.
        self.state: str = "AWAITING_WAKEUP"
        # Set of action_ids this agent has already fired on. Prevents
        # double-firing within the same window across multiple wakeups.
        self._fired_action_ids: set[str] = set()
        # The action we're about to fire on (filled in wakeup, consumed in
        # receive_message after the spread arrives).
        self._pending_action: Optional[CliqueAction] = None
        # Register off-bus with the Coordinator.
        coordinator.register_member(self.msa_trader_id(id))

    # ------------------------------------------------------------------
    # Convenience accessors (host-safe, no ABIDES required)
    # ------------------------------------------------------------------
    def manipulator_label(self) -> tuple[str, str, str]:
        return (
            self.coordinator.scenario_id,
            self.coordinator.scenario_label,
            self.coordinator.scenario_type,
        )

    # TradingAgent.receive_message calls this when it processes the
    # MarketHoursMsg reply, to schedule the first market-open wakeup.
    # MomentumAgent supports an optional Poisson arrival mode; we keep it
    # simple and return a constant cadence.
    def get_wake_frequency(self) -> "NanosecondTime":
        return self.wake_up_freq

    # ------------------------------------------------------------------
    # ABIDES lifecycle (snake_case, JPMC fork API)
    # ------------------------------------------------------------------
    def kernel_starting(self, start_time: "NanosecondTime") -> None:
        super().kernel_starting(start_time)
        # First wakeup: short delay so the exchange is up. Subsequent
        # wakeups are scheduled inside wakeup() based on coordinator state.
        self.set_wakeup(start_time + str_to_ns("100ms"))

    def kernel_stopping(self) -> None:
        # Flush manipulator labels BEFORE parent kernel_stopping so the
        # idempotency check in CliqueCoordinator runs while shared state is
        # still alive. Only the first member actually writes; the rest no-op.
        try:
            self.coordinator.flush_labels()
        except Exception as exc:  # noqa: BLE001
            logger.exception("CliqueCoordinator flush_labels failed: %s", exc)
        super().kernel_stopping()

    def wakeup(self, current_time: "NanosecondTime") -> None:
        # IMPORTANT: TradingAgent.wakeup returns a bool, but that's an
        # internal contract — the Kernel's runner loop interprets ANY
        # non-None return from wakeup as "gym experimental agent passed
        # state back" and exits the simulation. Subclasses MUST return
        # None (or use a bare `return`). MomentumAgent / NoiseAgent etc.
        # all return None implicitly. We do the same.
        can_trade = super().wakeup(current_time)
        if not can_trade:
            # Market not open yet (or already closed). Try again later.
            self.set_wakeup(current_time + self.wake_up_freq)
            return

        # Map ABIDES absolute nanoseconds → session-relative milliseconds,
        # using the Coordinator's anchor (set by the config harness to
        # mkt_open). With reference_time_ns=0 this becomes absolute ms,
        # which is fine for unit tests but wrong for real runs.
        current_ms = int((current_time - self.coordinator.reference_time_ns)
                         // 1_000_000)
        action = self.coordinator.action_at(current_ms)

        if action is not None and action.action_id not in self._fired_action_ids:
            # We're inside an active window we haven't fired on yet.
            # Request the current spread; receive_message will place the order.
            self._pending_action = action
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        # Not in an active window. Schedule wakeup at the next action's
        # firing window (with per-member jitter so the herd doesn't all wake
        # at the same instant).
        upcoming = self.coordinator.next_action_after(current_ms)
        if upcoming is None:
            # No more actions. Stay quiet until the market closes.
            self.set_wakeup(current_time + str_to_ns("60s"))
            return

        # Per-member jitter inside the window. If the window itself is wide
        # we sample uniformly across it; jitter_ms adds extra spread.
        rng = self.random_state if self.random_state is not None else np.random
        window_offset_ms = int(rng.uniform(0, upcoming.window_ms))
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
            self._place_clique_order(bid, ask)
            self.state = "AWAITING_WAKEUP"
            # Schedule the next wakeup so we can catch any further actions.
            self.set_wakeup(current_time + str_to_ns("500ms"))

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def _place_clique_order(self, bid: int, ask: int) -> None:
        action = self._pending_action
        if action is None:
            # Defensive: receive_message called outside the firing path.
            return
        if not bid or not ask:
            # No spread yet — try again next wakeup.
            logger.debug(
                "Clique %s member %d: no spread yet (bid=%s, ask=%s); "
                "will retry on next wakeup.",
                self.coordinator.clique_id, self.id, bid, ask,
            )
            return

        rng = self.random_state if self.random_state is not None else np.random
        size_mult = 1.0 + float(rng.uniform(-self.coordinator.size_jitter,
                                             +self.coordinator.size_jitter))
        quantity = max(1, int(round(action.base_size * size_mult)))

        if action.side == "buy":
            # Aggressive buy: lift the ask.
            self.place_limit_order(
                self.symbol,
                quantity=quantity,
                side=Side.BID,
                limit_price=ask,
            )
        elif action.side == "sell":
            # Aggressive sell: hit the bid.
            self.place_limit_order(
                self.symbol,
                quantity=quantity,
                side=Side.ASK,
                limit_price=bid,
            )
        else:
            logger.warning(
                "Clique %s action %s has unknown side %r; skipped.",
                self.coordinator.clique_id, action.action_id, action.side,
            )
            return

        self._fired_action_ids.add(action.action_id)
        self._pending_action = None
