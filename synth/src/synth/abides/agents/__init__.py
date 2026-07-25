"""Phase C: manipulator agents for injection into an ABIDES population.

Each agent extends abides.agent.TradingAgent (resolved at runtime — imports
are inside class bodies to keep this package importable without vendor/abides
present).

The three families mirror your existing synthesizer:
  - CollusiveCliqueAgent: members coordinate to push price in a chosen direction
  - RingTraderAgent:      members pass an order around a circle (wash-like)
  - FrontAccountAgent:    passive layer; obeys instructions from a ringleader
"""

from .collusive_clique import CollusiveCliqueAgent, CliqueCoordinator
from .ring_trader import RingTraderAgent, RingCoordinator
from .front_account import FrontAccountAgent

__all__ = [
    "CollusiveCliqueAgent",
    "CliqueCoordinator",
    "RingTraderAgent",
    "RingCoordinator",
    "FrontAccountAgent",
]
