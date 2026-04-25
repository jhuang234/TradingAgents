from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PortfolioPosition:
    ticker: str
    description: str
    quantity: float
    price: float
    market_value: float
    cost_basis: float
    gain_loss_percent: float | None
    gain_loss_value: float | None
    account_weight_percent: float | None
    asset_type: str
    raw_symbol: str = ""

    @property
    def average_cost(self) -> float | None:
        if self.quantity == 0:
            return None
        return self.cost_basis / self.quantity

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["average_cost"] = self.average_cost
        return payload


@dataclass
class PortfolioTotals:
    total_market_value: float | None = None
    total_cost_basis: float | None = None
    total_gain_loss_percent: float | None = None
    total_gain_loss_value: float | None = None
    cash_value: float | None = None
    cash_weight_percent: float | None = None
    invested_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    broker: str
    source_file: str
    account_label: str | None = None
    as_of: str | None = None
    totals: PortfolioTotals = field(default_factory=PortfolioTotals)
    positions: list[PortfolioPosition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "broker": self.broker,
            "source_file": self.source_file,
            "account_label": self.account_label,
            "as_of": self.as_of,
            "totals": self.totals.to_dict(),
            "positions": [position.to_dict() for position in self.positions],
        }
