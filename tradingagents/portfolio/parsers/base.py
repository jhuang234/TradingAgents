from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tradingagents.portfolio.models import PortfolioSnapshot


class PortfolioParser(ABC):
    broker_name = "unknown"

    @classmethod
    @abstractmethod
    def can_parse(cls, text: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, path: Path) -> PortfolioSnapshot:
        raise NotImplementedError
