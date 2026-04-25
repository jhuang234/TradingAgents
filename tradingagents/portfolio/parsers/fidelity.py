from __future__ import annotations

import csv
import re
from pathlib import Path

from tradingagents.portfolio.models import (
    PortfolioPosition,
    PortfolioSnapshot,
    PortfolioTotals,
)
from tradingagents.portfolio.parsers.base import PortfolioParser


def _parse_number(value: str) -> float | None:
    text = (value or "").strip().replace(",", "").replace("$", "").replace("%", "")
    if not text or text in {"--", "N/A"}:
        return None
    return float(text)


class FidelityPositionsCsvParser(PortfolioParser):
    broker_name = "fidelity"

    @classmethod
    def can_parse(cls, text: str) -> bool:
        return (
            "Positions for account" in text
            and '"Qty (Quantity)"' in text
            and '"% of Acct (% of Account)"' in text
        )

    def parse(self, path: Path) -> PortfolioSnapshot:
        raw_text = path.read_text(encoding="utf-8-sig")
        rows = list(csv.reader(raw_text.splitlines()))

        if len(rows) < 3:
            raise ValueError(f"Portfolio file {path} does not contain enough rows.")

        metadata = rows[0][0].strip() if rows[0] else ""
        as_of = None
        metadata_match = re.search(r"as of ([^,]+),\s*([0-9]{4}/[0-9]{2}/[0-9]{2})", metadata)
        if metadata_match:
            as_of = f"{metadata_match.group(2)} {metadata_match.group(1)}"
        header = rows[2]
        data_rows = rows[3:]

        positions: list[PortfolioPosition] = []
        totals = PortfolioTotals()

        for row in data_rows:
            if not any(cell.strip() for cell in row):
                continue

            cells = row + [""] * max(0, len(header) - len(row))
            record = dict(zip(header, cells))
            symbol = (record.get("Symbol") or "").strip()
            asset_type = (record.get("Asset Type") or "").strip()

            if symbol == "Cash & Cash Investments":
                totals.cash_value = _parse_number(record.get("Mkt Val (Market Value)", ""))
                totals.cash_weight_percent = _parse_number(record.get("% of Acct (% of Account)", ""))
                continue

            if symbol == "Positions Total":
                totals.total_market_value = _parse_number(record.get("Mkt Val (Market Value)", ""))
                totals.total_cost_basis = _parse_number(record.get("Cost Basis", ""))
                totals.total_gain_loss_percent = _parse_number(record.get("Gain % (Gain/Loss %)", ""))
                totals.total_gain_loss_value = _parse_number(record.get("Gain $ (Gain/Loss $)", ""))
                continue

            quantity = _parse_number(record.get("Qty (Quantity)", ""))
            price = _parse_number(record.get("Price", ""))
            market_value = _parse_number(record.get("Mkt Val (Market Value)", ""))
            cost_basis = _parse_number(record.get("Cost Basis", ""))

            if quantity is None or price is None or market_value is None or cost_basis is None:
                continue

            positions.append(
                PortfolioPosition(
                    ticker=symbol.upper(),
                    raw_symbol=symbol,
                    description=(record.get("Description") or "").strip(),
                    quantity=quantity,
                    price=price,
                    market_value=market_value,
                    cost_basis=cost_basis,
                    gain_loss_percent=_parse_number(record.get("Gain % (Gain/Loss %)", "")),
                    gain_loss_value=_parse_number(record.get("Gain $ (Gain/Loss $)", "")),
                    account_weight_percent=_parse_number(record.get("% of Acct (% of Account)", "")),
                    asset_type=asset_type,
                )
            )

        if totals.total_market_value is not None and totals.cash_value is not None:
            totals.invested_value = totals.total_market_value - totals.cash_value
        elif positions:
            totals.invested_value = sum(position.market_value for position in positions)

        return PortfolioSnapshot(
            broker=self.broker_name,
            source_file=str(path),
            account_label=metadata or None,
            as_of=as_of,
            totals=totals,
            positions=positions,
        )
