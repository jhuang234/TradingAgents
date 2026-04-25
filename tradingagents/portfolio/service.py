from __future__ import annotations

from pathlib import Path

from tradingagents.portfolio.models import PortfolioSnapshot
from tradingagents.portfolio.parsers.fidelity import FidelityPositionsCsvParser


REGISTERED_PARSERS = [FidelityPositionsCsvParser]


def parse_portfolio_file(path: str | Path) -> dict:
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8-sig")

    for parser_cls in REGISTERED_PARSERS:
        if parser_cls.can_parse(raw_text):
            snapshot = parser_cls().parse(file_path)
            return snapshot.to_dict()

    raise ValueError(
        f"No portfolio parser matched {file_path}. Add a broker-specific parser under tradingagents/portfolio/parsers."
    )


def format_portfolio_context_for_prompt(portfolio_context: dict, ticker: str) -> str:
    if not portfolio_context:
        return "No portfolio context was provided."

    totals = portfolio_context.get("totals", {})
    positions = portfolio_context.get("positions", [])
    normalized_ticker = ticker.upper()
    current_position = next(
        (position for position in positions if position.get("ticker", "").upper() == normalized_ticker),
        None,
    )
    top_positions = sorted(
        positions,
        key=lambda position: position.get("account_weight_percent") or 0.0,
        reverse=True,
    )[:5]

    lines = [
        f"Broker source: {portfolio_context.get('broker', 'unknown')}",
        f"Source file: {portfolio_context.get('source_file', 'unknown')}",
    ]
    if portfolio_context.get("account_label"):
        lines.append(f"Account snapshot: {portfolio_context['account_label']}")
    if totals:
        lines.extend(
            [
                f"Total account value: {_format_currency(totals.get('total_market_value'))}",
                f"Cash balance: {_format_currency(totals.get('cash_value'))}",
                f"Cash weight: {_format_percent(totals.get('cash_weight_percent'))}",
                f"Invested value: {_format_currency(totals.get('invested_value'))}",
            ]
        )

    if current_position:
        lines.extend(
            [
                f"Current {normalized_ticker} position: {current_position.get('quantity')} shares",
                f"Current {normalized_ticker} market value: {_format_currency(current_position.get('market_value'))}",
                f"Current {normalized_ticker} account weight: {_format_percent(current_position.get('account_weight_percent'))}",
                f"Current {normalized_ticker} average cost: {_format_currency(current_position.get('average_cost'))}",
                f"Current {normalized_ticker} unrealized gain/loss: {_format_percent(current_position.get('gain_loss_percent'))} / {_format_currency(current_position.get('gain_loss_value'))}",
            ]
        )
    else:
        lines.append(f"Current {normalized_ticker} position: no existing position in the supplied portfolio snapshot.")

    if top_positions:
        holdings = ", ".join(
            (
                f"{position['ticker']} ({_format_percent(position.get('account_weight_percent'))}, "
                f"{_format_currency(position.get('market_value'))})"
            )
            for position in top_positions
        )
        lines.append(f"Top current holdings by weight: {holdings}")

    lines.append(
        "Use this portfolio context to decide whether the correct action is to initiate, add, hold, trim, or exit while still mapping the final rating to Buy / Overweight / Hold / Underweight / Sell."
    )
    return "\n".join(lines)


def _format_currency(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:,.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}%"
