import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


CHECKPOINT_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint.json"
LEGACY_STAGE_MAPPINGS = {
    "risk_and_portfolio": ["risk_debate", "portfolio_decision"],
}


@dataclass
class StageCheckpoint:
    version: int
    ticker: str
    trade_date: str
    completed_stages: list[str]
    last_completed: str | None
    created_at: str
    updated_at: str
    state: Dict[str, Any]


def get_checkpoint_path(results_dir: str, ticker: str, trade_date: str) -> Path:
    return Path(results_dir) / ticker / trade_date / CHECKPOINT_FILENAME


def load_checkpoint(results_dir: str, ticker: str, trade_date: str) -> StageCheckpoint | None:
    checkpoint_path = get_checkpoint_path(results_dir, ticker, trade_date)
    if not checkpoint_path.exists():
        return None

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed_stages = _normalize_completed_stages(payload.get("completed_stages", []))
    last_completed = completed_stages[-1] if completed_stages else None

    return StageCheckpoint(
        version=payload["version"],
        ticker=payload["ticker"],
        trade_date=payload["trade_date"],
        completed_stages=completed_stages,
        last_completed=last_completed,
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        state=payload["state"],
    )


def rewrite_checkpoint(
    results_dir: str,
    ticker: str,
    trade_date: str,
    checkpoint: StageCheckpoint,
) -> Path:
    checkpoint_path = get_checkpoint_path(results_dir, ticker, trade_date)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": checkpoint.version,
        "ticker": checkpoint.ticker,
        "trade_date": checkpoint.trade_date,
        "completed_stages": checkpoint.completed_stages,
        "last_completed": checkpoint.last_completed,
        "created_at": checkpoint.created_at,
        "updated_at": _utc_now(),
        "state": checkpoint.state,
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return checkpoint_path


def save_checkpoint(
    results_dir: str,
    ticker: str,
    trade_date: str,
    completed_stages: list[str],
    state: Dict[str, Any],
    created_at: str | None = None,
) -> Path:
    checkpoint_path = get_checkpoint_path(results_dir, ticker, trade_date)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_now()
    payload = {
        "version": CHECKPOINT_VERSION,
        "ticker": ticker,
        "trade_date": trade_date,
        "completed_stages": completed_stages,
        "last_completed": completed_stages[-1] if completed_stages else None,
        "created_at": created_at or timestamp,
        "updated_at": timestamp,
        "state": sanitize_state_for_checkpoint(state),
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return checkpoint_path


def sanitize_state_for_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "company_of_interest": state.get("company_of_interest", ""),
        "trade_date": state.get("trade_date", ""),
        "portfolio_context": state.get("portfolio_context", {}),
        "market_report": state.get("market_report", ""),
        "sentiment_report": state.get("sentiment_report", ""),
        "news_report": state.get("news_report", ""),
        "fundamentals_report": state.get("fundamentals_report", ""),
        "investment_debate_state": dict(state.get("investment_debate_state", {})),
        "investment_plan": state.get("investment_plan", ""),
        "trader_investment_plan": state.get("trader_investment_plan", ""),
        "risk_debate_state": dict(state.get("risk_debate_state", {})),
        "final_trade_decision": state.get("final_trade_decision", ""),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_completed_stages(stage_names: list[str]) -> list[str]:
    normalized: list[str] = []
    for stage_name in stage_names:
        mapped = LEGACY_STAGE_MAPPINGS.get(stage_name, [stage_name])
        for resolved_name in mapped:
            if resolved_name not in normalized:
                normalized.append(resolved_name)
    return normalized
