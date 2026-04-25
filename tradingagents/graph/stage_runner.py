import time
from typing import Any, Callable, Dict

from .checkpoint import load_checkpoint, rewrite_checkpoint, save_checkpoint


TRANSIENT_API_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_API_MESSAGE_SNIPPETS = (
    "currently experiencing high demand",
    "try again later",
    "service unavailable",
    "servererror: 503",
    "503 unavailable",
    "resource exhausted",
    "too many requests",
    "rate limit",
    "temporarily unavailable",
    "deadline exceeded",
)

STAGES = [
    "analyst_reports",
    "investment_debate",
    "trader_plan",
    "risk_debate",
    "portfolio_decision",
    "signal_process",
]


def is_transient_api_error(exc: Exception) -> bool:
    current = exc
    visited = set()

    while current and id(current) not in visited:
        visited.add(id(current))

        for attr in ("status_code", "code"):
            status = _coerce_error_status(getattr(current, attr, None))
            if status in TRANSIENT_API_STATUS_CODES:
                return True

        message = str(current).lower()
        if any(snippet in message for snippet in TRANSIENT_API_MESSAGE_SNIPPETS):
            return True

        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return False


class StageRunner:
    def __init__(
        self,
        trading_graph,
        on_stage_start: Callable[[str, int, int], None] | None = None,
        on_stage_skip: Callable[[str, Dict[str, Any]], None] | None = None,
        on_retry: Callable[[str, int, int, int, Exception], None] | None = None,
        chunk_handler: Callable[[str, Dict[str, Any]], None] | None = None,
        debug_log: Callable[[str], None] | None = None,
        resume_from_stage: str | None = None,
        max_retries: int = 4,
        initial_backoff_seconds: int = 15,
        max_backoff_seconds: int = 120,
    ):
        self.trading_graph = trading_graph
        self.on_stage_start = on_stage_start
        self.on_stage_skip = on_stage_skip
        self.on_retry = on_retry
        self.chunk_handler = chunk_handler
        self.debug_log = debug_log
        self.resume_from_stage = resume_from_stage
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    def run(self, ticker: str, trade_date: str, portfolio_context: Dict[str, Any] | None = None, resume: bool = True):
        checkpoint = (
            load_checkpoint(self.trading_graph.config["results_dir"], ticker, trade_date)
            if resume
            else None
        )
        if checkpoint:
            self._log(
                f"Loaded checkpoint for {ticker} {trade_date}; completed={checkpoint.completed_stages}; last_completed={checkpoint.last_completed}"
            )
            if self.resume_from_stage:
                checkpoint = self._rewind_checkpoint(results_dir=self.trading_graph.config["results_dir"], checkpoint=checkpoint)
            state = checkpoint.state
            state.setdefault("messages", [("human", ticker)])
            completed_stages = list(checkpoint.completed_stages)
            created_at = checkpoint.created_at
        else:
            self._log(f"No checkpoint found for {ticker} {trade_date}; starting fresh run")
            state = self.trading_graph.propagator.create_initial_state(
                ticker, trade_date, portfolio_context=portfolio_context
            )
            completed_stages = []
            created_at = None

        for stage_index, stage_name in enumerate(STAGES, start=1):
            if stage_name in completed_stages:
                self._log(f"Skipping completed stage {stage_name}")
                if self.on_stage_skip:
                    self.on_stage_skip(stage_name, state)
                continue

            self._log(f"Starting stage {stage_index}/{len(STAGES)}: {stage_name}")
            if self.on_stage_start:
                self.on_stage_start(stage_name, stage_index, len(STAGES))

            state = self._run_with_retry(stage_name, state)
            completed_stages.append(stage_name)
            self._log(f"Completed stage {stage_name}; saving checkpoint")
            save_checkpoint(
                self.trading_graph.config["results_dir"],
                ticker,
                trade_date,
                completed_stages,
                state,
                created_at=created_at,
            )
            if created_at is None:
                checkpoint = load_checkpoint(self.trading_graph.config["results_dir"], ticker, trade_date)
                created_at = checkpoint.created_at if checkpoint else None
            self._log(f"Checkpoint saved for stage {stage_name}; completed={completed_stages}")

        self.trading_graph.curr_state = state
        self.trading_graph.ticker = ticker
        self.trading_graph._log_state(trade_date, state)
        self._log(f"All stages complete for {ticker} {trade_date}")
        return state

    def _run_with_retry(self, stage_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                self._log(f"Running stage {stage_name}; attempt {attempt + 1}")
                return self.trading_graph.run_stage(
                    stage_name,
                    state,
                    callbacks=self.trading_graph.callbacks,
                    chunk_handler=self.chunk_handler,
                )
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries or not is_transient_api_error(exc):
                    self._log(
                        f"Stage {stage_name} failed permanently on attempt {attempt}; transient={is_transient_api_error(exc)}; error={exc}"
                    )
                    raise
                wait_seconds = min(self.initial_backoff_seconds * (2 ** (attempt - 1)), self.max_backoff_seconds)
                self._log(
                    f"Stage {stage_name} transient failure on attempt {attempt}/{self.max_retries}; waiting {wait_seconds}s; error={exc}"
                )
                if self.on_retry:
                    self.on_retry(stage_name, attempt, self.max_retries, wait_seconds, exc)
                time.sleep(wait_seconds)

    def _log(self, message: str) -> None:
        if self.debug_log:
            self.debug_log(message)

    def _rewind_checkpoint(self, results_dir: str, checkpoint):
        if self.resume_from_stage not in STAGES:
            raise ValueError(f"Unknown resume stage: {self.resume_from_stage}")

        target_index = STAGES.index(self.resume_from_stage)
        completed = [
            stage_name for stage_name in checkpoint.completed_stages
            if STAGES.index(stage_name) < target_index
        ]
        checkpoint.completed_stages = completed
        checkpoint.last_completed = completed[-1] if completed else None
        rewrite_checkpoint(results_dir, checkpoint.ticker, checkpoint.trade_date, checkpoint)
        self._log(
            f"Rewound checkpoint to resume from {self.resume_from_stage}; completed now={checkpoint.completed_stages}"
        )
        return checkpoint


def _coerce_error_status(value):
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
