import tempfile
import unittest

from tradingagents.graph.checkpoint import load_checkpoint, save_checkpoint
from tradingagents.graph.stage_runner import StageRunner


class FakeTransientError(Exception):
    def __init__(self, message="busy", status_code=503):
        super().__init__(message)
        self.status_code = status_code


class FakeGraph:
    def __init__(self, results_dir):
        self.config = {"results_dir": results_dir}
        self.callbacks = []
        self.propagator = self
        self.curr_state = None
        self.ticker = None
        self.logged = []
        self.calls = []
        self.fail_once = {"investment_debate": True}

    def create_initial_state(self, ticker, trade_date, portfolio_context=None):
        return {
            "messages": [("human", ticker)],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "portfolio_context": portfolio_context or {},
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "final_trade_decision": "",
        }

    def run_stage(self, stage_name, state, callbacks=None, chunk_handler=None):
        self.calls.append(stage_name)
        updated = dict(state)
        if stage_name == "analyst_reports":
            updated["market_report"] = "market"
        elif stage_name == "investment_debate":
            if self.fail_once["investment_debate"]:
                self.fail_once["investment_debate"] = False
                raise FakeTransientError()
            updated["investment_debate_state"] = {
                **updated["investment_debate_state"],
                "judge_decision": "debate done",
            }
            updated["investment_plan"] = "debate plan"
        elif stage_name == "trader_plan":
            updated["trader_investment_plan"] = "trader plan"
        elif stage_name == "risk_debate":
            updated["risk_debate_state"] = {
                **updated["risk_debate_state"],
                "history": "risk complete",
            }
        elif stage_name == "portfolio_decision":
            updated["risk_debate_state"] = {
                **updated["risk_debate_state"],
                "judge_decision": "portfolio done",
            }
            updated["final_trade_decision"] = "BUY"
        return updated

    def process_signal(self, signal):
        return signal

    def _log_state(self, trade_date, state):
        self.logged.append((trade_date, state["final_trade_decision"]))


class StageRunnerTests(unittest.TestCase):
    def test_checkpoint_round_trip_sanitizes_messages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                "messages": [("human", "AAPL")],
                "company_of_interest": "AAPL",
                "trade_date": "2026-04-25",
                "portfolio_context": {},
                "market_report": "market",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {},
                "investment_plan": "",
                "trader_investment_plan": "",
                "risk_debate_state": {},
                "final_trade_decision": "",
            }
            save_checkpoint(temp_dir, "AAPL", "2026-04-25", ["analyst_reports"], state)
            checkpoint = load_checkpoint(temp_dir, "AAPL", "2026-04-25")
            self.assertEqual(checkpoint.completed_stages, ["analyst_reports"])
            self.assertNotIn("messages", checkpoint.state)

    def test_legacy_checkpoint_stage_name_is_migrated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                "company_of_interest": "AAPL",
                "trade_date": "2026-04-25",
                "portfolio_context": {},
                "market_report": "market",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {},
                "investment_plan": "",
                "trader_investment_plan": "",
                "risk_debate_state": {},
                "final_trade_decision": "BUY",
            }
            save_checkpoint(
                temp_dir,
                "AAPL",
                "2026-04-25",
                ["analyst_reports", "investment_debate", "trader_plan", "risk_and_portfolio", "signal_process"],
                state,
            )
            checkpoint = load_checkpoint(temp_dir, "AAPL", "2026-04-25")
            self.assertEqual(
                checkpoint.completed_stages,
                [
                    "analyst_reports",
                    "investment_debate",
                    "trader_plan",
                    "risk_debate",
                    "portfolio_decision",
                    "signal_process",
                ],
            )

    def test_stage_runner_retries_only_failed_stage_and_resumes_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = FakeGraph(temp_dir)
            runner = StageRunner(graph, initial_backoff_seconds=0, max_backoff_seconds=0, max_retries=2)

            state = runner.run("AAPL", "2026-04-25", portfolio_context={"cash": 1}, resume=True)

            self.assertEqual(state["final_trade_decision"], "BUY")
            self.assertEqual(
                graph.calls,
                [
                    "analyst_reports",
                    "investment_debate",
                    "investment_debate",
                    "trader_plan",
                    "risk_debate",
                    "portfolio_decision",
                    "signal_process",
                ],
            )

            resumed_graph = FakeGraph(temp_dir)
            resumed_runner = StageRunner(resumed_graph, initial_backoff_seconds=0, max_backoff_seconds=0, max_retries=1)
            resumed_state = resumed_runner.run("AAPL", "2026-04-25", portfolio_context={"cash": 1}, resume=True)

            self.assertEqual(resumed_state["final_trade_decision"], "BUY")
            self.assertEqual(resumed_graph.calls, [])

    def test_stage_runner_can_rewind_to_portfolio_decision_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = FakeGraph(temp_dir)
            runner = StageRunner(graph, initial_backoff_seconds=0, max_backoff_seconds=0, max_retries=1)
            runner.run("AAPL", "2026-04-25", portfolio_context={"cash": 1}, resume=True)

            rerun_graph = FakeGraph(temp_dir)
            rerun_runner = StageRunner(
                rerun_graph,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
                max_retries=1,
                resume_from_stage="portfolio_decision",
            )
            rerun_state = rerun_runner.run("AAPL", "2026-04-25", portfolio_context={"cash": 1}, resume=True)

            self.assertEqual(rerun_state["final_trade_decision"], "BUY")
            self.assertEqual(rerun_graph.calls, ["portfolio_decision", "signal_process"])
            checkpoint = load_checkpoint(temp_dir, "AAPL", "2026-04-25")
            self.assertIn("risk_debate", checkpoint.completed_stages)
            self.assertIn("portfolio_decision", checkpoint.completed_stages)


if __name__ == "__main__":
    unittest.main()
