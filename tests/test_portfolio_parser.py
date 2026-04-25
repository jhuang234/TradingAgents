import tempfile
import textwrap
import unittest
from pathlib import Path

from tradingagents.portfolio import (
    format_portfolio_context_for_prompt,
    parse_portfolio_file,
)


SAMPLE_PORTFOLIO_CSV = textwrap.dedent(
    '''\
    "Positions for account Individual ...894 as of 04:31 AM ET, 2026/04/25"

    "Symbol","Description","Qty (Quantity)","Price","Price Chng $ (Price Change $)","Price Chng % (Price Change %)","Mkt Val (Market Value)","Day Chng $ (Day Change $)","Day Chng % (Day Change %)","Cost Basis","Gain % (Gain/Loss %)","Gain $ (Gain/Loss $)","Ratings","Reinvest?","Reinvest Capital Gains?","% of Acct (% of Account)","Asset Type",
    "AMZN","AMAZON.COM INC","6","263.99","8.91","3.49%","$1,583.94","$53.46","3.49%","$1,365.00","16.04%","$218.94","C","No","N/A","0.91%","Equity",
    "XLP","STATE STREET CONSUMER STAPLES SELECT SECTOR SPDR ETF","100.3166","83.23","-0.25","-0.3%","$8,349.35","-$25.08","-0.3%","$8,389.56","-0.48%","-$40.21","--","Yes","N/A","4.8%","ETFs & Closed End Funds",
    "Cash & Cash Investments","--","--","--","--","--","$121,619.13","$0.00","0%","--","--","--","--","--","--","69.98%","Cash and Money Market",
    "Positions Total","","--","--","--","--","$173,787.25","$647.20","0.37%","$47,901.09","8.91%","$4,267.03","--","--","--","--","--",
    '''
)


class PortfolioParserTests(unittest.TestCase):
    def _write_temp_portfolio(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "portfolio.csv"
        path.write_text(SAMPLE_PORTFOLIO_CSV, encoding="utf-8")
        return path

    def test_parse_portfolio_file_keeps_cash_and_filters_summary_rows(self):
        portfolio_path = self._write_temp_portfolio()

        snapshot = parse_portfolio_file(portfolio_path)

        self.assertEqual(snapshot["broker"], "fidelity")
        self.assertEqual(snapshot["totals"]["cash_value"], 121619.13)
        self.assertEqual(snapshot["totals"]["cash_weight_percent"], 69.98)
        self.assertEqual(snapshot["totals"]["total_market_value"], 173787.25)
        self.assertAlmostEqual(snapshot["totals"]["invested_value"], 52168.12, places=2)
        self.assertEqual([position["ticker"] for position in snapshot["positions"]], ["AMZN", "XLP"])

    def test_format_prompt_mentions_cash_and_existing_position(self):
        portfolio_path = self._write_temp_portfolio()
        snapshot = parse_portfolio_file(portfolio_path)

        prompt_context = format_portfolio_context_for_prompt(snapshot, "AMZN")

        self.assertIn("Cash balance: $121,619.13", prompt_context)
        self.assertIn("Cash weight: 69.98%", prompt_context)
        self.assertIn("Current AMZN position: 6.0 shares", prompt_context)
        self.assertIn("Top current holdings by weight: XLP (4.80%, $8,349.35), AMZN (0.91%, $1,583.94)", prompt_context)


if __name__ == "__main__":
    unittest.main()
