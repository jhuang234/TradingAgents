import tempfile
import unittest
from pathlib import Path

from cli.utils import find_default_portfolio_path, get_portfolio_search_dirs


class PortfolioCliUtilsTests(unittest.TestCase):
    def test_search_dirs_include_repo_and_package_portfolio_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dirs = get_portfolio_search_dirs(base)

            self.assertEqual(
                dirs,
                [
                    base / "portfolio",
                    base / "tradingagents" / "portfolio",
                ],
            )

    def test_find_default_portfolio_path_prefers_newest_timestamp_in_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            portfolio_dir = Path(temp_dir) / "portfolio"
            package_dir = Path(temp_dir) / "tradingagents" / "portfolio"
            portfolio_dir.mkdir(parents=True)
            package_dir.mkdir(parents=True)

            older = portfolio_dir / "Individual-Positions-2026-04-25-043151.csv"
            newer = package_dir / "Individual-Positions-2026-04-26-010203.csv"
            older.write_text("older", encoding="utf-8")
            newer.write_text("newer", encoding="utf-8")

            resolved = find_default_portfolio_path([portfolio_dir, package_dir])

            self.assertEqual(resolved, str(newer))


if __name__ == "__main__":
    unittest.main()
