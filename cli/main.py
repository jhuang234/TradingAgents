from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(".env.enterprise", override=False)
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.checkpoint import load_checkpoint
from tradingagents.portfolio import (
    format_portfolio_context_for_prompt,
    parse_portfolio_file,
)
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Social Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()

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
INITIAL_RETRY_WAIT_SECONDS = 15
RETRY_WAIT_MULTIPLIER = 2
PRIMARY_MODEL_RETRY_ATTEMPTS = 3
FALLBACK_MODEL_RETRY_ATTEMPTS = 3
GOOGLE_FLASH_LITE_FALLBACK = "gemini-3.1-flash-lite-preview"


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


def is_transient_api_error(exc: Exception) -> bool:
    """Return True when an exception looks like a retryable provider failure."""
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


def build_analysis_attempt_configs(config: dict) -> list[dict]:
    """Build ordered retry/fallback configs for transient provider failures."""
    base = config.copy()
    provider = base.get("llm_provider", "").lower()
    if provider != "google":
        return [{"config": base, "wait_seconds": 0, "reason": "primary"}]

    attempts = []
    primary_spec = {
        "deep_think_llm": base["deep_think_llm"],
        "quick_think_llm": base["quick_think_llm"],
        "google_thinking_level": base.get("google_thinking_level"),
    }
    fallback_spec = {
        "deep_think_llm": GOOGLE_FLASH_LITE_FALLBACK,
        "quick_think_llm": GOOGLE_FLASH_LITE_FALLBACK,
        "google_thinking_level": None,
    }

    for retry_index in range(PRIMARY_MODEL_RETRY_ATTEMPTS):
        candidate = base.copy()
        candidate.update(primary_spec)
        attempts.append(
            {
                "config": candidate,
                "wait_seconds": 0 if retry_index == 0 else INITIAL_RETRY_WAIT_SECONDS * (RETRY_WAIT_MULTIPLIER ** (retry_index - 1)),
                "reason": "primary",
            }
        )

    for retry_index in range(FALLBACK_MODEL_RETRY_ATTEMPTS):
        candidate = base.copy()
        candidate.update(fallback_spec)
        attempts.append(
            {
                "config": candidate,
                "wait_seconds": INITIAL_RETRY_WAIT_SECONDS * (RETRY_WAIT_MULTIPLIER ** retry_index),
                "reason": "fallback",
            }
        )

    return attempts


def describe_attempt_config(config: dict) -> str:
    """Human-readable summary of the active model configuration."""
    provider = config.get("llm_provider", "").lower()
    summary = (
        f"deep={config.get('deep_think_llm')} | "
        f"quick={config.get('quick_think_llm')}"
    )
    if provider == "google":
        thinking = config.get("google_thinking_level") or "default"
        summary += f" | thinking={thinking}"
    return summary


def should_retry_full_analysis(exc: Exception, attempt_index: int, total_attempts: int, completed_reports_count: int) -> bool:
    """Allow whole-run retries only before any report section has completed."""
    return (
        attempt_index < total_attempts
        and completed_reports_count == 0
        and is_transient_api_error(exc)
    )


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections(
    portfolio_path_override: str | None = None,
    auto_load_portfolio: bool = True,
    llm_provider_override: str | None = None,
    shallow_thinker_override: str | None = None,
    deep_thinker_override: str | None = None,
    google_thinking_level_override: str | None = None,
    ticker_override: str | None = None,
    auto_defaults: bool = False,
):
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the exact ticker symbol to analyze, including exchange suffix when needed (examples: SPY, CNC.TO, 7203.T, 0700.HK)",
            "SPY",
        )
    )
    if auto_defaults and ticker_override:
        selected_ticker = normalize_ticker_symbol(ticker_override)
        console.print(f"[green]Ticker:[/green] {selected_ticker}")
    else:
        selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = get_recent_trading_day_for_ticker(selected_ticker) if auto_defaults else datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    if auto_defaults:
        analysis_date = default_date
        console.print(f"[green]Analysis date:[/green] {analysis_date}")
    else:
        analysis_date = get_analysis_date()

    # Step 3: Output language
    console.print(
        create_question_box(
            "Step 3: Output Language",
            "Select the language for analyst reports and final decision"
        )
    )
    if auto_defaults:
        output_language = "English"
        console.print(f"[green]Output language:[/green] {output_language}")
    else:
        output_language = ask_output_language()

    # Step 4: Portfolio source
    auto_portfolio_path = find_default_portfolio_path() if auto_load_portfolio else None
    resolved_portfolio_path = portfolio_path_override or auto_portfolio_path
    portfolio_default_label = resolved_portfolio_path or "Skip"
    portfolio_mode = (
        "Use --portfolio-file to force a specific CSV. Without it, the CLI auto-loads the newest portfolio CSV it can find."
        if auto_load_portfolio
        else "Portfolio auto-load disabled for this run."
    )
    console.print(
        create_question_box(
            "Step 4: Portfolio CSV",
            portfolio_mode,
            portfolio_default_label,
        )
    )
    if resolved_portfolio_path:
        console.print(f"[green]Portfolio file:[/green] {resolved_portfolio_path}")
    else:
        console.print("[yellow]Portfolio file:[/yellow] none")

    # Step 5: Select analysts
    console.print(
        create_question_box(
            "Step 5: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    if auto_defaults:
        selected_analysts = [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]
    else:
        selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 6: Research depth
    console.print(
        create_question_box(
            "Step 6: Research Depth", "Select your research depth level"
        )
    )
    if auto_defaults:
        selected_research_depth = 3
        console.print(f"[green]Research depth:[/green] Medium ({selected_research_depth})")
    else:
        selected_research_depth = select_research_depth()

    # Step 7: LLM Provider
    console.print(
        create_question_box(
            "Step 7: LLM Provider", "Select your LLM provider"
        )
    )
    if auto_defaults and not llm_provider_override:
        selected_llm_provider = config_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = DEFAULT_CONFIG.get("backend_url") or get_provider_backend_url(config_provider)
        console.print(f"[green]LLM provider:[/green] {selected_llm_provider}")
    elif llm_provider_override:
        selected_llm_provider = llm_provider_override.lower()
        backend_url = get_provider_backend_url(selected_llm_provider)
        console.print(f"[green]LLM provider:[/green] {selected_llm_provider}")
    else:
        selected_llm_provider, backend_url = select_llm_provider()

    # Step 8: Thinking agents
    console.print(
        create_question_box(
            "Step 8: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    if auto_defaults and not shallow_thinker_override:
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        console.print(f"[green]Quick-thinking model:[/green] {selected_shallow_thinker}")
    elif shallow_thinker_override:
        selected_shallow_thinker = shallow_thinker_override
        console.print(f"[green]Quick-thinking model:[/green] {selected_shallow_thinker}")
    else:
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)

    if auto_defaults and not deep_thinker_override:
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(f"[green]Deep-thinking model:[/green] {selected_deep_thinker}")
    elif deep_thinker_override:
        selected_deep_thinker = deep_thinker_override
        console.print(f"[green]Deep-thinking model:[/green] {selected_deep_thinker}")
    else:
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 9: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "Step 9: Thinking Mode",
                "Configure Gemini thinking mode"
            )
        )
        if auto_defaults and google_thinking_level_override is None:
            thinking_level = DEFAULT_CONFIG.get("google_thinking_level")
            console.print(f"[green]Gemini thinking mode:[/green] {thinking_level or 'default'}")
        elif google_thinking_level_override:
            thinking_level = google_thinking_level_override
            console.print(f"[green]Gemini thinking mode:[/green] {thinking_level}")
        else:
            thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 9: Reasoning Effort",
                "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()
    elif provider_lower == "anthropic":
        console.print(
            create_question_box(
                "Step 9: Effort Level",
                "Configure Claude effort level"
            )
        )
        anthropic_effort = ask_anthropic_effort()

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "portfolio_path": resolved_portfolio_path,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"])
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"])
        analyst_parts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"])
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"])
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"])
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"])
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"])
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"])
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"])
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"])
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"])
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            portfolio_context = final_state.get("portfolio_context", {})
            if portfolio_context:
                portfolio_summary = format_portfolio_context_for_prompt(portfolio_context, ticker)
                (portfolio_dir / "context.md").write_text(portfolio_summary)
                sections.append(f"## V. Portfolio Context\n\n{portfolio_summary}")
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"])
            decision_section = "VI" if portfolio_context else "V"
            sections.append(f"## {decision_section}. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections))
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            portfolio_context = final_state.get("portfolio_context", {})
            if portfolio_context:
                console.print(Panel("[bold]V. Portfolio Context[/bold]", border_style="green"))
                console.print(
                    Panel(
                        Markdown(format_portfolio_context_for_prompt(portfolio_context, final_state["company_of_interest"])),
                        title="Portfolio Snapshot",
                        border_style="blue",
                        padding=(1, 2),
                    )
                )
            decision_section_title = "[bold]VI. Portfolio Manager Decision[/bold]" if portfolio_context else "[bold]V. Portfolio Manager Decision[/bold]"
            console.print(Panel(decision_section_title, border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")


def hydrate_message_buffer_from_state(message_buffer, state):
    analyst_sections = {
        "market_report": "Market Analyst",
        "sentiment_report": "Social Analyst",
        "news_report": "News Analyst",
        "fundamentals_report": "Fundamentals Analyst",
    }
    for section, agent_name in analyst_sections.items():
        content = state.get(section)
        if content:
            message_buffer.update_report_section(section, content)
            message_buffer.update_agent_status(agent_name, "completed")

    debate_state = state.get("investment_debate_state") or {}
    if debate_state.get("bull_history"):
        message_buffer.update_agent_status("Bull Researcher", "completed")
    if debate_state.get("bear_history"):
        message_buffer.update_agent_status("Bear Researcher", "completed")
    if debate_state.get("judge_decision"):
        message_buffer.update_report_section(
            "investment_plan", f"### Research Manager Decision\n{debate_state['judge_decision']}"
        )
        message_buffer.update_agent_status("Research Manager", "completed")

    if state.get("trader_investment_plan"):
        message_buffer.update_report_section("trader_investment_plan", state["trader_investment_plan"])
        message_buffer.update_agent_status("Trader", "completed")

    risk_state = state.get("risk_debate_state") or {}
    if risk_state.get("aggressive_history"):
        message_buffer.update_agent_status("Aggressive Analyst", "completed")
    if risk_state.get("conservative_history"):
        message_buffer.update_agent_status("Conservative Analyst", "completed")
    if risk_state.get("neutral_history"):
        message_buffer.update_agent_status("Neutral Analyst", "completed")
    if risk_state.get("judge_decision"):
        message_buffer.update_report_section(
            "final_trade_decision", f"### Portfolio Manager Decision\n{risk_state['judge_decision']}"
        )
        message_buffer.update_agent_status("Portfolio Manager", "completed")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis(
    portfolio_path_override: str | None = None,
    auto_load_portfolio: bool = True,
    llm_provider_override: str | None = None,
    shallow_thinker_override: str | None = None,
    deep_thinker_override: str | None = None,
    google_thinking_level_override: str | None = None,
    resume_from_stage: str | None = None,
    ticker_override: str | None = None,
    auto_defaults: bool = False,
):
    # First get all user selections
    selections = get_user_selections(
        portfolio_path_override=portfolio_path_override,
        auto_load_portfolio=auto_load_portfolio,
        llm_provider_override=llm_provider_override,
        shallow_thinker_override=shallow_thinker_override,
        deep_thinker_override=deep_thinker_override,
        google_thinking_level_override=google_thinking_level_override,
        ticker_override=ticker_override,
        auto_defaults=auto_defaults,
    )
    portfolio_context = {}
    if selections.get("portfolio_path"):
        portfolio_context = parse_portfolio_file(selections["portfolio_path"])

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)
    stage_debug_log_file = results_dir / "stage_runner.log"
    stage_debug_log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w") as f:
                        f.write(text)
        return wrapper

    def write_stage_debug_log(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(stage_debug_log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()
    stats_handler = StatsCallbackHandler()
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    message_buffer.init_for_analysis(selected_analyst_keys)
    checkpoint = load_checkpoint(config["results_dir"], selections["ticker"], selections["analysis_date"])
    if checkpoint:
        hydrate_message_buffer_from_state(message_buffer, checkpoint.state)

    with Live(layout, refresh_per_second=4) as live:
        final_state = None

        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message("System", f"Analysis date: {selections['analysis_date']}")
        if portfolio_context:
            totals = portfolio_context.get("totals", {})
            message_buffer.add_message(
                "System",
                (
                    f"Loaded portfolio from {portfolio_context.get('source_file')} "
                    f"(cash {totals.get('cash_weight_percent', 'unknown')}%, "
                    f"{len(portfolio_context.get('positions', []))} positions)."
                ),
            )
        if checkpoint:
            message_buffer.add_message(
                "System",
                f"Resuming from checkpoint after stage: {checkpoint.last_completed or 'none'}",
            )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        def handle_stage_chunk(stage_name, chunk):
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            if stage_name == "analyst_reports":
                update_analyst_statuses(message_buffer, chunk)

            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                    )
                    message_buffer.update_agent_status("Aggressive Analyst", "completed")
                    message_buffer.update_agent_status("Conservative Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "completed")

            update_display(layout, stats_handler=stats_handler, start_time=start_time)

        def on_stage_start(stage_name, stage_index, stage_total):
            message_buffer.add_message(
                "System",
                f"Starting stage {stage_index}/{stage_total}: {stage_name}",
            )
            stage_agent = {
                "analyst_reports": f"{selected_analyst_keys[0].capitalize()} Analyst" if selected_analyst_keys else None,
                "investment_debate": "Bull Researcher",
                "trader_plan": "Trader",
                "risk_debate": "Aggressive Analyst",
                "portfolio_decision": "Portfolio Manager",
            }.get(stage_name)
            if stage_agent:
                message_buffer.update_agent_status(stage_agent, "in_progress")
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

        def on_stage_skip(stage_name, state):
            message_buffer.add_message("System", f"Skipping completed stage: {stage_name}")
            hydrate_message_buffer_from_state(message_buffer, state)
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

        def on_retry(stage_name, attempt, max_retries, wait_seconds, exc):
            message_buffer.add_message(
                "System",
                f"Transient API error in {stage_name}: {exc}. Retry {attempt}/{max_retries} after {wait_seconds}s.",
            )
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

        spinner_text = f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        final_state, decision = graph.propagate_staged(
            selections["ticker"],
            selections["analysis_date"],
            portfolio_context=portfolio_context,
            resume=True,
            resume_from_stage=resume_from_stage,
            on_stage_start=on_stage_start,
            on_stage_skip=on_stage_skip,
            on_retry=on_retry,
            chunk_handler=handle_stage_chunk,
            debug_log=write_stage_debug_log,
        )

        # Get final state and decision
        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
    try:
        report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
        console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
        console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
    except Exception as e:
        console.print(f"[red]Error saving report: {e}[/red]")

    display_complete_report(final_state)


@app.command()
def analyze(
    ticker: Optional[str] = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol to analyze. Use with --auto-defaults for non-interactive runs.",
    ),
    auto_defaults: bool = typer.Option(
        False,
        "--auto-defaults",
        help="Run non-interactively with automatic defaults: recent trading day, English output, all analysts, medium research depth, and no prompts beyond the provided ticker.",
    ),
    portfolio_file: Optional[Path] = typer.Option(
        None,
        "--portfolio-file",
        "-p",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a broker portfolio CSV. Overrides automatic latest-file discovery.",
    ),
    no_portfolio: bool = typer.Option(
        False,
        "--no-portfolio",
        help="Disable portfolio auto-loading for this run.",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm-provider",
        help="Override the interactive provider selection, e.g. google.",
    ),
    quick_model: Optional[str] = typer.Option(
        None,
        "--quick-model",
        help="Override the quick-thinking model selection.",
    ),
    deep_model: Optional[str] = typer.Option(
        None,
        "--deep-model",
        help="Override the deep-thinking model selection.",
    ),
    google_thinking_level: Optional[str] = typer.Option(
        None,
        "--google-thinking-level",
        help="Override Gemini thinking mode, e.g. high or minimal.",
    ),
    rerun_portfolio_manager: bool = typer.Option(
        False,
        "--rerun-portfolio-manager",
        help="Resume from the portfolio manager stage using the existing checkpoint for this ticker/date.",
    ),
):
    if auto_defaults and not ticker:
        raise typer.BadParameter("--ticker is required when --auto-defaults is used.")

    run_analysis(
        portfolio_path_override=str(portfolio_file) if portfolio_file else None,
        auto_load_portfolio=not no_portfolio,
        llm_provider_override=llm_provider,
        shallow_thinker_override=quick_model,
        deep_thinker_override=deep_model,
        google_thinking_level_override=google_thinking_level,
        resume_from_stage="portfolio_decision" if rerun_portfolio_manager else None,
        ticker_override=ticker,
        auto_defaults=auto_defaults,
    )


if __name__ == "__main__":
    app()
    def get_provider_backend_url(provider_key: str) -> str | None:
        return next(
            (
                url
                for provider, url in [
                    ("openai", "https://api.openai.com/v1"),
                    ("google", None),
                    ("anthropic", "https://api.anthropic.com/"),
                    ("xai", "https://api.x.ai/v1"),
                    ("deepseek", "https://api.deepseek.com"),
                    ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    ("glm", "https://open.bigmodel.cn/api/paas/v4/"),
                    ("openrouter", "https://openrouter.ai/api/v1"),
                    ("azure", None),
                    ("ollama", "http://localhost:11434/v1"),
                ]
                if provider == provider_key
            ),
            None,
        )
