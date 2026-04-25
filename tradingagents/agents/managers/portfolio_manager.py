from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.portfolio import format_portfolio_context_for_prompt


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        portfolio_context = state.get("portfolio_context", {})

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        portfolio_summary = format_portfolio_context_for_prompt(
            portfolio_context, state["company_of_interest"]
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**
- Portfolio context:
{portfolio_summary}

**Required Output Structure:**
1. **Rating**: State one of Buy / Overweight / Hold / Underweight / Sell.
2. **Current Position**
   - Current shares
   - Current account weight %
   - Current cash weight %
3. **Target Allocation**
   - Initial target weight %
   - Add-on target weight %
   - Max target weight %
4. **Execution Plan**
   - **Action Now**: initiate / add / hold / trim / exit
   - **Entry/Add Rule**: if price retraces to X, move to Y%; if price confirms above Z, move to Q%
   - **Stop-Loss**: exact price or % threshold that triggers a cut or full exit
   - **Take-Profit 1**: exact price, return %, or condition that triggers partial profit-taking, and what weight to trim to
   - **Take-Profit 2**: exact price, return %, or condition that triggers further profit-taking or full exit
   - **Thesis Invalidation**: exact condition that overrides the bullish thesis and forces reduction or exit
5. **Portfolio Fit**: Explain how the existing position, account weight, concentration risk, and cash balance affect the decision.
6. **Investment Thesis**: Detailed reasoning anchored in the analysts' debate and past reflections.

**Hard Requirements**
- Use explicit percentages for every target allocation item.
- If there is no current position, state `Current account weight: 0.00%`.
- If exact current shares are unknown, say `unknown`, but still provide percentage-based targets.
- Do not answer with only tranche language like "starter stake" or "small position"; translate every sizing recommendation into a percentage target.
- Be specific enough that a portfolio manager can execute the instruction without guessing the intended weight range.
- For any single stock or ETF, `Max target weight` must not exceed `5.00%` of total portfolio capital.
- `Execution Plan` must include a stop-loss mechanism. State exactly when to cut or exit the position if price action or thesis invalidation occurs.
- `Execution Plan` must include a take-profit mechanism. State exactly when to trim or realize gains if the trade works.
- Output every `Execution Plan` sub-item explicitly with the labels `Action Now`, `Entry/Add Rule`, `Stop-Loss`, `Take-Profit 1`, `Take-Profit 2`, and `Thesis Invalidation`.
- If the proposed target would otherwise exceed 5.00%, cap it at 5.00% and explain why.

---

**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
