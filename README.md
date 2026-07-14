# Supply Chain Disruption Agent — Hackathon Starter Kit

A single-file Streamlit app with an OpenAI tool-calling agent that detects stockout risk across a 4-warehouse pharma network and autonomously rebalances inventory, explaining every decision in plain English.

## Setup (5 minutes)

```bash
pip install streamlit openai pandas
export OPENAI_API_KEY=sk-...        # or paste it in the app sidebar
streamlit run hackathon_agent.py
```

Test tonight with `gpt-4o-mini` (costs a fraction of a rupee per query). Switch to `gpt-4o` in the sidebar at the event if credits are provided.

## The 3-minute judge demo

1. Show the green inventory table: "Here's our national pharma network — all healthy."
2. Sidebar → click **🔥 Demand spike: Delhi Paracetamol 3x** ("a dengue outbreak just hit Delhi").
3. In chat, type: **"Scan the network for risks and fix anything critical."**
4. Open the **🔧 Agent actions** expander — the agent visibly calls detect_risks → find_surplus → transfer_stock. Judges see real autonomy, not a chatbot.
5. Point at the table: stock has actually moved. Read the agent's plain-English summary aloud (what, why, cost, ETA).
6. Closer: "The guardrail matters — the agent physically cannot breach safety stock at a source warehouse. Try asking it to and it refuses. Safe autonomy, not just autonomy."

## Adapting to the event theme (60–90 min job)

The agent loop, chat UI, and tool-call display are theme-agnostic. You only touch three marked spots:

- **ADAPT #1 — data**: replace `default_inventory()` with whatever entities the theme needs (patients, orders, shipments, tickets...). Keep it a DataFrame.
- **ADAPT #2 — tools**: rewrite the four `tool_*` functions + `TOOLS_SCHEMA` for the new domain's actions. Pattern: read tools (inspect state) + one write tool (change state) + one guardrail (refuse unsafe actions).
- **ADAPT #3 — system prompt**: describe the new role, the workflow order, and always keep the "explain in plain English" instruction — explainability is your differentiator.

## Talking points if asked "what's novel?"

- Agent has **agency with guardrails**: it executes real state changes but validation logic (safety stock) sits in the tool, not the prompt — the LLM can't talk its way past it.
- **Explainability by design**: every tool call is logged and displayed; the final answer is a business-readable decision memo.
- Built on 5 years of real supply chain domain experience (control towers, S&OP) — the demo math (days of cover, safety stock, transfer cost/ETA) is how planners actually think.
