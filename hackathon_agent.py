"""
============================================================
SUPPLY CHAIN DISRUPTION AGENT — Hackathon Starter Kit
============================================================
A single-file Streamlit app with an OpenAI tool-calling agent
that monitors inventory, detects stockout risk, and proposes
(and executes) rebalancing transfers — explaining every step
in plain English for a business audience.

RUN IT:
    pip install streamlit openai pandas plotly
    export OPENAI_API_KEY=sk-...   (or paste key in the sidebar)
    streamlit run hackathon_agent.py

ADAPT IT (search for "ADAPT"):
    1. Swap the mock data for the hackathon's theme/dataset
    2. Rename / add tools in TOOL_FUNCTIONS + TOOLS_SCHEMA
    3. Rewrite SYSTEM_PROMPT for the new problem
Everything else (agent loop, chat UI, tool-call display) is
theme-agnostic plumbing you can keep as-is.
============================================================
"""

import json
import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"          # cheap for testing; switch to gpt-4o at the event
MAX_AGENT_STEPS = 8                    # safety cap on the reason->act loop
RISK_DAYS_OF_COVER = 3.0               # flag SKUs with < this many days of stock

# ADAPT #1 — distances only matter for this demo's transfer-cost math
DISTANCES_KM = {
    ("Delhi", "Mumbai"): 1400, ("Delhi", "Bangalore"): 2150,
    ("Delhi", "Kolkata"): 1500, ("Mumbai", "Bangalore"): 980,
    ("Mumbai", "Kolkata"): 1900, ("Bangalore", "Kolkata"): 1870,
}

WAREHOUSE_COORDS = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Bangalore": (12.97, 77.59),
    "Kolkata": (22.57, 88.36),
}


def distance_km(a: str, b: str) -> int:
    return DISTANCES_KM.get((a, b)) or DISTANCES_KM.get((b, a)) or 1000


# ------------------------------------------------------------------
# MOCK DATA  (ADAPT #1 — replace with the event's dataset / theme)
# ------------------------------------------------------------------
def default_inventory() -> pd.DataFrame:
    rows = [
        # sku, location, on_hand, daily_demand, safety_stock
        ("PARA-500 (Paracetamol)", "Delhi",     1800, 300, 900),
        ("PARA-500 (Paracetamol)", "Mumbai",    4200, 250, 750),
        ("PARA-500 (Paracetamol)", "Bangalore", 3900, 200, 600),
        ("PARA-500 (Paracetamol)", "Kolkata",   2100, 180, 540),
        ("AMOX-250 (Amoxicillin)", "Delhi",     2600, 220, 660),
        ("AMOX-250 (Amoxicillin)", "Mumbai",    1500, 260, 780),
        ("AMOX-250 (Amoxicillin)", "Bangalore", 5200, 210, 630),
        ("AMOX-250 (Amoxicillin)", "Kolkata",   2400, 150, 450),
        ("ORS-200  (ORS Sachets)", "Delhi",     3000, 400, 1200),
        ("ORS-200  (ORS Sachets)", "Mumbai",    6500, 380, 1140),
        ("ORS-200  (ORS Sachets)", "Bangalore", 2900, 300, 900),
        ("ORS-200  (ORS Sachets)", "Kolkata",   1600, 320, 960),
    ]
    return pd.DataFrame(rows, columns=["sku", "location", "on_hand", "daily_demand", "safety_stock"])


def with_cover(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["days_of_cover"] = (out["on_hand"] / out["daily_demand"]).round(1)
    out["status"] = out.apply(
        lambda r: "🔴 AT RISK" if (r.on_hand < r.safety_stock or r.days_of_cover < RISK_DAYS_OF_COVER)
        else ("🟡 WATCH" if r.days_of_cover < RISK_DAYS_OF_COVER * 2 else "🟢 OK"),
        axis=1,
    )
    return out


def inventory_map(df: pd.DataFrame, last_transfer: dict | None = None) -> go.Figure:
    """Map warehouse health and, when available, the most recent transfer."""
    view = with_cover(df)
    status_rank = {"OK": 0, "WATCH": 1, "AT RISK": 2}
    status_color = {"OK": "green", "WATCH": "yellow", "AT RISK": "red"}

    cities = []
    colors = []
    hover_text = []
    for city, (lat, lon) in WAREHOUSE_COORDS.items():
        city_rows = view[view["location"] == city]
        statuses = [
            next((label for label in status_rank if str(value).endswith(label)), "OK")
            for value in city_rows["status"]
        ]
        worst_status = max(statuses, key=status_rank.get) if statuses else "OK"
        sku_lines = [
            f"{row.sku}: {int(row.on_hand):,} units, {row.days_of_cover:.1f} days cover"
            for row in city_rows.itertuples()
        ]
        cities.append((city, lat, lon))
        colors.append(status_color[worst_status])
        hover_text.append(
            f"<b>{city}</b><br>Worst status: {worst_status}<br>" + "<br>".join(sku_lines)
        )

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=[lat for _, lat, _ in cities],
        lon=[lon for _, _, lon in cities],
        text=[city for city, _, _ in cities],
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
        mode="markers+text",
        textposition="top center",
        marker=dict(size=15, color=colors, line=dict(color="white", width=1.5)),
        name="Warehouses",
    ))

    if last_transfer:
        source = WAREHOUSE_COORDS.get(last_transfer["from"])
        destination = WAREHOUSE_COORDS.get(last_transfer["to"])
        if source and destination:
            src_lat, src_lon = source
            dst_lat, dst_lon = destination
            fig.add_trace(go.Scattergeo(
                lat=[src_lat, dst_lat],
                lon=[src_lon, dst_lon],
                mode="lines",
                line=dict(color="blue", width=3, dash="dash"),
                hoverinfo="skip",
                name=f"Last transfer: {last_transfer['qty']:,} {last_transfer['sku']}",
            ))
            arrow_fraction = 0.88
            arrow_lat = src_lat + (dst_lat - src_lat) * arrow_fraction
            arrow_lon = src_lon + (dst_lon - src_lon) * arrow_fraction
            fig.add_trace(go.Scattergeo(
                lat=[src_lat, arrow_lat],
                lon=[src_lon, arrow_lon],
                mode="markers",
                marker=dict(
                    size=[0, 13],
                    color="blue",
                    symbol="triangle-up",
                    angleref="previous",
                ),
                text=["", (
                    f"{last_transfer['sku']}: {last_transfer['qty']:,} units"
                    f"<br>{last_transfer['from']} to {last_transfer['to']}"
                )],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ))

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=0.01, x=0.01),
        geo=dict(
            projection_type="mercator",
            lataxis=dict(range=[6, 36]),
            lonaxis=dict(range=[66, 98]),
            showland=True,
            landcolor="#f3f4f6",
            showocean=True,
            oceancolor="#eaf4fb",
            showcountries=True,
            countrycolor="#9ca3af",
        ),
    )
    return fig


# ------------------------------------------------------------------
# TOOLS — pure functions that take the dataframe explicitly
# (ADAPT #2 — this is where you swap in the event theme's actions)
# ------------------------------------------------------------------
def tool_get_inventory(df: pd.DataFrame, location: str | None = None) -> dict:
    """Snapshot of inventory, optionally filtered to one location."""
    view = with_cover(df)
    if location:
        view = view[view["location"].str.lower() == location.lower()]
    return {"inventory": view.to_dict(orient="records")}


def tool_detect_risks(df: pd.DataFrame) -> dict:
    """Find SKU/location pairs at stockout risk."""
    view = with_cover(df)
    risks = view[view["status"] == "🔴 AT RISK"]
    return {
        "risk_threshold_days": RISK_DAYS_OF_COVER,
        "at_risk": risks.to_dict(orient="records"),
        "message": "No stockout risks detected." if risks.empty else f"{len(risks)} SKU-location(s) at risk.",
    }


def tool_find_surplus(df: pd.DataFrame, sku: str) -> dict:
    """Locations holding transferable surplus of a SKU (stock above 1.5x safety stock)."""
    view = df[df["sku"].str.contains(sku, case=False)]
    out = []
    for _, r in view.iterrows():
        surplus = int(r.on_hand - 1.5 * r.safety_stock)
        if surplus > 0:
            out.append({"location": r.location, "transferable_units": surplus, "on_hand": int(r.on_hand)})
    return {"sku_query": sku, "surplus_locations": out}


def tool_transfer_stock(df: pd.DataFrame, sku: str, from_location: str,
                        to_location: str, quantity: int) -> dict:
    """Validate and EXECUTE a stock transfer. Mutates inventory in session state."""
    quantity = int(quantity)
    mask_from = df["sku"].str.contains(sku, case=False) & (df["location"].str.lower() == from_location.lower())
    mask_to = df["sku"].str.contains(sku, case=False) & (df["location"].str.lower() == to_location.lower())
    if not mask_from.any() or not mask_to.any():
        return {"ok": False, "error": f"Could not find SKU '{sku}' at both locations."}

    src = df[mask_from].iloc[0]
    if src.on_hand - quantity < src.safety_stock:
        max_q = int(src.on_hand - src.safety_stock)
        return {"ok": False,
                "error": f"Transfer would breach safety stock at {src.location}. Max transferable: {max_q} units."}

    destination = df[mask_to].iloc[0]
    dist = distance_km(src.location, destination.location)
    cost_inr = round(quantity * (dist / 1000) * 2.0)      # ₹2 per unit per 1000 km (demo math)
    eta_days = math.ceil(dist / 500)                       # truck @ ~500 km/day

    # Execute
    df.loc[mask_from, "on_hand"] -= quantity
    df.loc[mask_to, "on_hand"] += quantity
    st.session_state.inv = df
    st.session_state.last_transfer = {
        "sku": sku,
        "from": src.location,
        "to": destination.location,
        "qty": quantity,
    }

    return {"ok": True, "sku": sku, "from": from_location, "to": to_location,
            "quantity": quantity, "estimated_cost_inr": cost_inr, "eta_days": eta_days}


TOOL_FUNCTIONS = {
    "get_inventory": tool_get_inventory,
    "detect_risks": tool_detect_risks,
    "find_surplus": tool_find_surplus,
    "transfer_stock": tool_transfer_stock,
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_inventory",
        "description": "Get current inventory snapshot with days-of-cover and risk status. Optionally filter by location.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "Optional city name, e.g. 'Delhi'"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "detect_risks",
        "description": "Scan all SKU-location pairs and return those at stockout risk (below safety stock or low days of cover).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "find_surplus",
        "description": "Find which locations hold transferable surplus of a given SKU.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string", "description": "SKU name or partial match, e.g. 'PARA-500'"}},
            "required": ["sku"]}}},
    {"type": "function", "function": {
        "name": "transfer_stock",
        "description": "Validate and execute a stock transfer between two locations. Fails if it would breach safety stock at the source.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string"},
            "from_location": {"type": "string"},
            "to_location": {"type": "string"},
            "quantity": {"type": "integer"}},
            "required": ["sku", "from_location", "to_location", "quantity"]}}},
]

# ADAPT #3 — rewrite this for whatever problem the event announces
SYSTEM_PROMPT = """You are a supply chain control tower agent for a pharma distributor with
four regional warehouses (Delhi, Mumbai, Bangalore, Kolkata).

Your job:
1. Use tools to inspect inventory and detect stockout risks — never guess numbers.
2. When risk exists, find surplus elsewhere and propose a specific rebalancing transfer
   (SKU, from, to, quantity). Prefer the cheapest viable source. Leave safety stock intact.
3. Execute the transfer with the transfer_stock tool when the user approves, or when they
   ask you to "fix it" directly.
4. ALWAYS end with a plain-English explanation a business stakeholder can follow:
   what you found, what you did, what it costs, when stock arrives, and the risk if no action is taken.

Be concise, numeric, and decisive. If no risks exist, say so and summarize network health."""


# ------------------------------------------------------------------
# AGENT LOOP — model reasons, calls tools, observes, repeats
# (theme-agnostic: no need to change this at the event)
# ------------------------------------------------------------------
def run_agent(client: OpenAI, model: str, chat_history: list) -> tuple[str, list]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
    tool_log = []

    for _ in range(MAX_AGENT_STEPS):
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS_SCHEMA)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or "(no response)", tool_log

        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = TOOL_FUNCTIONS.get(name)
            try:
                result = fn(st.session_state.inv, **args) if fn else {"error": f"Unknown tool {name}"}
            except Exception as e:  # surface tool errors to the model instead of crashing
                result = {"error": str(e)}
            tool_log.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, default=str)})  # default=str: safe on older pandas/numpy types

    return "⚠️ Agent hit the step limit — try a narrower question.", tool_log


# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Supply Chain Disruption Agent", page_icon="🚚", layout="wide")

if "inv" not in st.session_state:
    st.session_state.inv = default_inventory()
if "chat" not in st.session_state:
    st.session_state.chat = []          # [(role, content, tool_log)]
if "last_transfer" not in st.session_state:
    st.session_state.last_transfer = None

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("OpenAI API key", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""))
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5"], index=0,
                         help="Testing tonight: gpt-4o-mini (cheapest). At the event: try gpt-5-mini or gpt-5 with the provided key; if a model errors, fall back to gpt-4o.")

    st.divider()
    st.header("💥 Disruption Simulator")
    st.caption("Break the network, then ask the agent to fix it.")

    if st.button("🔥 Demand spike: Delhi Paracetamol 3x"):
        m = (st.session_state.inv.sku.str.contains("PARA")) & (st.session_state.inv.location == "Delhi")
        st.session_state.inv.loc[m, "daily_demand"] = 900
        st.toast("Delhi PARA-500 demand tripled (dengue outbreak scenario)")

    if st.button("🚧 Warehouse incident: Mumbai AMOX -70%"):
        m = (st.session_state.inv.sku.str.contains("AMOX")) & (st.session_state.inv.location == "Mumbai")
        st.session_state.inv.loc[m, "on_hand"] = (st.session_state.inv.loc[m, "on_hand"] * 0.3).astype(int)
        st.toast("Mumbai AMOX-250 stock damaged in warehouse flooding")

    if st.button("♻️ Reset everything"):
        st.session_state.inv = default_inventory()
        st.session_state.chat = []
        st.session_state.last_transfer = None
        st.rerun()

st.title("🚚 Supply Chain Disruption Agent")
st.caption("An autonomous agent that detects stockout risk and rebalances inventory — with explainable, plain-English decisions.")

st.subheader("📦 Live network inventory")
st.plotly_chart(
    inventory_map(st.session_state.inv, st.session_state.last_transfer),
    use_container_width=True,
    config={"displayModeBar": False},
)
st.dataframe(with_cover(st.session_state.inv), use_container_width=True, hide_index=True)

st.subheader("💬 Ask the agent")
st.caption('Try: "Scan the network for risks and fix anything critical."')

for role, content, tool_log in st.session_state.chat:
    with st.chat_message(role):
        if tool_log:
            with st.expander(f"🔧 Agent actions ({len(tool_log)} tool calls)"):
                for step in tool_log:
                    st.markdown(f"**{step['tool']}**  `{json.dumps(step['args'])}`")
                    st.json(step["result"], expanded=False)
        st.markdown(content)

if prompt := st.chat_input("e.g. What's at risk right now? Fix it."):
    if not api_key:
        st.error("Paste your OpenAI API key in the sidebar first.")
        st.stop()

    st.session_state.chat.append(("user", prompt, []))
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": r, "content": c} for r, c, _ in st.session_state.chat]
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning and acting..."):
            answer, tool_log = run_agent(OpenAI(api_key=api_key), model, history)
        if tool_log:
            with st.expander(f"🔧 Agent actions ({len(tool_log)} tool calls)"):
                for step in tool_log:
                    st.markdown(f"**{step['tool']}**  `{json.dumps(step['args'])}`")
                    st.json(step["result"], expanded=False)
        st.markdown(answer)

    st.session_state.chat.append(("assistant", answer, tool_log))
    st.rerun()
