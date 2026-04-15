"""
Claude Code Issues Analysis
- Time series: issues per day with Anthropic release annotations
- Knowledge graph: label co-occurrence + keyword clusters (D3.js, standalone HTML)
"""

import csv, json, re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

open_issues   = load("claude_code_issues.csv")
closed_issues = load("claude_code_issues_closed.csv")
all_issues    = open_issues + closed_issues

# ── Release milestones (from Anthropic docs + news) ───────────────────────────
RELEASES = [
    ("2025-03-06", "Claude Code\nPublic Beta"),
    ("2025-05-14", "Claude 4\n(Sonnet + Opus)"),
    ("2025-08-05", "Claude Opus 4.1"),
    ("2025-09-29", "Claude Sonnet 4.5"),
    ("2025-10-01", "Claude Haiku 4.5"),
    ("2025-11-01", "Claude Opus 4.5"),
    ("2026-02-05", "Claude Opus 4.6"),
    ("2026-02-17", "Claude Sonnet 4.6"),
    ("2026-03-24", "300k Output\nBeta"),
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. TIME SERIES
# ──────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame([
    {"date": r["created_at"][:10], "state": "open"   if r in open_issues else "closed"}
    for r in all_issues
])
df["date"] = pd.to_datetime(df["date"])

daily = df.groupby("date").size().reset_index(name="count")
daily = daily.set_index("date").reindex(
    pd.date_range(daily["date"].min(), daily["date"].max(), freq="D"), fill_value=0
).reset_index().rename(columns={"index": "date"})

daily["rolling7"]    = daily["count"].rolling(7,  min_periods=1).mean()
daily["rolling30"]   = daily["count"].rolling(30, min_periods=1).mean()
daily["cumulative"]  = daily["count"].cumsum()

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.06,
    subplot_titles=("Daily Issue Volume", "Cumulative Issues")
)

# Raw bars
fig.add_trace(go.Bar(
    x=daily["date"], y=daily["count"],
    name="Daily issues",
    marker_color="rgba(99,179,237,0.35)",
    hovertemplate="%{x|%b %d, %Y}: %{y} issues<extra></extra>",
), row=1, col=1)

# 7-day rolling
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["rolling7"],
    name="7-day avg",
    line=dict(color="#4299E1", width=2),
    hovertemplate="%{x|%b %d}: %{y:.0f} (7d avg)<extra></extra>",
), row=1, col=1)

# 30-day rolling
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["rolling30"],
    name="30-day avg",
    line=dict(color="#2B6CB0", width=2, dash="dot"),
    hovertemplate="%{x|%b %d}: %{y:.0f} (30d avg)<extra></extra>",
), row=1, col=1)

# Cumulative
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["cumulative"],
    name="Cumulative",
    fill="tozeroy",
    fillcolor="rgba(154,215,160,0.2)",
    line=dict(color="#48BB78", width=2),
    hovertemplate="%{x|%b %d}: %{y:,} total<extra></extra>",
), row=2, col=1)

# Release annotations
colours = ["#E53E3E","#DD6B20","#D69E2E","#38A169","#3182CE",
           "#805AD5","#D53F8C","#319795","#2C5282"]
for i, (date_str, label) in enumerate(RELEASES):
    dt = pd.Timestamp(date_str)
    col = colours[i % len(colours)]
    for row in [1, 2]:
        fig.add_vline(x=dt, line_width=1.5, line_dash="dash",
                      line_color=col, row=row, col=1)
    # annotation on top panel
    fig.add_annotation(
        x=dt, y=1.05, xref="x", yref="paper",
        text=label, showarrow=False,
        font=dict(size=8, color=col),
        textangle=-45, xanchor="left",
        bgcolor="rgba(255,255,255,0.7)",
    )

fig.update_layout(
    title=dict(
        text="<b>Claude Code GitHub Issues</b>  ·  March 2025 → April 2026<br>"
             "<sup>19,332 issues tracked · vertical lines = Anthropic model releases</sup>",
        font=dict(size=18),
        x=0.5,
    ),
    paper_bgcolor="#0F172A",
    plot_bgcolor="#1E293B",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(30,41,59,0.8)",
    ),
    hovermode="x unified",
    height=750,
    margin=dict(t=160, b=40, l=60, r=40),
)
fig.update_xaxes(gridcolor="#334155", zeroline=False)
fig.update_yaxes(gridcolor="#334155", zeroline=False)

fig.write_html(str(OUT / "timeseries.html"), include_plotlyjs="cdn")
fig.write_image(str(OUT / "timeseries.png"), width=1400, height=750, scale=2)
print("✓ timeseries.html + timeseries.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2. KNOWLEDGE GRAPH  (D3.js standalone HTML)
# ──────────────────────────────────────────────────────────────────────────────

# --- Label co-occurrence network ---
label_count  = Counter()
cooccurrence = Counter()

for r in all_issues:
    lbls = [l.strip() for l in r.get("labels","").split("|") if l.strip()]
    for l in lbls:
        label_count[l] += 1
    for i in range(len(lbls)):
        for j in range(i+1, len(lbls)):
            pair = tuple(sorted([lbls[i], lbls[j]]))
            cooccurrence[pair] += 1

# --- TF-IDF keywords per label cluster ---
label_titles = defaultdict(list)
for r in all_issues:
    lbls = [l.strip() for l in r.get("labels","").split("|") if l.strip()]
    title = r.get("title","")
    for l in lbls:
        if l.startswith("area:"):
            label_titles[l].append(title)

STOP = {"bug","feature","request","add","fix","error","issue","using","use",
        "not","does","when","with","from","into","for","the","and","in","on",
        "is","it","to","a","of","are","that","this","by","be","an","was",
        "has","have","if","at","or","as","can","cannot","will","should",
        "claude","code","claude's","vs","via","after","before","during",
        "update","new","support","allow","make","show","get","set","run",
        "does not","doesn't","don't","won't","isn't","can't","didn't"}

tfidf_keywords = {}
for label, titles in label_titles.items():
    if len(titles) < 3:
        continue
    tfidf = TfidfVectorizer(
        max_features=6, stop_words="english",
        ngram_range=(1,2), min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
    )
    try:
        tfidf.fit(titles)
        kws = [w for w in tfidf.get_feature_names_out()
               if w.lower() not in STOP and len(w) > 3]
        tfidf_keywords[label] = kws[:5]
    except Exception:
        pass

# --- Build D3 graph data ---
AREA_COLOUR = {
    "area:model":       "#EF4444",
    "area:cost":        "#F97316",
    "area:tui":         "#EAB308",
    "area:core":        "#22C55E",
    "area:agents":      "#3B82F6",
    "area:auth":        "#8B5CF6",
    "area:mcp":         "#EC4899",
    "area:cowork":      "#14B8A6",
    "area:plugins":     "#F59E0B",
    "area:tools":       "#6366F1",
    "area:desktop":     "#84CC16",
    "area:permissions": "#06B6D4",
    "area:docs":        "#A78BFA",
    "area:security":    "#FB7185",
}
DEFAULT_COL = "#94A3B8"

nodes, links = [], []
node_index   = {}

# Label nodes
TOP_LABELS = [l for l,_ in label_count.most_common(30) if l.startswith("area:")]
for label in TOP_LABELS:
    idx = len(nodes)
    node_index[label] = idx
    nodes.append({
        "id":    idx,
        "name":  label.replace("area:",""),
        "full":  label,
        "count": label_count[label],
        "type":  "label",
        "color": AREA_COLOUR.get(label, DEFAULT_COL),
        "keywords": tfidf_keywords.get(label, []),
    })

# Keyword nodes + edges
kw_node_index = {}
for label in TOP_LABELS:
    for kw in tfidf_keywords.get(label, []):
        if kw not in kw_node_index:
            kw_node_index[kw] = len(nodes)
            nodes.append({
                "id":    len(nodes),
                "name":  kw,
                "full":  kw,
                "count": 1,
                "type":  "keyword",
                "color": AREA_COLOUR.get(label, DEFAULT_COL) + "99",
                "keywords": [],
            })
        links.append({
            "source": node_index[label],
            "target": kw_node_index[kw],
            "value":  1,
            "type":   "keyword",
        })

# Co-occurrence edges between labels
for (l1, l2), weight in cooccurrence.most_common(60):
    if l1 in node_index and l2 in node_index and weight >= 5:
        links.append({
            "source": node_index[l1],
            "target": node_index[l2],
            "value":  weight,
            "type":   "cooccur",
        })

graph_data = json.dumps({"nodes": nodes, "links": links})
total_issues = len(all_issues)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude Code Issues · Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0F172A;
    color: #E2E8F0;
    font-family: 'Inter', system-ui, sans-serif;
    overflow: hidden;
  }}
  #header {{
    position: absolute; top: 0; left: 0; right: 0;
    padding: 16px 24px;
    background: linear-gradient(to bottom, #0F172A, transparent);
    z-index: 10;
    pointer-events: none;
  }}
  h1 {{ font-size: 1.4rem; font-weight: 700; color: #F1F5F9; }}
  h1 span {{ color: #60A5FA; }}
  p.sub {{ font-size: 0.78rem; color: #94A3B8; margin-top: 4px; }}
  #legend {{
    position: absolute; bottom: 20px; left: 20px;
    background: rgba(15,23,42,0.85);
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.72rem;
    z-index: 10;
    max-width: 200px;
  }}
  #legend h3 {{ color: #94A3B8; font-size: 0.7rem; text-transform: uppercase;
                letter-spacing: .05em; margin-bottom: 8px; }}
  .leg-item {{ display: flex; align-items: center; gap: 8px;
               margin-bottom: 5px; color: #CBD5E1; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  #tooltip {{
    position: absolute; pointer-events: none;
    background: rgba(15,23,42,0.95);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    max-width: 220px;
    display: none;
    z-index: 20;
  }}
  #tooltip .tt-title {{ font-weight: 600; color: #F1F5F9; margin-bottom: 4px; }}
  #tooltip .tt-count {{ color: #94A3B8; font-size: 0.72rem; }}
  #tooltip .tt-kw    {{ color: #60A5FA; font-size: 0.7rem; margin-top: 6px; }}
  svg {{ width: 100vw; height: 100vh; }}
  .node-label {{
    font-size: 11px; fill: #E2E8F0; pointer-events: none;
    text-shadow: 0 1px 3px #0F172A;
  }}
  .node-label.keyword {{ font-size: 9px; fill: #94A3B8; }}
  #controls {{
    position: absolute; top: 80px; right: 20px;
    display: flex; flex-direction: column; gap: 8px; z-index: 10;
  }}
  button {{
    background: #1E293B; border: 1px solid #334155;
    color: #94A3B8; border-radius: 6px; padding: 6px 12px;
    font-size: 0.75rem; cursor: pointer;
  }}
  button:hover {{ background: #334155; color: #E2E8F0; }}
</style>
</head>
<body>
<div id="header">
  <h1>Claude Code Issues · <span>Knowledge Graph</span></h1>
  <p class="sub">{total_issues:,} issues · nodes = area labels · edges = co-occurrence · outer ring = top keywords</p>
</div>

<div id="controls">
  <button onclick="resetZoom()">Reset zoom</button>
  <button onclick="toggleKeywords()">Toggle keywords</button>
</div>

<div id="legend">
  <h3>Top areas</h3>
  {"".join(f'<div class="leg-item"><div class="leg-dot" style="background:{AREA_COLOUR.get(l,DEFAULT_COL)}"></div>{l.replace("area:","")}</div>' for l in list(AREA_COLOUR.keys())[:10])}
</div>

<div id="tooltip">
  <div class="tt-title" id="tt-name"></div>
  <div class="tt-count" id="tt-count"></div>
  <div class="tt-kw"   id="tt-kw"></div>
</div>

<svg id="graph"></svg>

<script>
const DATA = {graph_data};

const svg = d3.select("#graph");
const W = window.innerWidth, H = window.innerHeight;
const g   = svg.append("g");
let showKw = true;

const zoom = d3.zoom().scaleExtent([0.3, 4]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const sizeScale = d3.scaleSqrt()
  .domain([0, d3.max(DATA.nodes.filter(n=>n.type==="label"), n=>n.count)])
  .range([12, 55]);

// links
const link = g.append("g").selectAll("line")
  .data(DATA.links.filter(l=>l.type==="cooccur"))
  .join("line")
  .attr("stroke", d => "#334155")
  .attr("stroke-opacity", 0.5)
  .attr("stroke-width", d => Math.sqrt(d.value * 0.3));

const kwLink = g.append("g").selectAll("line")
  .data(DATA.links.filter(l=>l.type==="keyword"))
  .join("line")
  .attr("stroke", "#1E3A5F")
  .attr("stroke-opacity", 0.4)
  .attr("stroke-width", 0.8);

// nodes
const node = g.append("g").selectAll("circle")
  .data(DATA.nodes)
  .join("circle")
  .attr("r", d => d.type==="label" ? sizeScale(d.count) : 5)
  .attr("fill", d => d.color)
  .attr("fill-opacity", d => d.type==="label" ? 0.85 : 0.6)
  .attr("stroke", d => d.type==="label" ? d.color : "none")
  .attr("stroke-width", d => d.type==="label" ? 2 : 0)
  .attr("stroke-opacity", 0.5)
  .style("cursor", "pointer")
  .call(d3.drag()
    .on("start", dragstart)
    .on("drag",  dragged)
    .on("end",   dragend))
  .on("mouseover", showTooltip)
  .on("mousemove", moveTooltip)
  .on("mouseout",  hideTooltip);

// labels
const label = g.append("g").selectAll("text")
  .data(DATA.nodes)
  .join("text")
  .attr("class", d => d.type==="keyword" ? "node-label keyword" : "node-label")
  .text(d => d.name)
  .attr("text-anchor", "middle")
  .attr("dy", d => d.type==="label" ? "0.35em" : "-7px");

// simulation
const sim = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.links).id(d=>d.id)
    .distance(d => d.type==="keyword" ? 80 : 160)
    .strength(d => d.type==="keyword" ? 0.3 : 0.08))
  .force("charge", d3.forceManyBody()
    .strength(d => d.type==="label" ? -600 : -80))
  .force("center", d3.forceCenter(W/2, H/2))
  .force("collision", d3.forceCollide()
    .radius(d => d.type==="label" ? sizeScale(d.count)+8 : 14));

sim.on("tick", () => {{
  link .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
       .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  kwLink.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
        .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node .attr("cx",d=>d.x).attr("cy",d=>d.y);
  label.attr("x",d=>d.x).attr("y",d=>d.y);
}});

function dragstart(e,d) {{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }}
function dragged(e,d)   {{ d.fx=e.x; d.fy=e.y; }}
function dragend(e,d)   {{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}

const tip = document.getElementById("tooltip");
function showTooltip(e,d) {{
  document.getElementById("tt-name").textContent  = d.full;
  document.getElementById("tt-count").textContent = d.type==="label"
    ? `${{d.count.toLocaleString()}} issues` : "keyword";
  document.getElementById("tt-kw").textContent = d.keywords?.length
    ? "Keywords: " + d.keywords.join(" · ") : "";
  tip.style.display = "block";
}}
function moveTooltip(e) {{
  tip.style.left = (e.clientX+14)+"px";
  tip.style.top  = (e.clientY-10)+"px";
}}
function hideTooltip() {{ tip.style.display="none"; }}

function resetZoom() {{
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
}}

function toggleKeywords() {{
  showKw = !showKw;
  kwLink.attr("display", showKw ? null : "none");
  g.selectAll("circle").filter(d=>d.type==="keyword").attr("display", showKw ? null : "none");
  g.selectAll("text").filter(d=>d.type==="keyword").attr("display", showKw ? null : "none");
}}
</script>
</body>
</html>"""

(OUT / "knowledge_graph.html").write_text(HTML, encoding="utf-8")
print("✓ knowledge_graph.html")
print("\nAll outputs in outputs/")
