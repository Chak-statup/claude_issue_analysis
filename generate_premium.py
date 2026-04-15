"""
Premium Claude Code Issues Analysis
  1. Time series  — Jan 2026 →, light theme, annotated
  2. Knowledge graph — cosine ≥ 0.75, Louvain, Ollama labels,
     D3.js force-directed interactive HTML (GCC only)
  3. CSV exports  — nodes.csv, edges.csv
"""

import csv, json, re, time, requests
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from community import best_partition

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

COMMUNITY_PALETTE = [
    "#E74C3C","#E67E22","#F1C40F","#2ECC71","#1ABC9C",
    "#3498DB","#9B59B6","#E91E63","#FF5722","#00BCD4",
    "#8BC34A","#FF9800","#607D8B","#795548","#F06292",
    "#4CAF50","#2196F3","#9C27B0","#FF5252","#69F0AE",
]

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

open_issues   = load("claude_code_issues.csv")
closed_issues = load("claude_code_issues_closed.csv")
all_issues    = open_issues + closed_issues
print(f"Loaded {len(all_issues):,} issues")

# ── Annotations ───────────────────────────────────────────────────────────────
ANNOTATIONS = [
    {"date":"2026-02-05","label":"Opus 4.6",            "type":"model"},
    {"date":"2026-02-17","label":"Sonnet 4.6",          "type":"model"},
    {"date":"2026-02-17","label":"CC v2.1.45",          "type":"cc"},
    {"date":"2026-03-07","label":"CC /loop + cron",     "type":"cc"},
    {"date":"2026-03-14","label":"CC MCP elicit",       "type":"cc"},
    {"date":"2026-03-17","label":"128k context",        "type":"cc"},
    {"date":"2026-03-24","label":"300k output beta",    "type":"cc"},
    {"date":"2026-04-07","label":"CC Bedrock+high",     "type":"cc"},
    {"date":"2026-04-10","label":"CC /ultraplan",       "type":"cc"},
    {"date":"2026-04-14","label":"Caching+recap",       "type":"cc"},
    {"date":"2026-03-09","label":"Anthropic sues DoD",  "type":"news"},
    {"date":"2026-03-11","label":"Anthropic Institute", "type":"news"},
    {"date":"2026-03-12","label":"$100M Partner net",   "type":"news"},
    {"date":"2026-03-24","label":"Code review launch",  "type":"news"},
    {"date":"2026-04-01","label":"Repo removal",        "type":"news"},
    {"date":"2026-04-07","label":"Glasswing",           "type":"news"},
]

TYPE_STYLE = {
    "model": {"color":"#DC2626", "dash":"solid",  "symbol":"star"},
    "cc":    {"color":"#16A34A", "dash":"dash",   "symbol":"diamond"},
    "news":  {"color":"#2563EB", "dash":"dot",    "symbol":"circle"},
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. TIME SERIES
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Building time series ──")

df = pd.DataFrame([{"date": r["created_at"][:10]} for r in all_issues])
df["date"] = pd.to_datetime(df["date"])
START      = pd.Timestamp("2026-01-01")
df         = df[df["date"] >= START]

date_range = pd.date_range(START, df["date"].max(), freq="D")
daily      = df.groupby("date").size().reindex(date_range, fill_value=0).reset_index()
daily.columns = ["date", "count"]
daily["r7"]  = daily["count"].rolling(7,  min_periods=1).mean()
daily["r14"] = daily["count"].rolling(14, min_periods=1).mean()

# Weekend spans (shade Sat + Sun)
weekends = [d for d in date_range if d.weekday() >= 5]
weekend_spans, s, p = [], weekends[0], weekends[0]
for d in weekends[1:]:
    if (d - p).days == 1:
        p = d
    else:
        weekend_spans.append((s, p)); s = d; p = d
weekend_spans.append((s, p))

fig = go.Figure()

for ws, we in weekend_spans:
    fig.add_vrect(
        x0=ws - pd.Timedelta(hours=12), x1=we + pd.Timedelta(hours=12),
        fillcolor="rgba(241,245,249,1)", layer="below", line_width=0,
    )

fig.add_trace(go.Bar(
    x=daily["date"], y=daily["count"], name="Daily issues",
    marker_color="#93C5FD", marker_line_width=0,
    hovertemplate="%{x|%A, %b %d %Y}: <b>%{y}</b> issues<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["r7"], name="7-day avg",
    line=dict(color="#2563EB", width=2.5),
    hovertemplate="%{x|%b %d}: %{y:.1f} (7d)<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["r14"], name="14-day avg",
    line=dict(color="#1E40AF", width=1.8, dash="dot"),
    hovertemplate="%{x|%b %d}: %{y:.1f} (14d)<extra></extra>",
))

# Annotations — slot-based vertical stagger to prevent overlap
placed = []   # list of (timestamp, slot_used)
for ann in sorted(ANNOTATIONS, key=lambda a: a["date"]):
    dt = pd.Timestamp(ann["date"])
    if dt < START:
        continue
    sty = TYPE_STYLE[ann["type"]]

    nearby_slots = {slot for (t, slot) in placed if abs((t - dt).days) < 10}
    slot = 0
    while slot in nearby_slots:
        slot += 1

    y_frac = 0.97 - slot * 0.115
    r,g,b  = int(sty["color"][1:3],16), int(sty["color"][3:5],16), int(sty["color"][5:7],16)
    rgba   = f"rgba({r},{g},{b},0.38)"

    fig.add_vline(x=dt, line_width=1.2, line_dash=sty["dash"], line_color=rgba)
    fig.add_annotation(
        x=dt, y=y_frac, xref="x", yref="paper",
        text=ann["label"],
        showarrow=True, arrowhead=0, arrowwidth=1,
        arrowcolor=sty["color"], ay=-22, ax=5,
        font=dict(size=7.5, color=sty["color"],
                  family="Inter, system-ui, sans-serif"),
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor=sty["color"], borderwidth=1, borderpad=2,
        opacity=0.96,
    )
    placed.append((dt, slot))

for kind, sty in TYPE_STYLE.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=sty["color"], symbol=sty["symbol"], size=10),
        name={"model":"Model release","cc":"Claude Code update","news":"Anthropic news"}[kind],
        showlegend=True,
    ))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(color="rgba(203,213,225,1)", symbol="square", size=12),
    name="Weekend", showlegend=True,
))

n_window = int(daily["count"].sum())
fig.update_layout(
    title=dict(
        text="<b>Claude Code · GitHub Issue Activity</b>",
        font=dict(size=20, family="Inter, system-ui, sans-serif", color="#0F172A"),
        x=0.5, y=0.975, xanchor="center",
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#374151", family="Inter, system-ui, sans-serif"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.055, xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#E2E8F0", borderwidth=1,
        font=dict(size=10), traceorder="normal",
    ),
    hovermode="x unified",
    bargap=0.18,
    height=600,
    margin=dict(t=120, b=65, l=68, r=40),
    xaxis=dict(
        gridcolor="#F1F5F9", zeroline=False,
        showline=True, linecolor="#CBD5E1", linewidth=1,
        tickformat="%b %d", tickangle=-30, dtick="M1", tick0="2026-01-01",
    ),
    yaxis=dict(
        gridcolor="#F1F5F9", zeroline=False,
        showline=True, linecolor="#CBD5E1", linewidth=1,
        title=dict(text="Issues filed per day",
                   font=dict(size=11, color="#6B7280"), standoff=8),
    ),
)
fig.add_annotation(
    text=f"{n_window:,} issues · Jan – Apr 2026",
    xref="paper", yref="paper", x=0.5, y=1.055,
    showarrow=False, xanchor="center",
    font=dict(size=10, color="#9CA3AF", family="Inter, system-ui, sans-serif"),
)

fig.write_html(str(OUT / "timeseries_premium.html"), include_plotlyjs="cdn")
fig.write_image(str(OUT / "timeseries_premium.png"), width=1500, height=600, scale=2)
print("✓  timeseries_premium.html + .png")


# ──────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDINGS (cache-first)
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Vectorising issue titles ──")

titles    = [r["title"].strip() for r in all_issues]
EMB_CACHE = Path("outputs/embeddings.npy")

if EMB_CACHE.exists():
    embs  = np.load(str(EMB_CACHE)).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.maximum(norms, 1e-10)
    print(f"   loaded cached embeddings {embs.shape}")
else:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    t0    = time.time()
    embs  = model.encode(titles, batch_size=512, show_progress_bar=True,
                         convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.maximum(norms, 1e-10)
    print(f"   embedded {len(titles):,} titles in {time.time()-t0:.1f}s  shape={embs.shape}")
    np.save(str(EMB_CACHE), embs)
    print(f"   saved embeddings to cache")


# ──────────────────────────────────────────────────────────────────────────────
# 3. COSINE SIMILARITY GRAPH (numpy, no FAISS)
# ──────────────────────────────────────────────────────────────────────────────
print("── Batched cosine similarity search (numpy, sim ≥ 0.75) ──")
THRESHOLD = 0.80
BATCH     = 512
N         = len(embs)
edges     = []
t0        = time.time()

for start in range(0, N, BATCH):
    batch = embs[start:start + BATCH]
    sims  = (batch @ embs.T).astype("float32")
    for bi in range(len(batch)):
        i        = start + bi
        row      = sims[bi]
        row[i]   = 0.0
        js       = np.where(row >= THRESHOLD)[0]
        for j in js:
            if j <= i:
                continue
            edges.append((int(i), int(j), float(row[j])))
    if (start // BATCH) % 8 == 0:
        print(f"   {start}/{N} …", flush=True)

print(f"   search done in {time.time()-t0:.1f}s  →  {len(edges):,} edges")


# ──────────────────────────────────────────────────────────────────────────────
# 4. NETWORKX · GCC · LOUVAIN
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Building graph ──")
G = nx.Graph()
G.add_nodes_from(range(len(titles)))
for i, j, w in edges:
    G.add_edge(i, j, weight=w)

components = sorted(nx.connected_components(G), key=len, reverse=True)
gcc_nodes  = components[0]
GCC        = G.subgraph(gcc_nodes).copy()
print(f"   GCC: {len(GCC):,} nodes · {GCC.number_of_edges():,} edges "
      f"({len(gcc_nodes)/len(titles)*100:.1f}% of all issues)")

print("── Louvain community detection ──")
partition   = best_partition(GCC, weight="weight", random_state=42)
communities = defaultdict(list)
for node, cid in partition.items():
    communities[cid].append(node)
n_comm    = len(communities)
comm_sizes = {c: len(v) for c, v in communities.items()}
top_comms  = sorted(comm_sizes, key=comm_sizes.get, reverse=True)
print(f"   {n_comm} communities")


# ──────────────────────────────────────────────────────────────────────────────
# 5. OLLAMA LABELS
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Labelling communities with Ollama/mistral ──")

def ollama_label(sample_titles, max_retries=2):
    prompt = (
        "You are labelling clusters of GitHub issues from the 'Claude Code' CLI tool.\n"
        "Given these issue titles, reply with ONLY a concise 2-5 word label "
        "describing the common theme. No punctuation, no explanation.\n\n"
        "Issues:\n" +
        "\n".join(f"- {t}" for t in sample_titles[:12]) +
        "\n\nLabel:"
    )
    for _ in range(max_retries):
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model":"mistral","prompt":prompt,
                      "stream":False,"options":{"temperature":0.2,"num_predict":15}},
                timeout=20,
            )
            label = r.json()["response"].strip().strip('"\'')
            label = re.sub(r"[^\w\s\-/]", "", label)[:40]
            return label or "Mixed issues"
        except Exception:
            time.sleep(1)
    return "Mixed issues"

comm_labels = {}
for rank, cid in enumerate(top_comms):
    sample = [titles[n] for n in communities[cid][:20]]
    label  = ollama_label(sample)
    comm_labels[cid] = label
    print(f"   [{rank+1:02d}/{n_comm}] {comm_sizes[cid]:4d} nodes  → {label}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. LAYOUT (two-step: community macro → intra-community ring + noise)
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Computing layout ──")

comm_graph = nx.Graph()
for c in communities:
    comm_graph.add_node(c, size=len(communities[c]))
for i, j, w in edges:
    if i in gcc_nodes and j in gcc_nodes:
        ci, cj = partition[i], partition[j]
        if ci != cj:
            if comm_graph.has_edge(ci, cj):
                comm_graph[ci][cj]["weight"] += w
            else:
                comm_graph.add_edge(ci, cj, weight=w)

macro_pos = nx.spring_layout(
    comm_graph, weight="weight", seed=42,
    k=3.0 / np.sqrt(n_comm), iterations=150,
)

SCALE = 0.28
pos   = {}
for cid, nodes_in in communities.items():
    cx, cy = macro_pos[cid]
    r      = SCALE * np.sqrt(len(nodes_in))
    angles = np.linspace(0, 2 * np.pi, len(nodes_in), endpoint=False)
    rng    = np.random.default_rng(cid)
    for node, angle in zip(nodes_in, angles):
        noise    = rng.uniform(-r * 0.35, r * 0.35, 2)
        pos[node] = (cx + r * np.cos(angle) + noise[0],
                     cy + r * np.sin(angle) + noise[1])

print("   layout done")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CSV EXPORTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Exporting CSVs ──")

comm_colour = {cid: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)]
               for i, cid in enumerate(top_comms)}
degrees     = dict(GCC.degree())

# nodes.csv
with open(OUT / "nodes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["node_id","title","community_id","community_label","degree","color"])
    for node in sorted(GCC.nodes()):
        cid = partition[node]
        w.writerow([node, titles[node], cid, comm_labels[cid],
                    degrees[node], comm_colour[cid]])
print(f"   nodes.csv  ({len(GCC):,} rows)")

# edges.csv — top 40k by weight
sorted_edges = sorted(edges, key=lambda e: e[2], reverse=True)
gcc_set      = set(gcc_nodes)
gcc_edges    = [(i,j,w) for i,j,w in sorted_edges if i in gcc_set and j in gcc_set]
display_edges = gcc_edges[:40_000]

with open(OUT / "edges.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source","target","cosine_similarity"])
    for i, j, sim in display_edges:
        w.writerow([i, j, f"{sim:.4f}"])
print(f"   edges.csv  ({len(display_edges):,} rows, top by cosine similarity)")


# ──────────────────────────────────────────────────────────────────────────────
# 8. D3.js FORCE-DIRECTED KNOWLEDGE GRAPH
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Rendering D3.js knowledge graph ──")

# Normalise positions to [-1, 1] range for D3 scaling
all_x = [pos[n][0] for n in GCC.nodes()]
all_y = [pos[n][1] for n in GCC.nodes()]
cx_range = (min(all_x), max(all_x))
cy_range = (min(all_y), max(all_y))

def norm_coord(v, lo, hi):
    span = hi - lo or 1.0
    return 2.0 * (v - lo) / span - 1.0

# Node data
nodes_data = [
    {
        "id":         node,
        "title":      titles[node][:110],
        "community":  partition[node],
        "comm_label": comm_labels[partition[node]],
        "color":      comm_colour[partition[node]],
        "size":       round(2.5 + min(degrees[node], 28) * 0.32, 2),
        "degree":     degrees[node],
        "ix":         round(norm_coord(pos[node][0], *cx_range), 4),
        "iy":         round(norm_coord(pos[node][1], *cy_range), 4),
    }
    for node in GCC.nodes()
]

# Edge data — cap at 25k highest-weight edges for browser performance
edge_data = [
    {"source": i, "target": j, "weight": round(w, 4)}
    for i, j, w in gcc_edges[:25_000]
]

# Community metadata for legend + labels
comm_meta = [
    {
        "id":    cid,
        "label": comm_labels[cid],
        "color": comm_colour[cid],
        "size":  comm_sizes[cid],
    }
    for cid in top_comms
]

nodes_json = json.dumps(nodes_data, ensure_ascii=False)
edges_json = json.dumps(edge_data)
comms_json = json.dumps(comm_meta, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Code Issues · Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: Inter, -apple-system, system-ui, sans-serif;
    background: #ffffff;
    color: #111827;
    overflow: hidden;
  }}
  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 20;
    padding: 16px 24px 10px;
    background: rgba(255,255,255,0.97);
    border-bottom: 1px solid #F1F5F9;
  }}
  #header h1 {{
    font-size: 17px; font-weight: 700; color: #0F172A;
    letter-spacing: -0.3px;
  }}
  #header p {{
    font-size: 11.5px; color: #9CA3AF; margin-top: 2px; line-height: 1.4;
  }}
  svg {{
    display: block;
    cursor: grab;
  }}
  svg:active {{ cursor: grabbing; }}
  .link {{
    stroke: #94A3B8;
    stroke-opacity: 0.7;
    fill: none;
  }}
  .node-circle {{
    stroke: white;
    stroke-width: 1.8;
    cursor: pointer;
    transition: stroke-width 0.15s, opacity 0.15s;
  }}
  .node-circle:hover {{
    stroke: #374151;
    stroke-width: 2;
    opacity: 1 !important;
  }}
  .comm-label {{
    font-size: 11px;
    font-weight: 650;
    fill: #1E293B;
    pointer-events: none;
    text-anchor: middle;
    letter-spacing: -0.2px;
    paint-order: stroke;
    stroke: rgba(255,255,255,0.85);
    stroke-width: 3px;
  }}
  .comm-count {{
    font-size: 9px;
    fill: #6B7280;
    pointer-events: none;
    text-anchor: middle;
  }}
  #tooltip {{
    position: fixed;
    pointer-events: none;
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 12px;
    max-width: 280px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    opacity: 0;
    transition: opacity 0.12s;
    z-index: 100;
    line-height: 1.5;
  }}
  #tooltip .tt-title {{
    font-weight: 600;
    color: #0F172A;
    font-size: 12px;
    margin-bottom: 4px;
  }}
  #tooltip .tt-comm {{
    display: inline-block;
    font-size: 10.5px;
    padding: 1px 7px;
    border-radius: 99px;
    color: white;
    margin-bottom: 3px;
  }}
  #tooltip .tt-deg {{
    font-size: 10px;
    color: #9CA3AF;
  }}
  #legend {{
    position: fixed;
    top: 72px; right: 16px;
    background: rgba(255,255,255,0.97);
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    max-height: calc(100vh - 90px);
    overflow-y: auto;
    z-index: 10;
    width: 190px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
  }}
  #legend h3 {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #6B7280;
    margin-bottom: 8px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 5px;
    cursor: pointer;
    border-radius: 4px;
    padding: 2px 3px;
    transition: background 0.1s;
  }}
  .legend-item:hover {{ background: #F8FAFC; }}
  .legend-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1.5px solid rgba(255,255,255,0.6);
  }}
  .legend-label {{
    font-size: 10.5px;
    color: #374151;
    line-height: 1.3;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .legend-count {{
    font-size: 9.5px;
    color: #9CA3AF;
    flex-shrink: 0;
  }}
  #controls {{
    position: fixed; bottom: 18px; left: 50%;
    transform: translateX(-50%);
    display: flex; gap: 8px; z-index: 20;
  }}
  #controls button {{
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 7px;
    padding: 6px 14px;
    font-size: 11px;
    font-family: inherit;
    color: #374151;
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: background 0.1s, box-shadow 0.1s;
  }}
  #controls button:hover {{
    background: #F8FAFC;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  #node-count {{
    position: fixed; bottom: 18px; left: 20px;
    font-size: 10px; color: #9CA3AF; z-index: 10;
  }}
</style>
</head>
<body>

<div id="header">
  <h1>Claude Code Issues · Semantic Similarity Graph</h1>
  <p>
    Giant Connected Component · {len(GCC):,} issues · {n_comm} communities ·
    cosine similarity ≥ {THRESHOLD} · all-MiniLM-L6-v2 + Louvain · labels by Mistral
  </p>
</div>

<div id="tooltip">
  <div class="tt-title" id="tt-title"></div>
  <span class="tt-comm" id="tt-comm"></span><br>
  <span class="tt-deg" id="tt-deg"></span>
</div>

<div id="legend">
  <h3>Communities</h3>
  <div id="legend-items"></div>
</div>

<div id="controls">
  <button onclick="resetZoom()">Reset view</button>
  <button onclick="toggleLabels()">Toggle labels</button>
  <button onclick="pauseResume()">Pause physics</button>
</div>
<div id="node-count"></div>

<svg id="graph"></svg>

<script>
const NODES      = {nodes_json};
const EDGES      = {edges_json};
const COMMUNITIES = {comms_json};

const HEADER_H = 58;
const W = window.innerWidth;
const H = window.innerHeight - HEADER_H;

const svg = d3.select("#graph")
    .attr("width",  W)
    .attr("height", H)
    .style("margin-top", HEADER_H + "px");

const zoomBehavior = d3.zoom()
    .scaleExtent([0.04, 10])
    .on("zoom", e => container.attr("transform", e.transform));
svg.call(zoomBehavior);

const container  = svg.append("g");
const linkGroup  = container.append("g").attr("class", "link-group");
const nodeGroup  = container.append("g").attr("class", "node-group");
const labelGroup = container.append("g").attr("class", "label-group");

// Scale initial positions
const xScale = d3.scaleLinear().domain([-1, 1]).range([60, W - 210]);
const yScale = d3.scaleLinear().domain([-1, 1]).range([40, H - 40]);

const nodeById = new Map(NODES.map(n => [n.id, n]));

NODES.forEach(n => {{
    n.x = xScale(n.ix);
    n.y = yScale(n.iy);
}});

const links = EDGES
    .map(e => ({{ source: nodeById.get(e.source), target: nodeById.get(e.target), weight: e.weight }}))
    .filter(e => e.source && e.target);

// ── Simulation ────────────────────────────────────────────────────────────────
const simulation = d3.forceSimulation(NODES)
    .force("link",    d3.forceLink(links).id(d => d.id)
                        .distance(d => 18 + (1.0 - d.weight) * 28)
                        .strength(0.18))
    .force("charge",  d3.forceManyBody().strength(-28).distanceMax(120))
    .force("collide", d3.forceCollide(d => d.size + 1.5).strength(0.6))
    .alphaDecay(0.03)
    .velocityDecay(0.38);

// ── Edges ─────────────────────────────────────────────────────────────────────
const link = linkGroup.selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "link")
    .attr("stroke-width", d => 0.55 + d.weight * 0.85);

// ── Nodes ─────────────────────────────────────────────────────────────────────
const node = nodeGroup.selectAll("circle")
    .data(NODES)
    .join("circle")
    .attr("class", "node-circle")
    .attr("r",    d => d.size)
    .attr("fill", d => d.color)
    .attr("opacity", 0.93)
    .call(drag(simulation));

// ── Community labels ──────────────────────────────────────────────────────────
const minLabelSize = 20;
const labelComms   = COMMUNITIES.filter(c => c.size >= minLabelSize);
const commNodes    = new Map(COMMUNITIES.map(c => [c.id, NODES.filter(n => n.community === c.id)]));

const commLabelText = labelGroup.selectAll("text.comm-label")
    .data(labelComms)
    .join("text")
    .attr("class", "comm-label")
    .text(d => d.label);

const commCountText = labelGroup.selectAll("text.comm-count")
    .data(labelComms)
    .join("text")
    .attr("class", "comm-count")
    .text(d => d.size + " issues");

function centroid(nodes) {{
    if (!nodes || nodes.length === 0) return {{x: 0, y: 0}};
    return {{
        x: nodes.reduce((s,n) => s + n.x, 0) / nodes.length,
        y: nodes.reduce((s,n) => s + n.y, 0) / nodes.length,
    }};
}}

// ── Tick ──────────────────────────────────────────────────────────────────────
simulation.on("tick", () => {{
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

    commLabelText.each(function(d) {{
        const c = centroid(commNodes.get(d.id));
        d3.select(this).attr("x", c.x).attr("y", c.y - 3);
    }});
    commCountText.each(function(d) {{
        const c = centroid(commNodes.get(d.id));
        d3.select(this).attr("x", c.x).attr("y", c.y + 10);
    }});
}});

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tooltip  = document.getElementById("tooltip");
const ttTitle  = document.getElementById("tt-title");
const ttComm   = document.getElementById("tt-comm");
const ttDeg    = document.getElementById("tt-deg");

node
    .on("mouseover", (event, d) => {{
        ttTitle.textContent = d.title;
        ttComm.textContent  = d.comm_label;
        ttComm.style.background = d.color;
        ttDeg.textContent   = "degree " + d.degree;
        tooltip.style.opacity = "1";
    }})
    .on("mousemove", event => {{
        const x = event.clientX + 14, y = event.clientY - 14;
        tooltip.style.left = (x + 280 > W ? x - 300 : x) + "px";
        tooltip.style.top  = (y + 80 > H ? y - 90 : y) + "px";
    }})
    .on("mouseout", () => {{ tooltip.style.opacity = "0"; }});

// ── Drag ─────────────────────────────────────────────────────────────────────
function drag(sim) {{
    return d3.drag()
        .on("start", (e, d) => {{
            if (!e.active) sim.alphaTarget(0.25).restart();
            d.fx = d.x; d.fy = d.y;
        }})
        .on("drag",  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
        .on("end",   (e, d) => {{
            if (!e.active) sim.alphaTarget(0);
            d.fx = null; d.fy = null;
        }});
}}

// ── Legend ───────────────────────────────────────────────────────────────────
const legendEl = document.getElementById("legend-items");
COMMUNITIES.forEach(c => {{
    const item = document.createElement("div");
    item.className = "legend-item";
    item.title = c.label;
    item.innerHTML =
        `<div class="legend-dot" style="background:${{c.color}}"></div>` +
        `<span class="legend-label">${{c.label}}</span>` +
        `<span class="legend-count">${{c.size}}</span>`;
    item.addEventListener("click", () => highlightComm(c.id));
    legendEl.appendChild(item);
}});

let highlighted = null;
function highlightComm(cid) {{
    if (highlighted === cid) {{
        node.attr("opacity", 0.93);
        highlighted = null;
    }} else {{
        node.attr("opacity", d => d.community === cid ? 0.96 : 0.08);
        highlighted = cid;
    }}
}}

// ── Controls ─────────────────────────────────────────────────────────────────
function resetZoom() {{
    svg.transition().duration(600)
       .call(zoomBehavior.transform, d3.zoomIdentity);
}}

let labelsVisible = true;
function toggleLabels() {{
    labelsVisible = !labelsVisible;
    labelGroup.style("display", labelsVisible ? null : "none");
}}

let paused = false;
function pauseResume() {{
    paused = !paused;
    paused ? simulation.stop() : simulation.restart();
    document.querySelector("#controls button:nth-child(3)").textContent =
        paused ? "Resume physics" : "Pause physics";
}}

document.getElementById("node-count").textContent =
    "{len(GCC):,} nodes · {len(edge_data):,} edges shown";

// Initial fit: zoom to bounding box after short settle
setTimeout(() => {{
    const bounds  = container.node().getBBox();
    const padding = 40;
    const scale   = Math.min(
        (W - padding * 2) / bounds.width,
        (H - padding * 2) / bounds.height,
        1.2
    );
    const tx = W / 2 - scale * (bounds.x + bounds.width  / 2);
    const ty = H / 2 - scale * (bounds.y + bounds.height / 2);
    svg.call(zoomBehavior.transform,
             d3.zoomIdentity.translate(tx, ty).scale(scale));
}}, 2200);
</script>
</body>
</html>"""

(OUT / "knowledge_graph_premium.html").write_text(html, encoding="utf-8")
print(f"✓  knowledge_graph_premium.html  ({len(html)//1024}KB)")

# ── Community metadata JSON ────────────────────────────────────────────────────
meta = {
    "generated":    "2026-04-15",
    "total_issues": len(all_issues),
    "gcc_size":     len(GCC),
    "threshold":    THRESHOLD,
    "n_communities": n_comm,
    "communities": [
        {
            "id":         cid,
            "label":      comm_labels[cid],
            "size":       comm_sizes[cid],
            "top_issues": [titles[n] for n in communities[cid][:15]],
        }
        for cid in top_comms
    ],
}
(OUT / "communities.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("✓  communities.json  (LLM context ready)")

print("\n✅  All outputs in outputs/")
print("   timeseries_premium.html / .png")
print("   knowledge_graph_premium.html   ← open in browser")
print("   nodes.csv · edges.csv · communities.json")
