import json
from datetime import datetime
from string import Template

from flask import Flask, request

app = Flask(__name__)

latest = {}
history = []

PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f778ba", "#a371f7", "#39c5cf", "#ff7b72", "#f0883e", "#7ee787", "#79c0ff"]


def fmt(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "-"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def jdata(x):
    return json.dumps(x)


@app.route("/", methods=["GET"])
def index():
    times = [h.get("time", "") for h in history]
    steps = [h.get("step", 0) for h in history]
    dbs = [h.get("db_count", 0) for h in history]
    rates = [h.get("rate") or 0 for h in history]

    temp_series = {}
    for h in history:
        seen = set()
        temps = h.get("gpu_temps")
        if temps:
            for g in temps:
                key = str(g.get("index"))
                temp_series.setdefault(key, {"label": "GPU %s" % g.get("index"), "data": []})
                temp_series[key]["data"].append(g.get("temp"))
                seen.add(key)
        elif h.get("gpu_temp") is not None:
            temp_series.setdefault("gpu", {"label": "GPU", "data": []})
            temp_series["gpu"]["data"].append(h["gpu_temp"])
            seen.add("gpu")
        for key, series in temp_series.items():
            if key not in seen:
                series["data"].append(None)

    temp_datasets = json.dumps(
        [
            {
                "label": series["label"],
                "data": series["data"],
                "borderColor": PALETTE[i % len(PALETTE)],
                "backgroundColor": PALETTE[i % len(PALETTE)] + "22",
                "fill": True,
                "tension": 0.2,
                "pointRadius": 0,
            }
            for i, series in enumerate(temp_series.values())
        ]
    )

    langs = latest.get("langs") or {}
    char_counts = latest.get("char_counts") or {}
    total_chunks = sum(langs.values())
    total_chars = sum(char_counts.values())

    lang_items = sorted(langs.items(), key=lambda kv: kv[1], reverse=True)[:15]
    lang_labels = [k for k, _ in lang_items]
    lang_values = [v for _, v in lang_items]
    lang_colors = [PALETTE[i % len(PALETTE)] for i in range(len(lang_labels))]

    char_items = sorted(char_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    char_labels = [k for k, _ in char_items]
    char_values = [v for _, v in char_items]

    rate_val = latest.get("rate")
    rate_display = ("%s/s" % fmt(rate_val)) if rate_val else "-"

    temp_cards = ""
    gpu_temps = latest.get("gpu_temps")
    if gpu_temps:
        for g in gpu_temps:
            temp_cards += '<div class="card"><div class="label">GPU %s Temp</div><div class="value">%s°C</div></div>' % (g.get("index"), g.get("temp"))
    elif latest.get("gpu_temp") is not None:
        temp_cards += '<div class="card"><div class="label">GPU Temp</div><div class="value">%s°C</div></div>' % latest.get("gpu_temp")
    else:
        temp_cards += '<div class="card"><div class="label">GPU Temp</div><div class="value">-</div></div>'

    cards = "".join(
        '<div class="card"><div class="label">%s</div><div class="value">%s</div></div>'
        % (label, value)
        for label, value in [
            ("Chunks Processed", fmt(latest.get("step", 0))),
            ("DB Entries", fmt(latest.get("db_count", 0))),
            ("Chunks", fmt(total_chunks)),
            ("Chars", fmt(total_chars)),
            ("Chunks/s", rate_display),
            ("Languages", str(len(langs))),
            ("Last Update", latest.get("time", "-")),
        ]
    ) + temp_cards

    rows = "".join(
        "<tr><td>%s</td><td class='num'>%s</td><td class='num'>%s</td><td class='num'>%s</td><td class='num'>%s</td></tr>"
        % (h.get("time", ""), fmt(h.get("step", 0)), fmt(h.get("db_count", 0)), fmt(h.get("rate") or 0), fmt(sum((h.get("langs") or {}).values())))
        for h in reversed(history[-100:])
    )

    return TEMPLATE.substitute(
        cards=cards,
        times=jdata(times),
        steps=jdata(steps),
        dbs=jdata(dbs),
        rates=jdata(rates),
        temp_datasets=temp_datasets,
        lang_labels=jdata(lang_labels),
        lang_values=jdata(lang_values),
        lang_colors=jdata(lang_colors),
        char_labels=jdata(char_labels),
        char_values=jdata(char_values),
        rows=rows,
        last_update=latest.get("time", "-"),
        total_steps=fmt(latest.get("step", 0)),
        total_db=fmt(latest.get("db_count", 0)),
        total_chunks=fmt(total_chunks),
    )


@app.route("/metrics", methods=["POST"])
def metrics():
    global latest
    try:
        data = request.get_json(force=True)
        latest = data
        history.append(data)
        if len(history) > 500:
            history.pop(0)
        return "ok"
    except Exception:
        return "bad request", 400


TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="60">
<title>FIM Metrics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:24px}
h1{font-size:22px;margin-bottom:4px;color:#f0f6fc}
.sub{font-size:13px;color:#8b949e;margin-bottom:20px}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;min-width:150px;flex:1}
.card .label{font-size:12px;color:#8b949e;margin-bottom:4px}
.card .value{font-size:24px;font-weight:700;color:#58a6ff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin-bottom:24px}
.chart-wrap{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px}
.chart-wrap h2{font-size:15px;margin-bottom:12px;color:#f0f6fc}
.chart-box{position:relative;height:260px}
table{width:100%;border-collapse:collapse}
thead th{text-align:left;padding:8px 12px;border-bottom:1px solid #21262d;font-size:12px;color:#8b949e;cursor:pointer;user-select:none;white-space:nowrap}
thead th:hover{color:#c9d1d9}
thead th .arrow{font-size:10px;margin-left:2px}
tbody td{padding:7px 12px;border-bottom:1px solid #21262d;font-size:13px}
tbody tr:hover{background:#1c2128}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.footer{margin-top:16px;font-size:12px;color:#484f58}
.filter-bar{margin-bottom:12px}
.filter-bar input{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:6px 12px;color:#c9d1d9;font-size:13px;width:240px;outline:none}
.filter-bar input:focus{border-color:#58a6ff}
.scroll{max-height:260px;overflow:auto}
</style>
</head>
<body>
<h1>FIM Embedding Monitor</h1>
<div class="sub">Auto-refresh every 60s &mdash; last update: $last_update</div>
<div class="cards">$cards</div>
<div class="grid">
<div class="chart-wrap"><h2>Progress (chunks &amp; DB entries)</h2><div class="chart-box"><canvas id="chProgress"></canvas></div></div>
<div class="chart-wrap"><h2>Rate (chunks/s)</h2><div class="chart-box"><canvas id="chRate"></canvas></div></div>
<div class="chart-wrap"><h2>GPU Temperature (°C)</h2><div class="chart-box"><canvas id="chTemp"></canvas></div></div>
</div>
<div class="grid">
<div class="chart-wrap"><h2>Chunks per language</h2><div class="chart-box"><canvas id="chLang"></canvas></div></div>
<div class="chart-wrap"><h2>Language distribution</h2><div class="chart-box"><canvas id="chDonut"></canvas></div></div>
</div>
<div class="grid">
<div class="chart-wrap"><h2>Characters per language</h2><div class="chart-box"><canvas id="chChars"></canvas></div></div>
<div class="chart-wrap">
<h2>History (last 100)</h2>
<div class="filter-bar"><input type="text" id="search" placeholder="Filter by time..." oninput="applyFilter()"></div>
<div class="scroll">
<table>
<thead><tr><th onclick="sortTable(0)">Time <span class="arrow">⇅</span></th><th onclick="sortTable(1)" class="num">Step <span class="arrow">⇅</span></th><th onclick="sortTable(2)" class="num">DB <span class="arrow">⇅</span></th><th onclick="sortTable(3)" class="num">Rate <span class="arrow">⇅</span></th><th onclick="sortTable(4)" class="num">Chunks <span class="arrow">⇅</span></th></tr></thead>
<tbody id="tbody">$rows</tbody>
</table>
</div>
</div>
</div>
<div class="footer">$total_steps chunks processed &mdash; $total_db db entries &mdash; $total_chunks chunks &mdash; data kept in memory on the logger</div>
<script>
const allRows=Array.from(document.querySelectorAll("#tbody tr"));
let curSort={col:1,dir:-1};
function parseNum(s){s=s.replace(/,/g,"");let m=1;if(s.endsWith("K")){s=s.slice(0,-1);m=1e3}else if(s.endsWith("M")){s=s.slice(0,-1);m=1e6}else if(s.endsWith("B")){s=s.slice(0,-1);m=1e9}return parseFloat(s)*m}
function compare(a,b){if(curSort.col===0){let va=a.cells[0].textContent.toLowerCase();let vb=b.cells[0].textContent.toLowerCase();return va<vb?-curSort.dir:va>vb?curSort.dir:0}let va=parseNum(a.cells[curSort.col].textContent);let vb=parseNum(b.cells[curSort.col].textContent);return (va-vb)*curSort.dir}
function sortTable(col){if(curSort.col===col)curSort.dir*=-1;else{curSort.col=col;curSort.dir=col===0?1:-1}applyFilter()}
function applyFilter(){let q=(document.getElementById("search").value||"").toLowerCase();let rows=allRows.filter(r=>r.cells[0].textContent.toLowerCase().includes(q));rows.sort(compare);let tbody=document.getElementById("tbody");tbody.innerHTML="";rows.forEach(r=>tbody.appendChild(r))}
const grid={color:"#21262d"},tick={color:"#8b949e"};
Chart.defaults.animation = false;
new Chart(document.getElementById("chProgress"),{type:"line",data:{labels:$times,datasets:[{label:"Files",data:$steps,borderColor:"#58a6ff",backgroundColor:"#58a6ff22",fill:true,tension:0.2,pointRadius:0},{label:"DB entries",data:$dbs,borderColor:"#3fb950",backgroundColor:"#3fb95022",fill:true,tension:0.2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:"#8b949e"}}},scales:{x:{grid:grid,ticks:tick},y:{grid:grid,ticks:tick}}}});
new Chart(document.getElementById("chRate"),{type:"line",data:{labels:$times,datasets:[{label:"chunks/s",data:$rates,borderColor:"#d29922",backgroundColor:"#d2992222",fill:true,tension:0.2,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:"#8b949e"}}},scales:{x:{grid:grid,ticks:tick},y:{grid:grid,ticks:tick}}}});
new Chart(document.getElementById("chTemp"),{type:"line",data:{labels:$times,datasets:$temp_datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:"#8b949e"}}},scales:{x:{grid:grid,ticks:tick},y:{grid:grid,ticks:tick,suggestedMin:0,suggestedMax:100}}}});
new Chart(document.getElementById("chLang"),{type:"bar",data:{labels:$lang_labels,datasets:[{data:$lang_values,backgroundColor:"#58a6ff55",borderColor:"#58a6ff",borderWidth:1}]},options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:grid,ticks:tick},y:{grid:{display:false},ticks:{color:"#8b949e",font:{size:11}}}}}});
new Chart(document.getElementById("chDonut"),{type:"doughnut",data:{labels:$lang_labels,datasets:[{data:$lang_values,backgroundColor:$lang_colors,borderColor:"#161b22",borderWidth:2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:"#8b949e",font:{size:11}}}}}});
new Chart(document.getElementById("chChars"),{type:"bar",data:{labels:$char_labels,datasets:[{data:$char_values,backgroundColor:"#3fb95055",borderColor:"#3fb950",borderWidth:1}]},options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:grid,ticks:{color:"#8b949e",callback:v=>v>=1e9?(v/1e9).toFixed(1)+"B":v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(1)+"K":v}},y:{grid:{display:false},ticks:{color:"#8b949e",font:{size:11}}}}}});
</script>
</body>
</html>""")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4242)
