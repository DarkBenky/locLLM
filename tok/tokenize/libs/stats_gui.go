package libs

import (
	"fmt"
	"sort"
	"strings"
)

func formatNum(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", float64(n)/1_000_000_000)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func formatNumInt64(n int64) string {
	return formatNum(int(n))
}

func escapeHTML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&#39;")
	return s
}

func RenderStatsHTML(stats *StatsResult) string {
	type catEntry struct {
		Name        string
		SampleCount int
		TokenCount  int
		ServedCount int
		ServedTok   int
	}

	entries := make([]catEntry, 0, len(stats.Categories))
	for _, c := range stats.Categories {
		entries = append(entries, catEntry{
			Name:        c.Category,
			SampleCount: c.SampleCount,
			TokenCount:  c.TokenCount,
			ServedCount: stats.CategoryServedCount[c.Category],
			ServedTok:   stats.CategoryServedTokCount[c.Category],
		})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].TokenCount > entries[j].TokenCount
	})

	chartN := 25
	if len(entries) < chartN {
		chartN = len(entries)
	}
	chartLabels := make([]string, chartN)
	chartData := make([]int, chartN)
	for i := 0; i < chartN; i++ {
		chartLabels[i] = entries[i].Name
		chartData[i] = entries[i].TokenCount
	}

	labelsJSON := "["
	dataJSON := "["
	for i := 0; i < chartN; i++ {
		if i > 0 {
			labelsJSON += ","
			dataJSON += ","
		}
		labelsJSON += "\"" + escapeHTML(chartLabels[i]) + "\""
		dataJSON += fmt.Sprintf("%d", chartData[i])
	}
	labelsJSON += "]"
	dataJSON += "]"

	served := make([]catEntry, len(entries))
	copy(served, entries)
	sort.Slice(served, func(i, j int) bool {
		return served[i].ServedTok > served[j].ServedTok
	})
	servedN := chartN
	if len(served) < servedN {
		servedN = len(served)
	}
	servedLabelsJSON := "["
	servedDataJSON := "["
	for i := 0; i < servedN; i++ {
		if i > 0 {
			servedLabelsJSON += ","
			servedDataJSON += ","
		}
		servedLabelsJSON += "\"" + escapeHTML(served[i].Name) + "\""
		servedDataJSON += fmt.Sprintf("%d", served[i].ServedTok)
	}
	servedLabelsJSON += "]"
	servedDataJSON += "]"

	totalServed := 0
	for _, v := range stats.CategoryServedTokCount {
		totalServed += v
	}

	tableRows := ""
	for _, e := range entries {
		tableRows += fmt.Sprintf(
			`<tr><td>%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td></tr>`,
			escapeHTML(e.Name),
			formatNum(e.SampleCount),
			formatNum(e.TokenCount),
			formatNum(e.ServedCount),
			formatNum(e.ServedTok),
		)
	}

	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Data Stats</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:24px}
h1{font-size:22px;margin-bottom:20px;color:#f0f6fc}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;min-width:180px;flex:1}
.card .label{font-size:12px;color:#8b949e;margin-bottom:4px}
.card .value{font-size:28px;font-weight:700;color:#58a6ff}
.chart-wrap{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:24px}
.chart-wrap h2{font-size:16px;margin-bottom:12px;color:#f0f6fc}
canvas{max-height:500px}
.filter-bar{margin-bottom:12px;display:flex;gap:8px;align-items:center}
.filter-bar input{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:6px 12px;color:#c9d1d9;font-size:13px;width:240px;outline:none}
.filter-bar input:focus{border-color:#58a6ff}
.filter-bar select{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:6px 12px;color:#c9d1d9;font-size:13px;outline:none}
table{width:100%%;border-collapse:collapse}
thead th{text-align:left;padding:8px 12px;border-bottom:1px solid #21262d;font-size:12px;color:#8b949e;cursor:pointer;user-select:none;white-space:nowrap}
thead th:hover{color:#c9d1d9}
thead th .arrow{font-size:10px;margin-left:2px}
tbody td{padding:7px 12px;border-bottom:1px solid #21262d;font-size:13px}
tbody tr:hover{background:#1c2128}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.footer{margin-top:16px;font-size:12px;color:#484f58}
</style>
</head>
<body>
<h1>Training Data Statistics</h1>
<div class="cards">
<div class="card"><div class="label">Total Samples</div><div class="value">%s</div></div>
<div class="card"><div class="label">Total Tokens</div><div class="value">%s</div></div>
<div class="card"><div class="label">Categories</div><div class="value">%d</div></div>
<div class="card"><div class="label">File Offset</div><div class="value">%s</div></div>
<div class="card"><div class="label">Total Served Tokens</div><div class="value">%s</div></div>
</div>
<div class="chart-wrap">
<h2>Top %d Categories by Token Count (stored)</h2>
<canvas id="chartStored"></canvas>
</div>
<div class="chart-wrap">
<h2>Top %d Categories by Tokens Served (trained on)</h2>
<canvas id="chartServed"></canvas>
</div>
<div class="chart-wrap">
<h2>All Categories (%d)</h2>
<div class="filter-bar">
<input type="text" id="search" placeholder="Filter categories..." oninput="filterTable()">
<select id="sortBy" onchange="sortTable()">
<option value="tokens" selected>Sort: Token Count</option>
<option value="samples">Sort: Sample Count</option>
<option value="served">Sort: Served Count</option>
<option value="servedTok">Sort: Served Tokens</option>
<option value="name">Sort: Name</option>
</select>
</div>
<table>
<thead>
<tr>
<th onclick="sortTableCol(0)">Category <span class="arrow">⇅</span></th>
<th onclick="sortTableCol(1)" class="num">Samples <span class="arrow">⇅</span></th>
<th onclick="sortTableCol(2)" class="num">Tokens <span class="arrow">⇅</span></th>
<th onclick="sortTableCol(3)" class="num">Served <span class="arrow">⇅</span></th>
<th onclick="sortTableCol(4)" class="num">Served Tok <span class="arrow">⇅</span></th>
</tr>
</thead>
<tbody id="tbody">
%s
</tbody>
</table>
</div>
<div class="footer">Data file offset: %d bytes &mdash; Refresh to update</div>
<script>
const allRows = Array.from(document.querySelectorAll("#tbody tr"));
let currentSort = {col:2, dir:-1};
function parseNum(s){s=s.replace(/[BMK,]/g,"");let m=1;if(s.endsWith("K")){s=s.slice(0,-1);m=1e3}else if(s.endsWith("M")){s=s.slice(0,-1);m=1e6}else if(s.endsWith("B")){s=s.slice(0,-1);m=1e9}return parseFloat(s)*m}
function compare(a,b){let va=parseNum(a.cells[currentSort.col].textContent);let vb=parseNum(b.cells[currentSort.col].textContent);return (va-vb)*currentSort.dir}
function sortTableCol(col){if(currentSort.col===col)currentSort.dir*=-1;else{currentSort.col=col;currentSort.dir=col>=1?-1:1}apply()}
function sortTable(){let v=document.getElementById("sortBy").value;let m={tokens:2,samples:1,served:3,servedTok:4,name:0};currentSort.col=m[v]??2;currentSort.dir=v==="name"?1:-1;apply()}
function filterTable(){apply()}
function apply(){let q=(document.getElementById("search").value||"").toLowerCase();let rows=allRows.filter(r=>r.cells[0].textContent.toLowerCase().includes(q));rows.sort(compare);let tbody=document.getElementById("tbody");tbody.innerHTML="";rows.forEach(r=>tbody.appendChild(r))}
new Chart(document.getElementById("chartStored"),{type:"bar",data:{labels:%s,datasets:[{label:"Token Count",data:%s,backgroundColor:"#58a6ff33",borderColor:"#58a6ff",borderWidth:1}]},options:{indexAxis:"y",responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:"#21262d"},ticks:{color:"#8b949e",callback:v=>v>=1e9?(v/1e9).toFixed(1)+"B":v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(1)+"K":v}},y:{grid:{display:false},ticks:{color:"#8b949e",font:{size:11}}}},maintainAspectRatio:false}})
new Chart(document.getElementById("chartServed"),{type:"bar",data:{labels:%s,datasets:[{label:"Served Tokens",data:%s,backgroundColor:"#3fb95033",borderColor:"#3fb950",borderWidth:1}]},options:{indexAxis:"y",responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:"#21262d"},ticks:{color:"#8b949e",callback:v=>v>=1e9?(v/1e9).toFixed(1)+"B":v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(1)+"K":v}},y:{grid:{display:false},ticks:{color:"#8b949e",font:{size:11}}}},maintainAspectRatio:false}})
</script>
</body>
</html>`,
		formatNum(stats.TotalSamples),
		formatNum(stats.TotalTokens),
		len(entries),
		formatNumInt64(stats.CurrentFileIndex),
		formatNum(totalServed),
		chartN,
		servedN,
		len(entries),
		tableRows,
		stats.CurrentFileIndex,
		labelsJSON,
		dataJSON,
		servedLabelsJSON,
		servedDataJSON,
	)
}
