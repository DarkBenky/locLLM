import os

import fastapi
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List

from main import build
from model import embed_texts
from db import CodeDB
from search_index import InMemoryIndex
from pydantic import BaseModel
import uvicorn


MODEL = None
DB = None
INDEX = None
GPU_INDEX = None

DB_PATH = "/media/user/2TB Clear/codeDB/db.db"
MAX_BATCH_SIZE = 8

app = fastapi.FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    texts: List[str]
    top_k: int = 1


@app.post("/search")
def search(req: SearchRequest):
    texts = req.texts
    top_k = req.top_k
    if not texts:
        return []

    batch = min(MAX_BATCH_SIZE, len(texts))
    texts_list = texts[:MAX_BATCH_SIZE]

    embeddings = embed_texts(MODEL, texts_list, batch_size=batch)
    response = []
    for embedding, text in zip(embeddings, texts_list):
        hits = INDEX.search(embedding, k=top_k)
        results = []
        for rowid, distance in hits:
            item = DB.get_item(rowid)
            if item is None:
                continue
            results.append({
                "hash": item["hash"],
                "code": item["code"],
                "lang": item["lang"],
                "distance": distance,
            })
        response.append({
            "text": text,
            "embedding": embedding.tolist(),
            "results": results,
        })
    return response


UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>codeRAG explorer</title>
<link rel="stylesheet" href="/static/github-dark.min.css">
<script src="/static/highlight.min.js"></script>
<style>
:root{--bg:#0b0f14;--panel:#10151c;--panel2:#121923;--border:#1c2431;--text:#e6edf3;--muted:#7d8ea1;--accent:#4c9aff;--green:#3fb950;--orange:#d29922;--red:#f85149}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:14px;padding:12px 20px;border-bottom:1px solid var(--border);background:var(--panel)}
header .logo{font-size:15px;font-weight:700;letter-spacing:.5px}
header .logo span{color:var(--accent)}
header .sub{color:var(--muted);font-size:11px}
header .controls{margin-left:auto;display:flex;gap:8px;align-items:center}
label{color:var(--muted);font-size:11px}
input[type=number]{width:64px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:6px 8px;font:inherit;font-size:12px;outline:none}
button{background:var(--accent);color:#081018;border:none;border-radius:6px;padding:7px 14px;font:inherit;font-size:12px;font-weight:700;cursor:pointer}
button:hover{filter:brightness(1.15)}
main{flex:1;display:flex;min-height:0}
.pane{flex:1;display:flex;flex-direction:column;min-width:0}
.pane.left{flex:0 0 42%;border-right:1px solid var(--border)}
.pane-head{padding:10px 16px;color:var(--muted);font-size:11px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.pane-head .count{color:var(--accent);text-transform:none}
textarea{flex:1;width:100%;resize:none;background:var(--panel);color:var(--text);border:none;outline:none;padding:16px;font:inherit;font-size:13px;line-height:1.55}
.hint{padding:8px 16px;color:var(--muted);font-size:11px;border-top:1px solid var(--border)}
#results{flex:1;overflow-y:auto;padding:12px}
.card{margin-bottom:12px;background:var(--panel2);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.card-head{display:flex;gap:10px;align-items:center;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--border);font-size:11px}
.rank{background:var(--accent);color:#081018;font-weight:700;border-radius:4px;padding:1px 7px}
.lang{color:var(--green)}
.dist{color:var(--orange)}
.hash{color:var(--muted)}
.copy{margin-left:auto;background:transparent;color:var(--muted);border:1px solid var(--border);padding:2px 10px;border-radius:4px;font-size:10px;cursor:pointer}
.copy:hover{color:var(--text)}
.card-body{max-height:38vh;overflow:auto;padding:10px 12px}
pre{font-size:12px;line-height:1.5;white-space:pre}
code{background:transparent!important;padding:0!important}
.empty{padding:24px;color:var(--muted);text-align:center;font-size:12px}
footer{padding:8px 20px;color:var(--muted);font-size:11px;border-top:1px solid var(--border)}
footer .ok{color:var(--green)}
footer .err{color:var(--red)}
</style>
</head>
<body>
<header>
  <div class="logo">code<span>RAG</span> explorer</div>
  <div class="sub">semantic search over the code index</div>
  <div class="controls">
    <label>top N</label>
    <input id="n" type="number" min="1" max="50" value="5">
    <button id="go">Search</button>
  </div>
</header>
<main>
  <section class="pane left">
    <div class="pane-head">query</div>
    <textarea id="q" spellcheck="false" placeholder="paste a code snippet..."></textarea>
    <div class="hint">Ctrl + Enter to search</div>
  </section>
  <section class="pane">
    <div class="pane-head">results <span class="count" id="count"></span></div>
    <div id="results"><div class="empty">Search something to explore the index.</div></div>
  </section>
</main>
<footer><span id="status" class="ok">idle</span></footer>
<script>
const LANG={python:'python',javascript:'javascript',typescript:'typescript',java:'java',c:'c',cpp:'cpp',csharp:'csharp',go:'go',golang:'go',rust:'rust',ruby:'ruby',php:'php',swift:'swift',kotlin:'kotlin',bash:'bash',shell:'bash',sql:'sql',json:'json',html:'xml',xml:'xml',css:'css',yaml:'yaml',markdown:'markdown',lua:'lua',scala:'scala',perl:'perl',objc:'objectivec',dart:'dart',fortran:'fortran',erlang:'erlang',julia:'julia',crystal:'crystal',ocaml:'ocaml',haxe:'haxe',verilog:'verilog',systemverilog:'verilog',solidity:'solidity',nim:'nim',r:'r',zig:'zig',tsx:'tsx',glsl:'glsl',hlsl:'hlsl'};
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function highlight(lang,code){
  const name=LANG[lang];
  try{
    if(window.hljs&&name&&hljs.getLanguage(name))return hljs.highlight(code,{language:name}).value;
    if(window.hljs){const a=hljs.highlightAuto(code);if(a.language)return a.value}
  }catch(e){}
  return esc(code);
}
function setStatus(msg,ok){const s=document.getElementById('status');s.textContent=msg;s.className=ok?'ok':'err'}
async function search(){
  const q=document.getElementById('q').value;
  const n=parseInt(document.getElementById('n').value)||5;
  if(!q.trim())return;
  const go=document.getElementById('go');go.textContent='...';go.disabled=true;
  setStatus('searching...',true);
  try{
    const res=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({texts:[q],top_k:n})});
    const data=await res.json();
    const item=data[0]||{results:[]};
    const box=document.getElementById('results');box.innerHTML='';
    document.getElementById('count').textContent=item.results.length?item.results.length+' hits':'';
    if(!item.results.length){box.innerHTML='<div class="empty">no results</div>';setStatus('no results',true);return}
    item.results.forEach((r,i)=>{
      const card=document.createElement('div');card.className='card';
      const head=document.createElement('div');head.className='card-head';
      head.innerHTML='<span class="rank">#'+(i+1)+'</span><span class="lang">'+esc(r.lang)+'</span><span class="dist">d='+r.distance.toFixed(3)+'</span><span class="hash">'+esc(r.hash.slice(0,12))+'</span>';
      const cp=document.createElement('button');cp.className='copy';cp.textContent='copy';
      cp.onclick=()=>{try{navigator.clipboard.writeText(r.code)}catch(e){}document.getElementById('q').value=r.code};
      head.appendChild(cp);
      const body=document.createElement('div');body.className='card-body';
      body.innerHTML='<pre><code>'+highlight(r.lang,r.code)+'</code></pre>';
      card.appendChild(head);card.appendChild(body);box.appendChild(card);
    });
    setStatus(item.results.length+' results',true);
  }catch(e){setStatus('error: '+e,false)}
  finally{go.textContent='Search';go.disabled=false}
}
document.getElementById('go').onclick=search;
document.getElementById('q').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')search()});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


@app.get("/health")
def health():
    return {"status": "ok", "gpu_index": GPU_INDEX}


if __name__ == "__main__":
    MODEL, GPU_INDEX = build()
    DB = CodeDB(DB_PATH)
    INDEX = InMemoryIndex(DB.conn)

    uvicorn.run(app, host="0.0.0.0", port=8234)

    