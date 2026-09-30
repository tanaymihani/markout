"""The approval desk: a local web page (standard library only) to review and approve trades.

Security: binds to 127.0.0.1 only; every POST needs the per-run token embedded in the
page (X-Markout-Token), and requests whose Host or Origin is not this machine are refused
(protects against other web pages and DNS rebinding).
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from markout.cup import risk
from markout.cup.service import Bot

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="markout-token" content="__TOKEN__">
<title>Markout Desk</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--accent:#2a78d6;--accent-ink:#fff;--good:#006300;--bad:#d03b3b;--warn-bg:#fff4d6;--warn-ink:#6b4a00}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5;--accent-ink:#fff;--good:#0ca30c;--bad:#e66767;--warn-bg:#3a3000;--warn-ink:#fad27a}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0;font-weight:650}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:0 0 10px;font-weight:600}
.muted{color:var(--muted)}.ink2{color:var(--ink2)}main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:18px;padding:18px 20px;max-width:1400px;margin:0 auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.row{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline}.stat b{font-variant-numeric:tabular-nums;font-size:15px}
.stat span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.chip{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--border);border-radius:999px;padding:1px 9px;font-size:12px;color:var(--ink2)}
.flag{background:var(--warn-bg);color:var(--warn-ink);border-color:transparent}
button{font:inherit;border-radius:7px;border:1px solid var(--border);background:var(--surface);color:var(--ink);padding:6px 12px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:transparent;font-weight:600}
button.danger{color:var(--bad)}button:disabled{opacity:.45;cursor:not-allowed}
input[type=number]{font:inherit;width:110px;padding:5px 8px;border-radius:7px;border:1px solid var(--border);background:var(--page);color:var(--ink);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--grid);font-size:13px}
th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
a{color:var(--accent)}details summary{cursor:pointer;color:var(--ink2);font-size:13px}.pos{color:var(--good)}.neg{color:var(--bad)}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--page);padding:8px 14px;border-radius:8px;opacity:0;transition:opacity .2s}
.toast.on{opacity:.95}.title{font-weight:600;font-size:15px}.sub{font-size:13px;color:var(--ink2)}
</style></head><body>
<header><div><h1>Markout Desk · Predictions Cup</h1><div class="muted" id="meta">loading…</div></div>
<div class="row"><span class="chip" id="mode"></span>
<button id="modeBtn" title="Switch sizing mode">Switch mode</button>
<button id="refreshBtn">Refresh now</button>
<button id="killBtn" class="danger"></button></div></header>
<main><section><h2>Proposals awaiting your decision</h2><div id="pending"></div></section>
<aside><div class="card"><h2>Account</h2><div class="row" id="acct"></div><svg id="spark" width="100%" height="44" viewBox="0 0 300 44" preserveAspectRatio="none" aria-label="Equity over time"></svg></div>
<div class="card"><h2>Positions</h2><div id="positions"></div></div>
<div class="card"><h2>Recent decisions</h2><div id="recent"></div></div>
<div class="card"><h2>How to read a proposal</h2><div class="sub">p is the pooled probability from the contest price and the references
(weights shown); the band must clear the ask. Kelly is the growth-optimal fraction of equity. Mode <b>prize</b> is the policy
pre-registered in report 05 (all-in on the best edge while outside the top 3; it busts in most simulated contests);
<b>steady</b> is 0.5× Kelly with a 25% cap. Approving a stake of 0 logs the forecast without trading.
Nothing trades without your click. Kill switch: <code>touch data/cup/STOP</code>.</div></div></aside></main>
<div class="toast" id="toast"></div>
<script>
const TOKEN=document.querySelector('meta[name=markout-token]').content;
const $=s=>document.querySelector(s);const fmt=(x,d=2)=>x==null?'–':Number(x).toFixed(d);const pct=x=>x==null?'–':(100*x).toFixed(1)+'%';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const safeUrl=u=>/^https?:\/\//i.test(String(u||''))?esc(u):'';
function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('on');setTimeout(()=>t.classList.remove('on'),2600)}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Markout-Token':TOKEN},body:JSON.stringify(body||{})});
const j=await r.json().catch(()=>({ok:false,message:'bad response'}));toast(j.message||(j.ok?'done':'failed'));load();return j}
function spark(snaps){const s=$('#spark');if(!snaps.length){s.innerHTML='';return}const ys=snaps.map(x=>x.equity);const lo=Math.min(...ys),hi=Math.max(...ys);
const n=ys.length,pts=ys.map((y,i)=>`${(i/(Math.max(n-1,1)))*300},${40-(hi>lo?(y-lo)/(hi-lo)*36:18)}`).join(' ');
s.innerHTML=`<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" points="${pts}"/>`}
function proposal(p){const flags=(p.flags||[]).map(f=>`<span class="chip flag">⚠ ${esc(f)}</span>`).join(' ');
const comps=(p.components||[]).map(c=>`<tr><td>${esc(c.label)}${safeUrl(c.url)?` · <a href="${safeUrl(c.url)}" target="_blank" rel="noopener noreferrer">source</a>`:''}</td><td>${fmt(c.p)}</td><td>${fmt(c.w,1)}</td><td class="muted">${esc(c.match||c.note||'')}</td></tr>`).join('');
const maps=(p.mappings||[]).filter(m=>m.status!=='rejected').map(m=>`<tr><td>${esc(m.source)}: ${esc(m.question)} → <b>${esc(m.outcome)}</b>${m.invert?' (inverted)':''}</td><td>${fmt(m.score)}</td>
<td>${m.status==='confirmed'?'<span class="chip">✓ confirmed</span>':`<button data-act="map" data-status="confirmed" data-cid="${esc(p.contract_id)}" data-key="${esc(m.key)}">Confirm</button> <button data-act="map" data-status="rejected" data-cid="${esc(p.contract_id)}" data-key="${esc(m.key)}">Reject</button>`}</td></tr>`).join('');
return `<div class="card"><div class="row" style="justify-content:space-between"><div><div class="title">${esc(p.market_title)}</div>
<div class="sub">Buy <b>${esc(p.contract_name)}</b> · <span class="chip">${esc(p.category)}</span> · mode ${esc(p.mode)}</div></div>${flags}</div>
<div class="row" style="margin:10px 0"><div class="stat"><span>probability</span><b>${fmt(p.p)}</b> <span class="muted" style="display:inline">[${fmt(p.lo)}, ${fmt(p.hi)}]</span></div>
<div class="stat"><span>ask</span><b>${fmt(p.ask)}</b></div><div class="stat"><span>edge</span><b class="pos">${p.edge>=0?'+':''}${fmt(p.edge)}</b></div>
<div class="stat"><span>Kelly</span><b>${pct(p.kelly)}</b></div><div class="stat"><span>stake</span><input type="number" min="0" step="1" data-stake="${esc(p.id)}" value="${fmt(p.stake,0)}"> <span class="muted" style="display:inline">${fmt(p.qty,0)} contracts</span></div></div>
<div class="row"><button class="primary" data-act="approve" data-id="${esc(p.id)}" ${p.mapping_confirmed?'':'disabled title="Confirm the reference mapping first"'}>Approve</button>
<button data-act="forecast" data-id="${esc(p.id)}" ${p.mapping_confirmed?'':'disabled'}>Log forecast only</button>
<button data-act="skip" data-id="${esc(p.id)}">Skip</button></div>
<details style="margin-top:10px"><summary>Sources and reasoning</summary><table><tr><th>source</th><th>p</th><th>weight</th><th>match</th></tr>${comps}</table>
${maps?`<table style="margin-top:8px"><tr><th>reference mapping</th><th>score</th><th></th></tr>${maps}</table>`:''}<p class="sub">${esc(p.rationale)}</p></details></div>`}
async function load(){let s;try{s=await (await fetch('/api/state')).json()}catch(e){$('#meta').textContent='desk unreachable';return}
$('#meta').textContent=`${s.adapter} adapter · now ${s.now.replace('T',' ').slice(0,16)} UTC · last refresh ${s.last_refresh?s.last_refresh.slice(11,19):'–'}${s.source_errors.length?' · source errors: '+s.source_errors.length:''}`;
$('#mode').textContent='sizing: '+s.mode;$('#modeBtn').onclick=()=>post('/api/mode',{mode:s.mode==='prize'?'steady':'prize'});
$('#killBtn').textContent=s.kill_switch?'Kill switch ON (resume)':'Kill switch';$('#killBtn').onclick=()=>post('/api/kill',{on:!s.kill_switch});
$('#refreshBtn').onclick=()=>post('/api/refresh');const a=s.account;
$('#acct').innerHTML=`<div class="stat"><span>equity</span><b>${fmt(a.equity,0)}</b></div><div class="stat"><span>cash</span><b>${fmt(a.cash,0)}</b></div>
<div class="stat"><span>rank</span><b>${a.rank??'–'}</b> <span class="muted" style="display:inline">of ${a.n_players??'–'}</span></div>`;spark(s.snapshots);
$('#pending').innerHTML=s.pending.length?s.pending.map(proposal).join(''):'<div class="card muted">No proposals right now. The desk refreshes on its own.</div>';
$('#positions').innerHTML=s.positions.length?`<table><tr><th>contract</th><th>qty</th><th>avg</th><th>mid</th><th>PnL</th></tr>${s.positions.map(p=>`<tr><td>${esc(p.market)} · ${esc(p.contract)}</td><td>${fmt(p.qty,0)}</td><td>${fmt(p.avg_price)}</td><td>${fmt(p.mid)}</td><td class="${p.pnl>=0?'pos':'neg'}">${fmt(p.pnl,0)}</td></tr>`).join('')}</table>`:'<div class="muted">None</div>';
$('#recent').innerHTML=s.recent.length?`<table><tr><th>when</th><th>market</th><th>decision</th><th>stake</th></tr>${s.recent.slice(0,12).map(p=>`<tr><td>${esc((p.decided||'').slice(5,16).replace('T',' '))}</td><td>${esc(p.market_title)}</td><td>${esc(p.status)}</td><td>${fmt(p.decided_stake,0)}</td></tr>`).join('')}</table>`:'<div class="muted">None yet</div>'}
document.addEventListener('click',e=>{const b=e.target.closest('button[data-act]');if(!b||b.disabled)return;const d=b.dataset;
if(d.act==='map')post('/api/mapping',{contract_id:d.cid,key:d.key,status:d.status});
else if(d.act==='approve'){const inp=[...document.querySelectorAll('input[data-stake]')].find(i=>i.dataset.stake===d.id);post('/api/approve',{id:d.id,stake:Number(inp?inp.value:0)})}
else if(d.act==='forecast')post('/api/approve',{id:d.id,stake:0});
else if(d.act==='skip')post('/api/skip',{id:d.id,note:prompt('Why skip? (optional)')||''})});
load();setInterval(load,5000);
</script></body></html>"""


class Desk:
    def __init__(self, bot: Bot, port: int = 8765, interval: float = 60.0):
        self.bot, self.port, self.interval = bot, port, interval
        self.token = secrets.token_urlsafe(24)
        self._stop = threading.Event()

    def loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.bot.refresh()
                self.bot.last_error = ""
            except Exception as e:  # noqa: BLE001 - keep the desk alive, show the error
                self.bot.last_error = f"{type(e).__name__}: {e}"
            self._stop.wait(self.interval)

    def handler(self):
        desk = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                return

            def _ok_host(self) -> bool:
                host = (self.headers.get("Host") or "").split(":")[0]
                origin = self.headers.get("Origin")
                if host not in ("127.0.0.1", "localhost"):
                    return False
                return origin is None or urlparse(origin).hostname in ("127.0.0.1", "localhost")

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200) -> None:
                self._send(code, json.dumps(obj, default=str).encode(), "application/json")

            def do_GET(self):
                if not self._ok_host():
                    return self._json({"ok": False, "message": "forbidden host"}, HTTPStatus.FORBIDDEN)
                if self.path in ("/", "/index.html"):
                    return self._send(200, PAGE.replace("__TOKEN__", desk.token).encode(), "text/html; charset=utf-8")
                if self.path == "/api/state":
                    return self._json(desk.bot.state())
                return self._json({"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self):
                if not self._ok_host() or self.headers.get("X-Markout-Token") != desk.token:
                    return self._json({"ok": False, "message": "forbidden"}, HTTPStatus.FORBIDDEN)
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._json({"ok": False, "message": "bad json"}, HTTPStatus.BAD_REQUEST)
                b = desk.bot
                routes = {
                    "/api/approve": lambda: b.approve(body["id"], body.get("stake"), body.get("note", "")),
                    "/api/skip": lambda: b.skip(body["id"], body.get("note", "")),
                    "/api/mapping": lambda: b.set_mapping(body["contract_id"], body["key"], body["status"]),
                    "/api/mode": lambda: b.set_mode(body["mode"]),
                    "/api/refresh": lambda: {"ok": True, "message": f"refreshed: {b.refresh()}"},
                    "/api/kill": lambda: desk.kill(bool(body.get("on"))),
                }
                if self.path not in routes:
                    return self._json({"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)
                try:
                    return self._json(routes[self.path]())
                except (KeyError, ValueError, AssertionError) as e:
                    return self._json({"ok": False, "message": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)

        return H

    def kill(self, on: bool) -> dict:
        risk.STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
        if on:
            risk.STOP_FILE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " kill switch from the desk\n")
        elif risk.STOP_FILE.exists():
            risk.STOP_FILE.unlink()
        return {"ok": True, "message": "kill switch on: no orders will be placed" if on else "kill switch off"}

    def serve(self, background_loop: bool = True) -> ThreadingHTTPServer:
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), self.handler())
        if background_loop:
            threading.Thread(target=self.loop, daemon=True).start()
        return srv

    def stop(self) -> None:
        self._stop.set()
