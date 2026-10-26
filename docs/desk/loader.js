// Runs the Markout desk inside this tab. Pyodide (CPython compiled to WebAssembly) loads the
// desk's own Python modules from markout-desk.zip, and every /api/ request the page makes is
// answered by markout.cup.browser.BrowserDesk instead of the network. Nothing leaves the tab.
(() => {
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
  const raw = window.fetch.bind(window);

  const boot = document.createElement("div");
  boot.setAttribute("role", "status");
  boot.style.cssText = "position:fixed;inset:0;z-index:9;display:grid;place-items:center;padding:16px;" +
    "background:color-mix(in srgb, var(--page) 88%, transparent)";
  boot.innerHTML = '<div class="card" style="max-width:440px;margin:0">' +
    '<div class="title">Starting the desk in your browser</div>' +
    '<p class="sub" id="boot-step" style="margin:8px 0">Loading Python (Pyodide)…</p>' +
    '<p class="muted" style="margin:0;font-size:12px">The first visit downloads about 20 MB, which your browser ' +
    "then keeps. Everything runs on this device; nothing you do is sent anywhere.</p></div>";
  document.body.appendChild(boot);
  const step = (text) => { const el = document.getElementById("boot-step"); if (el) el.textContent = text; };

  const script = (src) => new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(s);
  });

  const ready = (async () => {
    await script(PYODIDE + "pyodide.js");
    step("Starting Python…");
    const py = await loadPyodide({ indexURL: PYODIDE });
    step("Loading numpy and pandas…");
    await py.loadPackage(["numpy", "pandas"]);
    step("Opening your sandbox…");
    const bundle = await raw("markout-desk.zip", { cache: "no-cache" });
    if (!bundle.ok) throw new Error(`markout-desk.zip: HTTP ${bundle.status}`);
    py.unpackArchive(await bundle.arrayBuffer(), "zip", { extractDir: "/app" });
    py.runPython("import sys\nsys.path.insert(0, '/app/src')\nfrom markout.cup.browser import BrowserDesk\ndesk = BrowserDesk()");
    const desk = py.globals.get("desk");
    const sbx = document.getElementById("sbx");
    if (sbx) sbx.textContent = `Sandbox seed ${desk.seed}.`;
    boot.remove();
    return desk;
  })();
  ready.catch((e) => {
    step(`The desk could not start in this browser (${e.message}).`);
    const card = boot.querySelector(".card");
    if (card) card.insertAdjacentHTML("beforeend",
      '<p class="sub" style="margin:10px 0 0"><a href="../demo/desk.html">Open the read-only snapshot</a> or ' +
      '<a href="../">go back to Markout</a>.</p>');
  });

  // The page's own script calls fetch('/api/...'); answer those here and pass anything else through.
  window.fetch = async (url, options = {}) => {
    if (!String(url).startsWith("/api/")) return raw(url, options);
    const desk = await ready;
    const body = typeof options.body === "string" ? options.body : "";
    const reply = JSON.parse(desk.handle(options.method || "GET", String(url), body));
    return new Response(JSON.stringify(reply.body), { status: reply.status, headers: { "Content-Type": "application/json" } });
  };

  // The time controls, the same buttons as on the hosted desk. post() and markets() belong to the page.
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-time]");
    if (!b) return;
    if (b.dataset.time === "reset") {
      if (confirm("Start over with a fresh sandbox?")) post("/api/reset").then(() => markets());
    } else {
      post("/api/advance", { hours: Number(b.dataset.time) }).then(() => markets());
    }
  });
})();
