/* Shared behaviour for the plain-language and technical pages: tabs, Kaplan-Meier and forest charts with hover,
   verdict table with filters, figure lightbox. Data arrives via window.GENIE_DATA (data.js). */
const Site = (() => {
  const D = () => window.GENIE_DATA;
  const PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
  const PAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2fb52f", "#9085e9", "#e66767"];
  const isDark = () => document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const pal = () => (isDark() ? PAL_DARK : PAL);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtP = (p) => (p == null ? "n/a" : p < 0.001 ? p.toExponential(1).replace("e-", "e−") : p.toFixed(3));
  const fmtMo = (m) => (m == null ? "not reached" : m >= 24 ? `${(m / 12).toFixed(1)} y` : `${m.toFixed(1)} mo`);
  const LAY = { holds: "Holds up", illusion: "An illusion", unclear: "No clear answer", descriptive: "Description" };
  const VCLASS = { replicated: "replicated", signal: "signal", artefact: "artefact", null: "no signal", descriptive: "descriptive" };

  // ---------------------------------------------------------------- Kaplan-Meier
  function survAt(g, t) {
    const T = g.t, S = g.s;
    if (t < T[0]) return 1;
    let lo = 0, hi = T.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (T[mid] <= t) lo = mid; else hi = mid - 1; }
    return S[lo];
  }

  function renderKM(container, specs) {
    const keys = specs.filter((s) => D().curves[s[0]]);
    if (!keys.length) { container.remove(); return; }
    let current = 0;
    const visible = {};
    const seg = document.createElement("div");
    if (keys.length > 1) {
      seg.className = "seg"; seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "Index date");
      keys.forEach((k, i) => {
        const b = document.createElement("button"); b.type = "button"; b.textContent = k[1] || k[0]; b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
        b.addEventListener("click", () => { current = i; seg.querySelectorAll("button").forEach((x, j) => x.setAttribute("aria-pressed", j === i ? "true" : "false")); draw(); });
        seg.appendChild(b);
      });
      container.appendChild(seg);
    }
    const title = document.createElement("div"); title.className = "ctitle"; container.appendChild(title);
    const legend = document.createElement("ul"); legend.className = "legend"; container.appendChild(legend);
    const host = document.createElement("div"); host.className = "cwrap"; container.appendChild(host);
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; container.appendChild(tip);

    function draw() {
      const set = D().curves[keys[current][0]];
      const colors = pal();
      title.textContent = set.title;
      legend.innerHTML = "";
      set.groups.forEach((g, i) => {
        if (visible[i] === undefined) visible[i] = true;
        const li = document.createElement("li");
        const b = document.createElement("button"); b.type = "button"; b.className = visible[i] ? "" : "off"; b.setAttribute("aria-pressed", visible[i] ? "true" : "false");
        b.innerHTML = `<span class="sw" style="background:${colors[i % colors.length]}"></span>${esc(g.name)} <span class="small">(n=${g.n}, median ${esc(fmtMo(g.median))})</span>`;
        b.addEventListener("click", () => { visible[i] = !visible[i]; draw(); });
        li.appendChild(b); legend.appendChild(li);
      });
      const W = 760, ML = 132, MR = 14, MT = 14, PH = 300, AX = 42, RH = 16;
      const riskH = 22 + RH * set.groups.length;
      const H = MT + PH + AX + riskH;
      const x = (t) => ML + (t / set.xmax) * (W - ML - MR);
      const y = (s) => MT + (1 - s) * PH;
      let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(set.title)}">`;
      svg += `<g class="grid">` + [0.25, 0.5, 0.75].map((s) => `<line x1="${ML}" x2="${W - MR}" y1="${y(s)}" y2="${y(s)}"/>`).join("") + `</g>`;
      svg += `<g class="axis"><line x1="${ML}" x2="${ML}" y1="${MT}" y2="${MT + PH}"/><line x1="${ML}" x2="${W - MR}" y1="${MT + PH}" y2="${MT + PH}"/></g>`;
      [0, 0.25, 0.5, 0.75, 1].forEach((s) => { svg += `<text x="${ML - 6}" y="${y(s) + 4}" text-anchor="end">${Math.round(s * 100)}%</text>`; });
      set.ticks.forEach((t) => { svg += `<text x="${x(t)}" y="${MT + PH + 16}" text-anchor="middle">${t}</text>`; });
      svg += `<text x="${(ML + W - MR) / 2}" y="${MT + PH + 34}" text-anchor="middle">${esc(set.xlabel)}</text>`;
      svg += `<text class="plabel" x="${W - MR}" y="${MT + 12}" text-anchor="end">log-rank p = ${fmtP(set.p)}</text>`;
      set.groups.forEach((g, i) => {
        if (!visible[i]) return;
        const c = colors[i % colors.length];
        let d = `M${x(0)},${y(1)}`;
        for (let k = 0; k < g.t.length; k++) d += `H${x(Math.min(g.t[k], set.xmax)).toFixed(1)}V${y(g.s[k]).toFixed(1)}`;
        const tEnd = Math.min(set.xmax, Math.max(g.t[g.t.length - 1], g.c.length ? g.c[g.c.length - 1] : 0));
        d += `H${x(tEnd).toFixed(1)}`;
        svg += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2.2" stroke-linejoin="round"/>`;
        svg += `<g stroke="${c}" stroke-width="1.2" opacity="0.7">` + g.c.filter((t) => t <= set.xmax).map((t) => { const yy = y(survAt(g, t)); return `<line x1="${x(t).toFixed(1)}" x2="${x(t).toFixed(1)}" y1="${(yy - 4).toFixed(1)}" y2="${(yy + 4).toFixed(1)}"/>`; }).join("") + `</g>`;
      });
      // risk table
      svg += `<g class="risk"><text class="rname" x="${ML - 6}" y="${MT + PH + AX + 8}" text-anchor="end">At risk</text>`;
      set.groups.forEach((g, i) => {
        if (!visible[i]) return;
        const yy = MT + PH + AX + 8 + RH * (i + 1);
        const nm = g.name.length > 22 ? g.name.slice(0, 21) + "…" : g.name;
        svg += `<text class="rname" x="${ML - 6}" y="${yy}" text-anchor="end" fill="${colors[i % colors.length]}">${esc(nm)}</text>`;
        set.ticks.forEach((t, k) => { svg += `<text x="${x(t)}" y="${yy}" text-anchor="middle">${g.ar[k]}</text>`; });
      });
      svg += `</g>`;
      svg += `<line class="cross" x1="0" x2="0" y1="${MT}" y2="${MT + PH}" stroke="var(--muted)" stroke-dasharray="3 3" visibility="hidden"/>`;
      svg += `<rect class="hit" x="${ML}" y="${MT}" width="${W - ML - MR}" height="${PH}" fill="transparent"/></svg>`;
      host.innerHTML = svg;
      const el = host.querySelector("svg"), cross = el.querySelector(".cross"), hit = el.querySelector(".hit");
      const move = (ev) => {
        const r = el.getBoundingClientRect();
        const px = ((ev.clientX - r.left) / r.width) * W;
        const t = Math.max(0, Math.min(set.xmax, ((px - ML) / (W - ML - MR)) * set.xmax));
        cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("visibility", "visible");
        const step = set.ticks.length > 1 ? set.ticks[1] - set.ticks[0] : 12;
        const k = Math.min(set.ticks.length - 1, Math.floor(t / step));
        let html = `<div><strong>Month ${t.toFixed(0)}</strong></div>`;
        set.groups.forEach((g, i) => { if (!visible[i]) return; html += `<div><span style="color:${colors[i % colors.length]}">●</span> ${esc(g.name)}: <strong>${Math.round(survAt(g, t) * 100)}%</strong> alive · ${g.ar[k]} at risk</div>`; });
        tip.innerHTML = html; tip.hidden = false;
        const cr = container.getBoundingClientRect();
        let left = ev.clientX - cr.left + 14, top = ev.clientY - cr.top + 12;
        if (left + 270 > cr.width) left = Math.max(0, ev.clientX - cr.left - 280);
        tip.style.left = left + "px"; tip.style.top = top + "px";
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerdown", move);
      hit.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    }
    draw();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  }

  // ---------------------------------------------------------------- forest plot
  function renderForest(container, key) {
    const f = D().forests[key];
    if (!f) { container.remove(); return; }
    const colors = pal();
    const title = document.createElement("div"); title.className = "ctitle"; title.textContent = f.title; container.appendChild(title);
    const host = document.createElement("div"); host.className = "cwrap"; container.appendChild(host);
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; container.appendChild(tip);
    const rows = f.rows;
    const W = 760, LW = 250, RW = 150, RH = 26, MT = 10, AX = 40;
    const H = MT + RH * rows.length + AX;
    const lo = Math.max(0.05, Math.min(...rows.map((r) => r.lo ?? r.HR)) / 1.3), hi = Math.min(60, Math.max(...rows.map((r) => r.hi ?? r.HR)) * 1.3);
    const x = (v) => LW + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (W - LW - RW);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(f.title)}">`;
    const ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32].filter((t) => t >= lo && t <= hi);
    svg += `<g class="grid">` + ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${MT}" y2="${MT + RH * rows.length}"/>`).join("") + `</g>`;
    svg += `<line class="ref" x1="${x(1)}" x2="${x(1)}" y1="${MT}" y2="${MT + RH * rows.length}"/>`;
    ticks.forEach((t) => { svg += `<text x="${x(t)}" y="${MT + RH * rows.length + 16}" text-anchor="middle">${t}</text>`; });
    svg += `<text x="${LW + (W - LW - RW) / 2}" y="${MT + RH * rows.length + 34}" text-anchor="middle">hazard ratio (altered vs not), 95% CI · log scale</text>`;
    rows.forEach((r, i) => {
      const yy = MT + RH * i + RH / 2, sig = r.q != null && r.q < 0.1;
      const c = sig ? colors[0] : "var(--muted)";
      svg += `<g class="frow" data-i="${i}"><rect class="hit" x="0" y="${MT + RH * i}" width="${W}" height="${RH}" fill="transparent"/>`;
      svg += `<text class="${sig ? "sig" : ""}" x="${LW - 10}" y="${yy + 4}" text-anchor="end">${esc(r.feature)} (n=${r.n_alt})</text>`;
      if (r.lo != null && r.hi != null) svg += `<line x1="${x(Math.max(lo, r.lo))}" x2="${x(Math.min(hi, r.hi))}" y1="${yy}" y2="${yy}" stroke="${c}" stroke-width="2"/>`;
      svg += `<circle cx="${x(Math.min(hi, Math.max(lo, r.HR)))}" cy="${yy}" r="5" fill="${c}" stroke="var(--surface)" stroke-width="1.5"/>`;
      svg += `<text x="${W - RW + 10}" y="${yy + 4}">p=${fmtP(r.p)}${r.q != null ? "  q=" + fmtP(r.q) : ""}</text></g>`;
    });
    svg += `</svg>`;
    host.innerHTML = svg;
    const el = host.querySelector("svg");
    el.querySelectorAll(".frow").forEach((g) => {
      const r = rows[+g.dataset.i];
      const show = (ev) => {
        tip.innerHTML = `<div><strong>${esc(r.feature)}</strong></div><div>altered n=${r.n_alt}${r.n_wt != null ? ", not altered n=" + r.n_wt : ""}</div>` +
          (r.median_alt !== undefined ? `<div>median ${esc(fmtMo(r.median_alt))} vs ${esc(fmtMo(r.median_wt))}</div>` : "") +
          `<div>HR ${r.HR?.toFixed(2)} [${r.lo?.toFixed(2)}–${r.hi?.toFixed(2)}]</div><div>log-rank p ${fmtP(r.p)}${r.q != null ? ", FDR q " + fmtP(r.q) : ""}</div>`;
        tip.hidden = false;
        const cr = container.getBoundingClientRect();
        let left = ev.clientX - cr.left + 14; if (left + 270 > cr.width) left = Math.max(0, ev.clientX - cr.left - 280);
        tip.style.left = left + "px"; tip.style.top = (ev.clientY - cr.top + 12) + "px";
      };
      g.addEventListener("pointermove", show); g.addEventListener("pointerdown", show);
      g.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
  }

  // ---------------------------------------------------------------- lightbox
  function figThumbs(ds, q) {
    if (!q.figures.length) return "";
    return `<div class="figs">` + q.figures.map((f) => `<button type="button" data-src="figures/${ds.id}/${esc(f)}" data-cap="${esc(f)}"><img loading="lazy" src="figures/${ds.id}/${esc(f)}" alt="${esc(f)}"><span class="cap">${esc(f.replace(/\.png$/, "").replace(/^[A-Za-z0-9]+_/, "").replace(/_/g, " "))}</span></button>`).join("") + `</div>`;
  }
  function initLightbox() {
    const lb = document.getElementById("lightbox"); if (!lb) return;
    const img = lb.querySelector("img"), cap = lb.querySelector(".lb-cap");
    document.addEventListener("click", (ev) => {
      const b = ev.target.closest(".figs button");
      if (b) { img.src = b.dataset.src; img.alt = b.dataset.cap; cap.textContent = b.dataset.cap; lb.hidden = false; return; }
      if (ev.target === lb || ev.target.closest(".lb-close")) lb.hidden = true;
    });
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") lb.hidden = true; });
  }

  // ---------------------------------------------------------------- page assembly
  function chartBlock(specs) { const d = document.createElement("div"); d.className = "chart"; renderKM(d, specs); return d; }
  function forestBlock(key) { const d = document.createElement("div"); d.className = "chart"; renderForest(d, key); return d; }

  function layCard(ds, q) {
    const card = document.createElement("article"); card.className = "card"; card.id = "q-" + q.id;
    card.innerHTML = `<h3><span class="qid">${esc(q.id)}</span>${esc(q.text)} <span class="chip ${q.lay_class}">${LAY[q.lay_class]}</span></h3><p>${esc(q.lay)}</p>`;
    if (q.charts.length) card.appendChild(chartBlock(q.charts));
    const det = document.createElement("details");
    det.innerHTML = `<summary>The numbers</summary><p>${esc(q.note)}</p>`;
    card.appendChild(det);
    return card;
  }

  function techCard(ds, q) {
    const card = document.createElement("article"); card.className = "card"; card.id = "q-" + q.id;
    card.innerHTML = `<h3><span class="qid">${esc(q.id)}</span>${esc(q.text)} <span class="chip ${q.vclass}">${esc(q.verdict)}</span></h3>` +
      `<div class="compare">${esc(q.compare)}</div><div class="note">${esc(q.note)}</div>` + (q.take ? `<p class="take">${esc(q.take)}</p>` : "");
    if (q.charts.length) card.appendChild(chartBlock(q.charts));
    if (q.forest) card.appendChild(forestBlock(q.forest));
    card.insertAdjacentHTML("beforeend", figThumbs(ds, q));
    return card;
  }

  function buildTabs(lay) {
    const tabs = document.getElementById("tabs"), panels = document.getElementById("panels");
    if (!tabs) return;
    const ds = D().datasets;
    ds.forEach((d, i) => {
      const b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "tab"); b.id = "tab-" + d.id; b.textContent = d.tab;
      b.addEventListener("click", () => select(d.id, true)); tabs.appendChild(b);
      const p = document.createElement("div"); p.className = "panel"; p.id = "panel-" + d.id; p.setAttribute("role", "tabpanel"); p.hidden = true;
      const nb = `${D().repo}/blob/${D().branch}/notebooks/${d.nb}.ipynb`, res = `${D().repo}/tree/${D().branch}/results/${d.id}`;
      p.innerHTML = `<div class="intro"><h3>${esc(d.name)}</h3><div class="meta">${esc(d.source)} · ${esc(d.cohort)}</div><p>${esc(lay ? d.lay : d.expert)}</p>` +
        (lay ? "" : `<div class="links"><a href="${nb}">notebook</a><a href="${res}">results folder</a></div>`) + `</div>`;
      const cards = document.createElement("div"); cards.className = "cards";
      d.questions.forEach((q) => cards.appendChild(lay ? layCard(d, q) : techCard(d, q)));
      p.appendChild(cards); panels.appendChild(p);
    });
    function select(id, push) {
      ds.forEach((d) => { document.getElementById("tab-" + d.id).setAttribute("aria-selected", d.id === id ? "true" : "false"); document.getElementById("panel-" + d.id).hidden = d.id !== id; });
      if (push) history.replaceState(null, "", "#" + id);
    }
    const h = location.hash.replace("#", "");
    const target = h.startsWith("q-") ? ds.find((d) => d.questions.some((q) => "q-" + q.id === h)) : ds.find((d) => d.id === h);
    select((target || ds[0]).id, false);
    if (h.startsWith("q-")) setTimeout(() => document.getElementById(h)?.scrollIntoView({ block: "start" }), 50);
  }

  function buildTiles(lay) {
    const t = document.getElementById("tiles"); if (!t) return;
    if (lay) {
      const c = D().counts;
      t.innerHTML = Object.entries(LAY).map(([k, lab]) => `<div class="tile ${k}"><div class="num">${c[k] || 0}</div><div class="lab">${lab}</div></div>`).join("");
    } else {
      const all = D().datasets.flatMap((d) => d.questions);
      const n = (k) => all.filter((q) => q.vclass === k).length;
      t.innerHTML = [["replicated", "replicated as stated"], ["signal", "signal at the pre-specified threshold"], ["artefact", "artefact of the index date"], ["null", "no signal"], ["descriptive", "descriptive or feasibility"]]
        .map(([k, lab]) => `<div class="tile ${k === "null" ? "unclear" : k === "artefact" ? "illusion" : k === "signal" ? "holds" : "descriptive"}"><div class="num">${n(k)}</div><div class="lab">${lab}</div></div>`).join("");
    }
  }

  function buildTable() {
    const table = document.getElementById("vtable"), filters = document.getElementById("filters");
    if (!table) return;
    const rows = D().datasets.flatMap((d) => d.questions.map((q) => ({ ds: d, q })));
    const state = { ds: "", classes: new Set(), text: "" };
    const sel = document.createElement("select"); sel.id = "f-ds"; sel.innerHTML = `<option value="">All datasets</option>` + D().datasets.map((d) => `<option value="${d.id}">${esc(d.tab)}</option>`).join("");
    sel.addEventListener("change", () => { state.ds = sel.value; render(); }); filters.appendChild(sel);
    Object.entries(VCLASS).forEach(([k, lab]) => {
      const b = document.createElement("button"); b.type = "button"; b.textContent = lab; b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => { if (state.classes.has(k)) state.classes.delete(k); else state.classes.add(k); b.setAttribute("aria-pressed", state.classes.has(k) ? "true" : "false"); render(); });
      filters.appendChild(b);
    });
    const inp = document.createElement("input"); inp.type = "search"; inp.id = "f-text"; inp.placeholder = "search (gene, drug, question)"; inp.addEventListener("input", () => { state.text = inp.value.toLowerCase(); render(); }); filters.appendChild(inp);
    const tb = table.querySelector("tbody");
    function render() {
      const vis = rows.filter(({ ds, q }) => (!state.ds || ds.id === state.ds) && (!state.classes.size || state.classes.has(q.vclass)) &&
        (!state.text || (q.id + " " + q.text + " " + q.note + " " + q.verdict).toLowerCase().includes(state.text)));
      tb.innerHTML = vis.map(({ ds, q }) => `<tr><td>${esc(ds.tab)}</td><td class="q"><a href="#q-${q.id}" data-ds="${ds.id}">${q.id}</a></td><td><span class="chip ${q.vclass}">${esc(q.verdict)}</span></td><td class="res">${esc(q.note)}</td></tr>`).join("") ||
        `<tr><td colspan="4">No questions match.</td></tr>`;
    }
    tb.addEventListener("click", (ev) => {
      const a = ev.target.closest("a[data-ds]"); if (!a) return;
      ev.preventDefault();
      document.getElementById("tab-" + a.dataset.ds)?.click();
      history.replaceState(null, "", a.getAttribute("href"));
      document.getElementById(a.getAttribute("href").slice(1))?.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    render();
  }

  function init() {
    const lay = window.GENIE_PAGE === "lay";
    buildTiles(lay);
    document.querySelectorAll("[data-km]").forEach((el) => renderKM(el, JSON.parse(el.dataset.km)));
    buildTabs(lay);
    if (!lay) buildTable();
    initLightbox();
  }
  return { init };
})();
