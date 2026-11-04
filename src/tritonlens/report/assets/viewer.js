/* tritonlens report viewer: vanilla JS, no dependencies. */
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("tl-data").textContent);
  const $ = (sel, el = document) => el.querySelector(sel);

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return el;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtLoc = (loc) => (loc ? `${loc.file.split(/[\\/]/).pop()}:${loc.line}` : "");
  const opsById = new Map(D.ops.map((o) => [o.id, o]));
  const tensorsByKey = new Map(D.tensors.map((t) => [t.key, t]));
  const findingsByOp = new Map();
  for (const f of D.findings) {
    if (f.op === null) continue;
    if (!findingsByOp.has(f.op)) findingsByOp.set(f.op, []);
    findingsByOp.get(f.op).push(f);
  }

  // ---- colors -------------------------------------------------------------------------
  const WARP_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295", "#fabfd2", "#b6992d",
    "#499894", "#79706e"];
  function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function shade(hex, f) {
    const [r, g, b] = hexToRgb(hex);
    const mix = (c) => Math.round(f >= 0 ? c + (255 - c) * f : c * (1 + f));
    return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
  }
  const warpColor = (w) => WARP_COLORS[w % WARP_COLORS.length];
  const hsl = (hue, s, l) => `hsl(${hue},${s}%,${l}%)`;
  const bankColor = (b) => hsl((b * 360) / 32, 62, b % 2 ? 44 : 58);
  function viridis(t) {
    const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // ---- tooltip ------------------------------------------------------------------------
  const tip = $("#tooltip");
  function showTip(ev, html) {
    tip.innerHTML = html;
    tip.style.display = "block";
    const pad = 14;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight) y = ev.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => (tip.style.display = "none");
  const kv = (pairs) => `<div class="kv">${pairs.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`).join("")}</div>`;

  function setupCanvas(canvas, w, hgt) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.ceil(w * dpr);
    canvas.height = Math.ceil(hgt * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = hgt + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // ---- grid geometry shared by layout / bank maps --------------------------------------
  function gridGeom(shape, maxW, maxH) {
    let rows, cols;
    if (shape.length === 1) {
      cols = Math.min(shape[0], 64);
      rows = Math.ceil(shape[0] / cols);
    } else {
      rows = shape[0];
      cols = shape[1];
    }
    let cell = Math.floor(Math.min(maxW / cols, maxH / rows));
    cell = Math.max(3, Math.min(cell, 26));
    return { rows, cols, cell, w: cols * cell + 1, h: rows * cell + 1,
      index: (r, c) => (shape.length === 1 ? r * cols + c : r * cols + c),
      coord: (i) => (shape.length === 1 ? [i] : [Math.floor(i / cols), i % cols]),
      rc: (coord) => (shape.length === 1 ? [Math.floor(coord[0] / cols), coord[0] % cols] : coord),
      numel: shape.length === 1 ? shape[0] : rows * cols };
  }

  // ---- header -------------------------------------------------------------------------
  function renderHeader() {
    const k = D.kernel;
    const counts = { error: 0, warning: 0, info: 0 };
    for (const f of D.findings) counts[f.severity] = (counts[f.severity] || 0) + 1;
    const chips = [
      ["target", k.target], ["warps", k.numWarps], ["threads/warp", k.threadsPerWarp],
      ["grid", k.grid ? k.grid.join("x") : "n/a"], ["stages", k.numStages],
      ["shared", k.sharedBytes !== null && k.sharedBytes !== undefined ? `${k.sharedBytes} B` : null],
      ["regs", k.registers], ["spills", k.spills],
    ].filter(([, v]) => v !== null && v !== undefined);
    const hdr = $("header");
    hdr.append(
      h("h1", {}, "tritonlens ", h("span", {}, "/ " + k.name)),
      h("div", { class: "chips" }, chips.map(([a, b]) => h("span", { class: "chip" }, a + " ", h("b", {}, String(b))))),
      h("div", { class: "sev-counts" },
        counts.error ? h("span", { class: "badge error" }, `${counts.error} error`) : null,
        counts.warning ? h("span", { class: "badge warning" }, `${counts.warning} warning`) : null,
        counts.info ? h("span", { class: "badge info" }, `${counts.info} info`) : null,
        !D.findings.length ? h("span", { class: "badge ok" }, "no findings") : null));
  }

  // ---- tabs ---------------------------------------------------------------------------
  const TABS = [["overview", "Overview"], ["layouts", "Register layouts"], ["shared", "Shared memory"],
    ["global", "Global memory"], ["tiling", "Program tiling"], ["ir", "IR & source"]];
  const rendered = new Set();
  function showTab(name) {
    for (const [id] of TABS) {
      $(`#tab-${id}`).classList.toggle("active", id === name);
      $(`nav button[data-tab=${id}]`).classList.toggle("active", id === name);
    }
    if (!rendered.has(name)) {
      rendered.add(name);
      RENDER[name]();
    }
    location.hash = name;
  }

  // ---- overview -----------------------------------------------------------------------
  function findingEl(f) {
    const links = [];
    if (f.op !== null) links.push(h("a", { onclick: () => focusOp(f.op) }, "IR"));
    const op = f.op !== null ? opsById.get(f.op) : null;
    if (op && D.opTensors[f.op]) links.push(h("a", { onclick: () => focusTensor(D.opTensors[f.op][0].key) }, "layout"));
    if (["TL003", "TL004"].includes(f.rule)) links.push(h("a", { onclick: () => focusShared(f.op) }, "banks"));
    if (["TL001", "TL002"].includes(f.rule)) links.push(h("a", { onclick: () => focusGlobal(f.op) }, "access"));
    return h("div", { class: `finding ${f.severity}` },
      h("div", { class: "head" },
        h("span", { class: `badge ${f.severity}` }, f.severity),
        h("span", { class: "rule" }, f.rule),
        h("b", {}, f.title),
        h("span", { class: "where" }, fmtLoc(f.loc) || (f.line ? `ttgir:${f.line}` : ""), " ",
          ...links.flatMap((l) => [" ", l]))),
      h("div", {}, f.message),
      f.fix ? h("div", { class: "fix" }, h("b", {}, "Fix: "), f.fix) : null);
  }

  function renderOverview() {
    const sec = $("#tab-overview");
    const k = D.kernel;
    const gl = D.global;
    const avgEff = gl.length ? gl.reduce((a, g) => a + g.efficiency, 0) / gl.length : null;
    const narrow = gl.filter((g) => g.vecBytes < 16).length;
    const sh = D.shared.accesses.filter((a) => !a.estimated);
    const worstBank = sh.length ? Math.max(...sh.map((a) => a.degree)) : null;
    const card = (key, val, sub) => h("div", { class: "card" }, h("div", { class: "k" }, key),
      h("div", { class: "v" }, String(val)), sub ? h("div", { class: "s" }, sub) : null);
    sec.append(h("div", { class: "cards" },
      card("Global accesses", gl.length, narrow ? `${narrow} narrower than 16 B/lane` : "all 16 B/lane"),
      card("Coalescing", avgEff === null ? "n/a" : `${Math.round(avgEff * 100)}%`, "mean sector efficiency"),
      card("Shared accesses", D.shared.accesses.length, `${D.shared.buffers.length} buffers`),
      card("Worst bank conflict", worstBank === null ? "n/a" : `${worstBank.toFixed(2)}x`, "explicit shared ops"),
      card("Programs", k.grid ? k.grid.reduce((a, b) => a * b, 1) : "n/a", k.grid ? `grid ${k.grid.join(" x ")}` : "no launch captured"),
      card("Threads / program", k.numWarps * k.threadsPerWarp, `${k.numWarps} warps`)));

    const left = h("div", { class: "panel" }, h("h2", {}, `Findings (${D.findings.length})`));
    if (!D.findings.length) left.append(h("div", { class: "empty" }, "No issues found."));
    for (const f of D.findings) left.append(findingEl(f));

    const right = h("div", { class: "panel" }, h("h2", {}, "Layouts in this kernel"));
    for (const [name, lay] of Object.entries(D.layouts)) {
      right.append(h("div", { class: "small", style: "margin-bottom:8px" },
        h("code", { class: "tk-lay" }, "#" + name), " ", h("span", { class: "muted" }, lay.kind),
        h("div", { class: "mono muted", style: "font-size:11.5px;word-break:break-all" }, lay.text)));
    }
    if (D.ptx) {
      right.append(h("h3", {}, "PTX memory instructions"));
      const tb = h("table", { class: "t" }, h("tr", {}, h("th", {}, "instruction"), h("th", {}, "count")));
      for (const [op, n] of Object.entries(D.ptx.histogram).sort((a, b) => b[1] - a[1])) {
        tb.append(h("tr", {}, h("td", { class: "mono" }, op), h("td", {}, String(n))));
      }
      if (D.ptx.mma) tb.append(h("tr", {}, h("td", { class: "mono" }, "mma.sync"), h("td", {}, String(D.ptx.mma))));
      right.append(tb);
    }
    sec.append(h("div", { class: "grid-ir" }, left, right));
  }

  // ---- register layouts ---------------------------------------------------------------
  const L = { key: null, compare: null, colorBy: "warp", pinned: null, overlay: null };

  function cellColor(t, i, mode) {
    const w = t.warp[i], l = t.lane[i], r = t.reg[i];
    if (mode === "warp") return shade(warpColor(w), ((l % 8) / 8) * 0.45 - 0.1);
    if (mode === "lane") return hsl((l * 360) / 32, 60, 52);
    if (mode === "register") return viridis(t.regsPerThread > 1 ? r / (t.regsPerThread - 1) : 0);
    const tid = w * 32 + l;
    return hsl((tid * 137.508) % 360, 58, 50);
  }

  function drawLayout(canvas, t, opts) {
    const g = gridGeom(t.shape, opts.maxW || 760, opts.maxH || 620);
    const ctx = setupCanvas(canvas, g.w, g.h);
    ctx.fillStyle = "#0b0e13";
    ctx.fillRect(0, 0, g.w, g.h);
    for (let i = 0; i < g.numel; i++) {
      const [r, c] = g.rc(g.coord(i));
      let color = cellColor(t, i, L.colorBy);
      if (L.pinned && !(t.warp[i] === L.pinned[0] && t.lane[i] === L.pinned[1])) color = shade("#3a4152", -0.2);
      if (opts.overlay && !opts.overlay.has(i)) color = shade("#3a4152", -0.35);
      ctx.fillStyle = color;
      ctx.fillRect(c * g.cell + 1, r * g.cell + 1, g.cell, g.cell);
    }
    // Thread boundaries: a dark line wherever the neighbouring element belongs to another
    // thread, so each thread's sizePerThread chunk reads as one block.
    if (g.cell >= 4) {
      ctx.strokeStyle = "#0b0e13";
      ctx.lineWidth = g.cell >= 10 ? 2 : 1;
      ctx.beginPath();
      const owner = (r, c) => {
        const i = g.index(r, c);
        return i < g.numel ? t.warp[i] * 64 + t.lane[i] : -1;
      };
      for (let r = 0; r < g.rows; r++) {
        for (let c = 0; c < g.cols; c++) {
          const o = owner(r, c);
          if (c + 1 < g.cols && owner(r, c + 1) !== o) { ctx.moveTo((c + 1) * g.cell + 0.5, r * g.cell); ctx.lineTo((c + 1) * g.cell + 0.5, (r + 1) * g.cell + 1); }
          if (r + 1 < g.rows && owner(r + 1, c) !== o) { ctx.moveTo(c * g.cell, (r + 1) * g.cell + 0.5); ctx.lineTo((c + 1) * g.cell + 1, (r + 1) * g.cell + 0.5); }
        }
      }
      ctx.stroke();
    }
    if (g.cell >= 20) {
      ctx.fillStyle = "rgba(255,255,255,.85)";
      ctx.font = `${Math.floor(g.cell * 0.42)}px Consolas, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < g.numel; i++) {
        const [r, c] = g.rc(g.coord(i));
        ctx.fillText(L.colorBy === "register" ? t.reg[i] : t.lane[i], c * g.cell + g.cell / 2 + 0.5, r * g.cell + g.cell / 2 + 1);
      }
    }
    if (opts.marks) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      for (const [i, label] of opts.marks) {
        const [r, c] = g.rc(g.coord(i));
        ctx.strokeRect(c * g.cell + 1.5, r * g.cell + 1.5, g.cell - 2, g.cell - 2);
        if (label !== undefined && g.cell >= 14) {
          ctx.fillStyle = "#fff";
          ctx.font = `${Math.floor(g.cell * 0.5)}px Consolas, monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, c * g.cell + g.cell / 2, r * g.cell + g.cell / 2 + 1);
        }
      }
    }
    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor((ev.clientX - rect.left) / g.cell), r = Math.floor((ev.clientY - rect.top) / g.cell);
      if (r < 0 || c < 0 || r >= g.rows || c >= g.cols) return hideTip();
      const i = g.index(r, c);
      if (i >= g.numel) return hideTip();
      const coord = g.coord(i);
      const rows = [["element", `[${coord.join(", ")}]`], ["warp", t.warp[i]], ["lane", t.lane[i]],
        ["thread id", t.warp[i] * 32 + t.lane[i]], ["register", t.reg[i]]];
      if (t.repl) rows.push(["held by", `${t.repl[i]} threads/regs`]);
      if (opts.extra) rows.push(...opts.extra(i));
      showTip(ev, kv(rows));
    };
    canvas.onmouseleave = hideTip;
    canvas.onclick = (ev) => {
      if (!opts.pinnable) return;
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor((ev.clientX - rect.left) / g.cell), r = Math.floor((ev.clientY - rect.top) / g.cell);
      const i = g.index(r, c);
      if (i >= g.numel) return;
      const p = [t.warp[i], t.lane[i]];
      L.pinned = L.pinned && L.pinned[0] === p[0] && L.pinned[1] === p[1] ? null : p;
      redrawLayouts();
    };
    return g;
  }

  function layoutParams(t) {
    const rows = [];
    const add = (lay, prefix) => {
      for (const [k, v] of Object.entries(lay)) {
        if (k === "parent") add(v, prefix + "parent.");
        else rows.push([prefix + k, Array.isArray(v) ? `[${v.join(", ")}]` : String(v)]);
      }
    };
    add(t.layout, "");
    rows.push(["shape", `[${t.shape.join(", ")}]`], ["regs/thread", t.regsPerThread], ["replication", t.replication]);
    return rows;
  }

  function renderLayouts() {
    const sec = $("#tab-layouts");
    if (!D.tensors.length) { sec.append(h("div", { class: "empty" }, "No modelled register layouts.")); return; }
    if (!L.key) L.key = D.tensors[0].key;
    const list = h("div", { class: "list", id: "tensor-list" });
    for (const t of D.tensors) {
      const use = t.uses[0];
      list.append(h("div", { class: "item", "data-key": t.key, onclick: () => { L.key = t.key; L.pinned = null; L.overlay = null; redrawLayouts(); } },
        h("div", { class: "t" }, `${t.layout.kind} `, h("span", { class: "mono muted" }, `${t.elem}[${t.shape.join("x")}]`)),
        h("div", { class: "d" }, `${t.encoding} - ${t.uses.length} use${t.uses.length > 1 ? "s" : ""}, first: ${use.opName.split(".").pop()} (${use.role})`)));
    }
    const sameShape = h("select", { id: "cmp-sel", onchange: (e) => { L.compare = e.target.value || null; redrawLayouts(); } });
    const controls = h("div", { class: "controls" },
      h("label", {}, "color by", h("select", { onchange: (e) => { L.colorBy = e.target.value; redrawLayouts(); } },
        ...["warp", "lane", "thread", "register"].map((m) => h("option", { value: m }, m)))),
      h("label", {}, "compare with", sameShape),
      h("span", { class: "muted small" }, "click an element to pin its thread"));
    const right = h("div", { class: "panel" }, h("h2", { id: "lay-title" }), controls,
      h("div", { class: "row-flex" },
        h("div", {}, h("div", { class: "canvas-wrap" }, h("canvas", { id: "lay-canvas" })), h("div", { class: "legend", id: "lay-legend" })),
        h("div", { id: "cmp-wrap" }),
        h("div", { style: "min-width:220px" }, h("h3", {}, "Layout"), h("div", { id: "lay-params" }), h("h3", {}, "Used by"), h("div", { id: "lay-uses" }))));
    sec.append(h("div", { class: "grid2" }, h("div", { class: "panel" }, h("h2", {}, "Tensors"), list), right));
    redrawLayouts();
  }

  function redrawLayouts() {
    const t = tensorsByKey.get(L.key);
    if (!t) return;
    for (const el of document.querySelectorAll("#tensor-list .item")) el.classList.toggle("sel", el.dataset.key === L.key);
    $("#lay-title").textContent = `${t.encoding}  ${t.elem}[${t.shape.join(" x ")}]`;
    const sel = $("#cmp-sel");
    sel.innerHTML = "";
    sel.append(h("option", { value: "" }, "none"));
    for (const o of D.tensors) {
      if (o.key !== t.key && o.shape.join() === t.shape.join()) sel.append(h("option", { value: o.key, selected: o.key === L.compare }, `${o.encoding} (${o.layout.kind})`));
    }
    const cmp = L.compare && tensorsByKey.get(L.compare) && tensorsByKey.get(L.compare).shape.join() === t.shape.join() ? tensorsByKey.get(L.compare) : null;
    const maxW = cmp ? 420 : 760;
    const overlay = L.overlay && L.overlay.key === t.key ? L.overlay : null;
    drawLayout($("#lay-canvas"), t, { maxW, maxH: 720, pinnable: true, overlay: overlay ? overlay.set : null, marks: overlay ? overlay.marks : null,
      extra: cmp ? (i) => [["other layout", `warp ${cmp.warp[i]} lane ${cmp.lane[i]} reg ${cmp.reg[i]}`]] : null });
    const cw = $("#cmp-wrap");
    cw.innerHTML = "";
    if (cmp) {
      const c2 = h("canvas");
      cw.append(h("div", { class: "muted small" }, cmp.encoding), c2);
      drawLayout(c2, cmp, { maxW, pinnable: true, extra: (i) => [["other layout", `warp ${t.warp[i]} lane ${t.lane[i]} reg ${t.reg[i]}`]] });
    }
    const leg = $("#lay-legend");
    leg.innerHTML = "";
    if (L.colorBy === "warp") {
      const nw = D.kernel.numWarps;
      for (let w = 0; w < nw; w++) leg.append(h("span", {}, h("i", { style: `background:${warpColor(w)}` }), `warp ${w}`));
      leg.append(h("span", {}, "lighter = higher lane"));
    } else if (L.colorBy === "register") {
      leg.append(h("span", {}, h("i", { style: `background:${viridis(0)}` }), "reg 0"),
        h("span", {}, h("i", { style: `background:${viridis(1)}` }), `reg ${t.regsPerThread - 1}`));
    }
    if (L.pinned) leg.append(h("span", {}, `pinned: warp ${L.pinned[0]} lane ${L.pinned[1]} (tid ${L.pinned[0] * 32 + L.pinned[1]})`));
    if (overlay) leg.append(h("span", {}, overlay.label));
    $("#lay-params").innerHTML = kv(layoutParams(t));
    const uses = $("#lay-uses");
    uses.innerHTML = "";
    for (const u of t.uses) uses.append(h("div", { class: "small" }, h("a", { onclick: () => focusOp(u.op) }, `ttgir:${u.line}`), " ", h("span", { class: "muted" }, `${u.opName} (${u.role})`)));
  }

  // ---- shared memory -----------------------------------------------------------------
  const S = { buffer: null, access: null, instr: "worst" };

  function renderShared() {
    const sec = $("#tab-shared");
    const { buffers, accesses } = D.shared;
    if (!accesses.length && !buffers.length) { sec.append(h("div", { class: "empty" }, "This kernel does not use shared memory explicitly.")); return; }
    if (S.access === null && accesses.length) S.access = 0;
    const list = h("div", { class: "list", id: "acc-list" });
    accesses.forEach((a, idx) => {
      const sev = a.degree >= 2 ? "warning" : a.degree > 1.05 ? "info" : "ok";
      list.append(h("div", { class: "item", "data-idx": idx, onclick: () => { S.access = idx; S.instr = "worst"; redrawShared(); } },
        h("div", { class: "t" }, a.opName.split(".").pop(), " ", h("span", { class: `badge ${sev}` }, `${a.degree.toFixed(2)}x`), a.estimated ? h("span", { class: "muted small" }, " estimated") : null),
        h("div", { class: "d" }, `${a.mechanism} - ${a.direction} - ${a.instructionsPerWarp} instr/warp - ttgir:${a.line}`)));
    });
    const right = h("div", { class: "panel" },
      h("h2", { id: "sh-title" }),
      h("div", { class: "controls" }, h("label", {}, "instruction", h("select", { id: "sh-instr", onchange: (e) => { S.instr = e.target.value; redrawShared(); } }))),
      h("div", { class: "row-flex" },
        h("div", {}, h("div", { class: "canvas-wrap" }, h("canvas", { id: "bank-canvas" })), h("div", { class: "legend", id: "bank-legend" })),
        h("div", { style: "min-width:300px;flex:1" }, h("h3", {}, "Bank load per lane group"), h("canvas", { id: "bank-hist" }), h("div", { id: "sh-detail", class: "small" }))));
    sec.append(h("div", { class: "grid2" }, h("div", { class: "panel" }, h("h2", {}, "Shared-memory accesses"), list), right));
    redrawShared();
  }

  function laneGroups(lanes, width) {
    const group = width <= 4 ? 32 : Math.max(1, Math.floor(128 / width));
    const groups = [];
    for (let s = 0; s < 32; s += group) groups.push(lanes.filter((l) => l.lane >= s && l.lane < s + group));
    return groups.filter((g) => g.length);
  }
  function bankWords(group, width) {
    const words = Array.from({ length: 32 }, () => new Set());
    for (const l of group) {
      const first = Math.floor(l.addr / 4), last = Math.floor((l.addr + Math.max(width, 1) - 1) / 4);
      for (let w = first; w <= last; w++) words[w % 32].add(w);
    }
    return words.map((s) => s.size);
  }

  function redrawShared() {
    const a = D.shared.accesses[S.access];
    for (const el of document.querySelectorAll("#acc-list .item")) el.classList.toggle("sel", Number(el.dataset.idx) === S.access);
    if (!a) return;
    const buf = D.shared.buffers.find((b) => b.id === a.bufferId);
    $("#sh-title").textContent = `${a.mechanism}  (${a.opName}, ttgir:${a.line}${a.loc ? ", " + fmtLoc(a.loc) : ""})`;
    const sel = $("#sh-instr");
    sel.innerHTML = "";
    sel.append(h("option", { value: "worst", selected: S.instr === "worst" }, "worst-case instruction"));
    a.warp0.forEach((ins, i) => sel.append(h("option", { value: String(i), selected: S.instr === String(i) }, `warp 0, instruction ${i} (${ins.wavefronts}/${ins.ideal} wavefronts)`)));
    const instr = S.instr === "worst" ? a.worst : a.warp0[Number(S.instr)];
    const canvas = $("#bank-canvas");
    if (!buf) { setupCanvas(canvas, 10, 10); return; }
    const [rows, cols] = buf.tile;
    const g = gridGeom([rows, cols], 640, 720);
    const ctx = setupCanvas(canvas, g.w, g.h);
    ctx.fillStyle = "#0b0e13";
    ctx.fillRect(0, 0, g.w, g.h);
    const touched = new Map();
    const contigDim = (buf.layout.order || [1, 0])[0];
    if (instr) {
      const elems = Math.max(1, Math.round(instr.width / buf.elemBytes));
      for (const l of instr.lanes) {
        for (let e = 0; e < elems; e++) {
          const rc = [l.tile[0], l.tile[1]];
          rc[contigDim] += e;
          if (rc[0] < rows && rc[1] < cols) touched.set(rc[0] * cols + rc[1], l.lane);
        }
      }
    }
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const bank = buf.banks[r][c];
        ctx.globalAlpha = instr && !touched.has(r * cols + c) ? 0.3 : 1;
        ctx.fillStyle = bankColor(bank);
        ctx.fillRect(c * g.cell + 1, r * g.cell + 1, g.cell - (g.cell > 5 ? 1 : 0), g.cell - (g.cell > 5 ? 1 : 0));
        ctx.globalAlpha = 1;
        if (g.cell >= 16) {
          ctx.fillStyle = "rgba(255,255,255,.8)";
          ctx.font = `${Math.floor(g.cell * 0.42)}px Consolas, monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(bank, c * g.cell + g.cell / 2, r * g.cell + g.cell / 2 + 1);
        }
      }
    }
    if (instr) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      for (const l of instr.lanes) {
        const [r, c] = l.tile;
        if (r < rows && c < cols) ctx.strokeRect(c * g.cell + 1.5, r * g.cell + 1.5, g.cell - 2, g.cell - 2);
      }
    }
    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor((ev.clientX - rect.left) / g.cell), r = Math.floor((ev.clientY - rect.top) / g.cell);
      if (r < 0 || c < 0 || r >= rows || c >= cols) return hideTip();
      const pairs = [["element", `[${r}, ${c}]`], ["bank", buf.banks[r][c]]];
      if (touched.has(r * cols + c)) pairs.push(["accessed by lane", touched.get(r * cols + c)]);
      showTip(ev, kv(pairs));
    };
    canvas.onmouseleave = hideTip;
    const leg = $("#bank-legend");
    leg.innerHTML = "";
    leg.append(h("span", {}, `${buf.name}: tile ${rows}x${cols}, ${buf.elemBytes} B elements, ${buf.bytes} B${buf.estimated ? " (estimated scratch)" : ""}`));
    const lay = buf.layout;
    leg.append(h("span", { class: "mono" }, Object.entries(lay).filter(([k]) => k !== "kind").map(([k, v]) => `${k}=${Array.isArray(v) ? "[" + v.join(",") + "]" : v}`).join(" ")));
    // Histogram: one row of 32 bank bars per lane group.
    const hist = $("#bank-hist");
    const groups = instr ? laneGroups(instr.lanes, instr.width) : [];
    const bw = 9, gh = 46;
    const hctx = setupCanvas(hist, 32 * bw + 40, Math.max(1, groups.length) * (gh + 8) + 4);
    hctx.font = "10px Consolas, monospace";
    groups.forEach((grp, gi) => {
      const counts = bankWords(grp, instr.width);
      const max = Math.max(1, ...counts);
      const y0 = gi * (gh + 8) + 4;
      hctx.fillStyle = "#8b95a7";
      hctx.fillText(`g${gi}`, 0, y0 + gh / 2 + 3);
      counts.forEach((n, b) => {
        const hh = (n / Math.max(max, 4)) * gh;
        hctx.fillStyle = n > 1 ? "#ff5c6c" : n === 1 ? bankColor(b) : "#232938";
        hctx.fillRect(28 + b * bw, y0 + gh - Math.max(hh, n ? 3 : 1), bw - 2, Math.max(hh, n ? 3 : 1));
      });
      hctx.fillStyle = "#d8dee9";
      hctx.fillText(`${max}x`, 28 + 32 * bw - 2, y0 + 10);
    });
    const det = $("#sh-detail");
    det.innerHTML = "";
    det.append(h("div", { html: kv([
      ["per CTA", `${a.wavefronts} wavefronts (ideal ${a.ideal})`],
      ["conflict factor", `${a.degree.toFixed(2)}x`],
      ["instructions/warp", a.instructionsPerWarp],
      ["bytes per lane", a.width],
      ["view", `[${a.viewShape.join("x")}] at offset [${a.offsets.join(", ")}]`],
      ["this instruction", instr ? `${instr.wavefronts} wavefronts (ideal ${instr.ideal})` : "-"],
    ]) }));
    for (const n of a.notes || []) det.append(h("div", { class: "note" }, n));
    det.append(h("div", { class: "note" }, "Bars: distinct 4-byte words each bank must serve for one lane group (128 B per wavefront). Red bars are conflicts."));
  }

  // ---- global memory -----------------------------------------------------------------
  const G = { idx: 0 };
  function renderGlobal() {
    const sec = $("#tab-global");
    if (!D.global.length) { sec.append(h("div", { class: "empty" }, "No global loads or stores.")); return; }
    const tb = h("table", { class: "t", id: "gl-table" }, h("tr", {}, ...["op", "source", "pointer", "tile", "bytes/lane", "PTX", "sectors/request", "efficiency"].map((x) => h("th", {}, x))));
    D.global.forEach((g, i) => {
      const effSev = g.efficiency < 0.2 ? "error" : g.efficiency < 0.5 ? "warning" : "ok";
      const vecSev = g.vecBytes < g.potentialVec * g.elemBytes ? "warning" : "ok";
      tb.append(h("tr", { class: "click", "data-idx": i, onclick: () => { G.idx = i; redrawGlobal(); } },
        h("td", { class: "mono" }, `${g.kind} #${g.op}`), h("td", { class: "mono" }, fmtLoc(g.loc)),
        h("td", { class: "mono" }, g.baseArgName || "-"), h("td", { class: "mono" }, g.shape.join("x")),
        h("td", {}, h("span", { class: `badge ${vecSev}` }, `${g.vecBytes} B`), g.vecBytes < g.potentialVec * g.elemBytes ? h("span", { class: "muted small" }, ` of ${g.potentialVec * g.elemBytes}`) : null),
        h("td", { class: "mono small" }, g.ptx.length ? [...new Set(g.ptx.map((p) => p.opcode))].join(", ") : "-"),
        h("td", {}, `${g.sectorsPerRequest} (ideal ${g.idealSectors})`),
        h("td", {}, h("span", { class: `badge ${effSev}` }, `${Math.round(g.efficiency * 100)}%`))));
    });
    sec.append(h("div", { class: "panel" }, h("h2", {}, "Global loads and stores"), tb),
      h("div", { class: "row-flex", style: "margin-top:16px" },
        h("div", { class: "panel", style: "flex:1;min-width:340px" }, h("h2", {}, "Vector width limits"), h("div", { id: "gl-limits", class: "limits" })),
        h("div", { class: "panel", style: "flex:2;min-width:460px" }, h("h2", {}, "Warp 0, first request"), h("canvas", { id: "gl-sectors" }), h("div", { id: "gl-note", class: "note" }))));
    redrawGlobal();
  }

  function redrawGlobal() {
    const g = D.global[G.idx];
    for (const el of document.querySelectorAll("#gl-table tr[data-idx]")) el.classList.toggle("sel", Number(el.dataset.idx) === G.idx);
    const lim = $("#gl-limits");
    lim.innerHTML = "";
    const maxEl = Math.max(...g.limits.map((l) => l.elements), 1);
    for (const l of g.limits) {
      const bind = l.elements === g.vec && l.factor !== "max";
      lim.append(h("div", { class: `row${bind ? " bind" : ""}`, title: l.detail },
        h("span", { class: "name" }, l.factor), h("div", { class: "bar" }, h("div", { style: `width:${(100 * Math.min(l.elements, maxEl)) / maxEl}%;${bind ? "background:var(--warning)" : ""}` })),
        h("span", { class: "mono" }, `${l.elements * g.elemBytes} B`)));
      lim.append(h("div", { class: "note", style: "margin:-2px 0 6px 98px" }, l.detail));
    }
    lim.append(h("div", { class: "small", style: "margin-top:8px" }, `Result: ${g.vec} x ${g.elemBytes} B = `, h("b", {}, `${g.vecBytes} B per lane`), g.ptxBytes ? ` (PTX: ${g.ptxBytes} B)` : ""));
    // Sector diagram: distinct 32 B sectors touched by warp 0's first request.
    const canvas = $("#gl-sectors");
    const width = g.vecBytes;
    const sectors = new Map();
    for (const s of g.sample) {
      for (let b = 0; b < width; b += Math.min(width, 4)) {
        const sec = Math.floor((s.addr + b) / 32);
        if (!sectors.has(sec)) sectors.set(sec, []);
        sectors.get(sec).push([(s.addr + b) % 32, Math.min(4, width), s.lane]);
      }
    }
    const keys = [...sectors.keys()].sort((a, b) => a - b);
    const sw = 64, perRow = 10, sh = 26;
    const rowsN = Math.ceil(keys.length / perRow);
    const ctx = setupCanvas(canvas, perRow * (sw + 6) + 4, rowsN * (sh + 18) + 4);
    ctx.font = "10px Consolas, monospace";
    keys.forEach((k, i) => {
      const x = (i % perRow) * (sw + 6) + 2, y = Math.floor(i / perRow) * (sh + 18) + 14;
      ctx.fillStyle = "#232938";
      ctx.fillRect(x, y, sw, sh);
      for (const [off, len, lane] of sectors.get(k)) {
        ctx.fillStyle = hsl((lane * 360) / 32, 62, 55);
        ctx.fillRect(x + (off / 32) * sw, y, Math.max(1, (len / 32) * sw), sh);
      }
      ctx.strokeStyle = "#5a6478";
      ctx.strokeRect(x + 0.5, y + 0.5, sw - 1, sh - 1);
      if (i > 0 && keys[i - 1] !== k - 1) { ctx.fillStyle = "#ff5c6c"; ctx.fillText("gap", x, y - 3); }
      else { ctx.fillStyle = "#8b95a7"; ctx.fillText(`+${(k - keys[0]) * 32}`, x, y - 3); }
    });
    $("#gl-note").textContent = `${keys.length} sectors of 32 B touched by ${g.sample.length} lanes x ${width} B (ideal ${g.idealSectors}). Colors are lanes; "gap" marks a jump in the address space. Runtime strides use the captured launch arguments when available.`;
  }

  // ---- program tiling ----------------------------------------------------------------
  const T = { n: null, wave: 20, playing: null, hover: null };
  function renderTiling() {
    const sec = $("#tab-tiling");
    const t = D.tiling;
    if (!t) { sec.append(h("div", { class: "empty" }, "Program tiling needs a captured launch (run the kernel through `tritonlens run`).")); return; }
    T.n = t.programs.length;
    const slider = h("input", { type: "range", min: 1, max: t.programs.length, value: t.programs.length, oninput: (e) => { T.n = Number(e.target.value); drawTiling(); } });
    const play = h("button", { class: "btn", onclick: () => {
      if (T.playing) { clearInterval(T.playing); T.playing = null; play.textContent = "play"; return; }
      play.textContent = "pause";
      if (T.n >= t.programs.length) T.n = 0;
      const step = Math.max(1, Math.round(t.programs.length / 240));
      T.playing = setInterval(() => {
        T.n = Math.min(t.programs.length, T.n + step);
        slider.value = T.n;
        drawTiling();
        if (T.n >= t.programs.length) { clearInterval(T.playing); T.playing = null; play.textContent = "play"; }
      }, 30);
    } }, "play");
    const wave = h("input", { type: "number", min: 1, value: T.wave, style: "width:70px", oninput: (e) => { T.wave = Math.max(1, Number(e.target.value) || 1); drawTiling(); } });
    sec.append(h("div", { class: "panel" },
      h("h2", {}, `Output tiles written by each program (${t.arg || "output"}: ${t.matrixShape.join(" x ")}, tile ${t.tileShape.join(" x ")})`),
      h("div", { class: "controls" }, play, h("label", {}, "programs launched", slider, h("span", { id: "tile-count", class: "mono" })),
        h("label", {}, "resident programs per wave", wave)),
      h("div", { class: "canvas-wrap" }, h("canvas", { id: "tile-canvas" })),
      h("div", { class: "note" }, "Color = launch order (program id); the current wave is outlined. Tile origins come from evaluating the store's address expression for every program with the captured arguments.")));
    drawTiling();
  }

  function drawTiling() {
    const t = D.tiling;
    const canvas = $("#tile-canvas");
    const twoD = t.matrixShape.length === 2;
    let rows, cols, tr, tc, place;
    if (twoD) {
      [rows, cols] = t.matrixShape;
      [tr, tc] = t.tileShape.length === 2 ? t.tileShape : [1, t.tileShape[0]];
      place = (p) => p.origin;
    } else {
      const n = t.matrixShape[0], tile = t.tileShape[0];
      const tilesPerRow = Math.max(1, Math.min(64, Math.ceil(Math.sqrt(n / tile) * 2)));
      cols = tilesPerRow * tile;
      rows = Math.ceil(n / cols) * 8;
      tr = 8; tc = tile;
      place = (p) => [Math.floor(p.origin[0] / cols) * 8, p.origin[0] % cols];
    }
    const maxW = Math.min(1100, document.body.clientWidth - 80), maxH = 640;
    const s = Math.min(maxW / cols, maxH / rows);
    const W = Math.ceil(cols * s) + 2, H = Math.ceil(rows * s) + 2;
    const ctx = setupCanvas(canvas, W, H);
    ctx.fillStyle = "#0b0e13";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#3a4152";
    ctx.strokeRect(0.5, 0.5, cols * s + 1, rows * s + 1);
    const total = t.programs.length;
    const waveStart = Math.max(0, Math.floor((T.n - 1) / T.wave) * T.wave);
    const rects = [];
    t.programs.forEach((p, i) => {
      const [r, c] = place(p);
      const x = 1 + c * s, y = 1 + r * s, w = Math.max(1, Math.min(tc, cols - c) * s), hh = Math.max(1, Math.min(tr, rows - r) * s);
      rects.push([x, y, w, hh, i]);
      ctx.fillStyle = i < T.n ? viridis(i / Math.max(1, total - 1)) : "#1c2130";
      ctx.fillRect(x, y, w, hh);
      if (w > 4 && hh > 4) { ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, hh - 1); }
    });
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    for (const [x, y, w, hh, i] of rects) if (i >= waveStart && i < T.n) ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, hh - 2));
    ctx.lineWidth = 1;
    if (s * Math.min(tr, tc) >= 18) {
      ctx.font = "10px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const [x, y, w, hh, i] of rects) if (i < T.n) { ctx.fillStyle = i / total > 0.6 ? "#111" : "#fff"; ctx.fillText(i, x + w / 2, y + hh / 2); }
    }
    $("#tile-count").textContent = ` ${T.n} / ${total}`;
    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const hit = rects.find(([x, y, w, hh]) => mx >= x && mx < x + w && my >= y && my < y + hh);
      if (!hit) return hideTip();
      const p = t.programs[hit[4]];
      showTip(ev, kv([["program id", `(${p.pid.join(", ")})`], ["launch order", hit[4]], ["tile origin", `[${p.origin.join(", ")}]`],
        ["wave", Math.floor(hit[4] / T.wave)]]));
    };
    canvas.onmouseleave = hideTip;
  }

  // ---- IR & source ---------------------------------------------------------------------
  const I = { op: null };
  function highlightIR(line) {
    let s = esc(line);
    s = s.replace(/(loc\([^)]*\)(?:\))?)/g, '<span class="tk-loc">$1</span>');
    s = s.replace(/(#(?:triton_gpu|ttg)\.[a-z_]+|#(?:blocked|mma|shared|smem)\w*)/g, '<span class="tk-lay">$1</span>');
    s = s.replace(/(^|\s|=\s)((?:tt|triton_gpu|ttg|arith|scf|math|ttng)\.[\w.]+)/g, '$1<span class="tk-op">$2</span>');
    s = s.replace(/(%[\w#$.-]+)/g, '<span class="tk-val">$1</span>');
    return s;
  }
  function renderIR() {
    const sec = $("#tab-ir");
    const lines = D.ir.split("\n");
    const opByLine = new Map(D.ops.map((o) => [o.line, o]));
    const pre = h("pre", { class: "code", id: "ir-pre" });
    lines.forEach((ln, i) => {
      const op = opByLine.get(i + 1);
      const fs = op ? findingsByOp.get(op.id) : null;
      const sev = fs ? fs.map((f) => f.severity).sort((a, b) => ["error", "warning", "info"].indexOf(a) - ["error", "warning", "info"].indexOf(b))[0] : "";
      const el = h("span", { class: `ln${op ? " op" : ""}`, "data-line": i + 1, html: `<span class="no">${i + 1}</span><span class="gut ${sev}"></span> ${highlightIR(ln)}` });
      if (op) el.onclick = () => selectOp(op.id);
      pre.append(el);
    });
    const src = D.source;
    const srcPre = h("pre", { class: "code", id: "src-pre" });
    if (src && src.lines) src.lines.forEach((ln, i) => srcPre.append(h("span", { class: "ln", "data-line": i + 1, html: `<span class="no">${i + 1}</span> ${esc(ln)}` })));
    else srcPre.append(h("span", { class: "ln muted" }, src ? `source not found: ${src.file}` : "no source locations"));
    sec.append(h("div", { class: "grid-ir" },
      h("div", { class: "panel" }, h("h2", {}, "Triton GPU IR (TTGIR)"), pre),
      h("div", {}, h("div", { class: "panel", style: "margin-bottom:12px" }, h("h2", {}, "Selected op"), h("div", { id: "op-detail", class: "small muted" }, "Click an op in the IR.")),
        h("div", { class: "panel" }, h("h2", {}, src ? src.file : "Source"), srcPre))));
    if (I.op === null) {
      const first = D.findings.find((f) => f.op !== null);
      const firstMem = D.ops.find((o) => /\.(load|store)$/.test(o.name));
      I.op = first ? first.op : firstMem ? firstMem.id : null;
    }
    if (I.op !== null) selectOp(I.op);
  }
  function selectOp(id) {
    I.op = id;
    const op = opsById.get(id);
    if (!op) return;
    for (const el of document.querySelectorAll("#ir-pre .ln.sel")) el.classList.remove("sel");
    const center = (pre, el) => { pre.scrollTop = el.offsetTop - pre.offsetTop - pre.clientHeight / 2; };
    const line = document.querySelector(`#ir-pre .ln[data-line="${op.line}"]`);
    if (line) { line.classList.add("sel"); center($("#ir-pre"), line); }
    for (const el of document.querySelectorAll("#src-pre .ln.hl")) el.classList.remove("hl");
    if (op.loc) {
      const sl = document.querySelector(`#src-pre .ln[data-line="${op.loc.line}"]`);
      if (sl) { sl.classList.add("hl"); center($("#src-pre"), sl); }
    }
    const det = $("#op-detail");
    det.innerHTML = "";
    det.classList.remove("muted");
    det.append(h("div", { html: kv([["op", op.name], ["ttgir line", op.line], ["source", op.loc ? `${op.loc.file}:${op.loc.line}:${op.loc.col}` : "-"],
      ["result", op.type || "-"], ["layout", op.encoding || "-"], ["loop depth", op.depth]]) }));
    for (const t of D.opTensors[id] || []) det.append(h("div", {}, h("a", { onclick: () => focusTensor(t.key) }, `show ${t.role} layout`)));
    const sh = D.shared.accesses.findIndex((a) => a.op === id);
    if (sh >= 0) det.append(h("div", {}, h("a", { onclick: () => focusShared(id) }, "show shared-memory banks")));
    const gl = D.global.findIndex((g) => g.op === id);
    if (gl >= 0) det.append(h("div", {}, h("a", { onclick: () => focusGlobal(id) }, "show global access")));
    for (const f of findingsByOp.get(id) || []) det.append(findingEl(f));
  }

  // ---- cross navigation ----------------------------------------------------------------
  function focusOp(id) { I.op = id; showTab("ir"); selectOp(id); }
  function focusTensor(key) { L.key = key; L.pinned = null; L.overlay = null; showTab("layouts"); redrawLayouts(); }
  function focusShared(opId) {
    const idx = D.shared.accesses.findIndex((a) => a.op === opId);
    if (idx >= 0) { S.access = idx; S.instr = "worst"; }
    showTab("shared");
    redrawShared();
  }
  function focusGlobal(opId) {
    const idx = D.global.findIndex((g) => g.op === opId);
    if (idx >= 0) G.idx = idx;
    showTab("global");
    redrawGlobal();
  }

  const RENDER = { overview: renderOverview, layouts: renderLayouts, shared: renderShared, global: renderGlobal, tiling: renderTiling, ir: renderIR };

  renderHeader();
  const nav = $("nav");
  for (const [id, label] of TABS) nav.append(h("button", { "data-tab": id, onclick: () => showTab(id) }, label));
  const initial = (location.hash || "").slice(1);
  showTab(TABS.some(([id]) => id === initial) ? initial : "overview");
})();
