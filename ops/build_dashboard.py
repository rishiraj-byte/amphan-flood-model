"""build_dashboard.py - write dashboard.html into a FloodModel package folder (23 Nov 2026).

  python build_dashboard.py PACKAGE_FOLDER

Reads the package's results_*/summary.json (+ eval.json) and ops/improve/research_state.json and writes one offline
HTML page (no internet needed; images are linked from the package's own results folders). build_final.py calls this.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

LABELS = {
    "v4_B": ("old recipe, seed 0", "base"), "v4_B_seed1": ("old recipe, seed 1", "base"),
    "v4_sarOnly": ("old recipe, radar-only run", "base"),
    "v5_wide384": ("bigger patches (384 px)", "test"), "v5_long": ("longer training", "test"),
    "v6_ff80": ("more flood-centred patches (80 %)", "test"), "v6_base64": ("wider network (base 64)", "test"),
    "v6_lr1e3": ("higher learning rate (1e-3)", "test"), "v6_wd1e3": ("stronger weight decay (1e-3)", "test"),
    "v6_ff50": ("fewer flood-centred patches (50 %)", "test"),
}


def storm(pkg, folder, title, notes):
    d = pkg / folder
    if not (d / "summary.json").exists():
        return {"folder": folder, "title": title, "ready": False, "notes": notes}
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    det = s.get("detect", {})
    out = {"folder": folder, "title": title, "ready": True, "notes": notes, "created": s.get("created"),
           "flood_km2": det.get("flood_km2"), "water_km2": det.get("permanent_water_km2"),
           "covered_km2": det.get("radar_covered_km2"), "flood07_km2": det.get("flood_km2_if_cutoff_0.7"),
           "models": det.get("models", []), "tta": s.get("tta"), "cut": s.get("threshold"),
           "risk": (s.get("risk") or {}).get("zones", {}),
           "img_flood": f"{folder}/preview_flood_over_radar.png" if (d / "preview_flood_over_radar.png").exists() else f"{folder}/flood_map.png",
           "img_risk": f"{folder}/preview_risk_over_radar.png" if (d / "preview_risk_over_radar.png").exists() else f"{folder}/risk_map.png"}
    if (d / "eval.json").exists():
        e = json.loads((d / "eval.json").read_text(encoding="utf-8")).get("ensemble", {})
        out["eval"] = {k: e[k] for k in ("independent_test", "published_test") if k in e}
    return out


def build(pkg):
    pkg = Path(pkg).resolve()
    st =json.loads((HERE / "improve" / "research_state.json").read_text(encoding="utf-8"))
    tests = [{"id": k, "label": LABELS.get(k, (k, "test"))[0], "kind": LABELS.get(k, (k, "test"))[1],
              "mean": v["mean"], "pub": v.get("pub"), "ind": v.get("ind"), "cut": v.get("cut")}
             for k, v in st["scores"].items()]
    data = {
        "version": pkg.name.replace("FloodModel_", ""), "built": time.strftime("%d %b %Y %H:%M"),
        "base_mean": st["best"]["mean"], "spread": st["spread"], "bar": round(st["best"]["mean"] + st["spread"], 4),
        "tests": tests,
        "storms": [
            storm(pkg, "results_amphan_2020", "Amphan 2020",
                  "Training storm. Scores are on held-out test blocks the models never trained on."),
            storm(pkg, "results_yaas_2021", "Yaas 2021",
                  "Unseen storm (never trained on). Radar pairs T012A 14→26 May, T048D 17→29 May, T085A 19→31 May. "
                  "No reference flood map exists, so there is no score: check it by eye against the Charter 714 maps."),
        ],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (pkg / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"wrote {pkg / 'dashboard.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flood Model Dashboard</title>
<style>
/* Layout: one column of storm panels (map + numbers), then the training record and the pipeline. Radar-night palette. */
:root{
  --bg:#f3f6f8; --panel:#ffffff; --ink:#122130; --muted:#5b6b7a; --line:#d8e0e7;
  --water:#1668b8; --water-soft:#dceaf8; --base:#7b8a99; --warn:#b26b00; --good:#1d7a4a;
  --mono:ui-monospace,"Cascadia Mono",Consolas,monospace;
  --sans:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0a1119; --panel:#111c27; --ink:#e4edf5; --muted:#93a5b6; --line:#243444;
  --water:#5aa9f0; --water-soft:#14304a; --base:#8796a5; --warn:#f0b13a; --good:#4cc488; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0a1119; --panel:#111c27; --ink:#e4edf5; --muted:#93a5b6; --line:#243444;
  --water:#5aa9f0; --water-soft:#14304a; --base:#8796a5; --warn:#f0b13a; --good:#4cc488; color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans)}
.wrap{max-width:1120px;margin:0 auto;padding-block:28px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
header{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px 24px}
h1{font-size:28px;line-height:1.15;margin:0;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:19px;margin:0;text-wrap:balance}
h3{font-size:13px;margin:0;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
.sub{color:var(--muted);margin:4px 0 0}
.stamp{font:13px var(--mono);color:var(--muted)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px;display:grid;gap:16px;min-width:0}
.storm{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:20px}
@media (max-width:820px){.storm{grid-template-columns:minmax(0,1fr)}}
.maps{display:grid;gap:8px;min-width:0}
.tabs{display:flex;gap:6px}
.tabs button{font:600 13px var(--sans);padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--water);border-color:var(--water);color:var(--panel)}
.tabs button:focus-visible,.toggle:focus-visible{outline:2px solid var(--water);outline-offset:2px}
.maps img{width:100%;max-width:100%;height:auto;border-radius:6px;border:1px solid var(--line);background:var(--bg);display:block}
.cap{font-size:12px;color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:grid;gap:2px}
.tile .v{font:600 22px/1.2 var(--mono);font-variant-numeric:tabular-nums}
.tile .k{font-size:12px;color:var(--muted)}
.pill{display:inline-flex;align-items:center;gap:6px;font:600 12px var(--sans);padding:3px 10px;border-radius:999px;border:1px solid currentColor;width:max-content}
.pill.ok{color:var(--good)} .pill.wait{color:var(--warn)} .pill.info{color:var(--water)}
.note{font-size:13px;color:var(--muted);margin:0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)} th{color:var(--muted);font-weight:600}
td.n{font-family:var(--mono);text-align:right} th.n{text-align:right}
.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block}
.sw.line{width:16px;height:0;border-top:2px solid var(--warn);border-radius:0}
.sw.dash{width:16px;height:0;border-top:2px dashed var(--base);border-radius:0}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--panel);font-size:12px;padding:6px 8px;border-radius:6px;max-width:260px;z-index:5}
.toggle{font:600 13px var(--sans);background:none;border:0;color:var(--water);cursor:pointer;padding:0;justify-self:start}
ol.steps{margin:0;padding-left:20px;display:grid;gap:8px}
ol.steps b{font-weight:600}
pre{margin:0;font:13px/1.5 var(--mono);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
@media (max-width:820px){.two{grid-template-columns:minmax(0,1fr)}}
.caveat{border-left:3px solid var(--warn);padding:10px 14px;background:var(--panel);border-radius:0 8px 8px 0;font-size:14px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Flood Model <span id="ver"></span></h1>
      <p class="sub">Cyclone flood maps from Sentinel-1 radar (before/after pairs) · Bay of Bengal</p>
    </div>
    <div class="stamp" id="built"></div>
  </header>
  <div class="caveat"><b>Read scores with care.</b> The training labels come from the published Mondal map (Zenodo 4390084), so scores against that map are optimistic. The independent score uses the NASA JPL ARIA flood proxy map.</div>
  <div id="storms" style="display:grid;gap:28px"></div>
  <section class="panel" aria-labelledby="t-train">
    <div style="display:grid;gap:4px">
      <h2 id="t-train">Training record: did any recipe change beat the model in use?</h2>
      <p class="note">Validation score = mean of the published-map and independent-map IoU on validation blocks (never test). A change counts only if it clears the old mean plus the seed-to-seed spread.</p>
    </div>
    <div class="legend" id="lg"></div>
    <div class="scroll"><svg id="dots" role="img" aria-label="Validation score per recipe test"></svg></div>
    <button class="toggle" id="tbtn" aria-expanded="false">Show as table</button>
    <div class="scroll" id="tbl" hidden></div>
    <p class="note" id="verdict"></p>
  </section>
  <section class="two">
    <div class="panel">
      <h2>Automatic storm pipeline</h2>
      <ol class="steps">
        <li><b>PC</b> downloads the radar scenes, builds the storm stack and sends <code>stack.tar</code> + <code>.sha256</code> over the bridge.</li>
        <li><b>Laptop</b> (<code>storm_autorun.py</code>) waits for it and asks the PC's <code>pc_auto.py</code> for a fresh checksum, and for one resend if the copies differ.</li>
        <li>The laptop checks the SHA-256 and unpacks only into a new folder (no paths outside it).</li>
        <li>The laptop runs Flood Model v1 (two-model ensemble, 8-way test-time augmentation, cutoff 0.30) on the GPU, falling back to the CPU.</li>
        <li>The laptop sends the flood km² and maps back to the PC.</li>
      </ol>
      <p class="note">Both helpers stop by themselves after a set number of hours and never run code taken from a message.</p>
    </div>
    <div class="panel">
      <h2>Run it yourself</h2>
      <pre>run_flood_model.bat STACK_FOLDER OUTPUT_FOLDER [--cpu]
./run_flood_model.sh STACK_FOLDER OUTPUT_FOLDER [--cpu]</pre>
      <p class="note">STACK_FOLDER is a storm stack in kit v4 layout (meta.json + inputs_detect.npy + inputs_forecast.npy). Outputs: flood_class.tif, flood_probability.tif, flood_risk.tif, PNG maps, summary.json. Check the files with SHA256SUMS.txt.</p>
    </div>
  </section>
</div>
<div id="tip" hidden></div>
<script>
const D = __DATA__;
const $ = (s, el = document) => el.querySelector(s);
const fmt = (v, d = 1) => v == null ? "–" : Number(v).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
function el(tag, props = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) { if (k === "text") e.textContent = v; else e.setAttribute(k, v); }
  for (const c of [].concat(kids)) if (c) e.append(c);
  return e;
}
$("#ver").textContent = D.version;
$("#built").textContent = "built " + D.built;

const tip = $("#tip");
function showTip(ev, text) { tip.textContent = text; tip.hidden = false; tip.style.left = Math.min(ev.clientX + 12, innerWidth - 270) + "px"; tip.style.top = (ev.clientY + 12) + "px"; }
function hideTip() { tip.hidden = true; }

// storm panels
for (const s of D.storms) {
  const p = el("section", {class: "panel", "aria-label": s.title});
  const head = el("div", {style: "display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;justify-content:space-between"}, [
    el("h2", {text: s.title}),
    el("span", {class: "pill " + (s.ready ? (s.eval ? "ok" : "info") : "wait"),
                text: s.ready ? (s.eval ? "mapped · scored" : "mapped · no reference map") : "waiting for the radar stack"})]);
  p.append(head);
  if (!s.ready) { p.append(el("p", {class: "note", text: s.notes})); $("#storms").append(p); continue; }
  const grid = el("div", {class: "storm"});
  const maps = el("div", {class: "maps"});
  const img = el("img", {src: s.img_flood, alt: s.title + " flood map over the after-storm radar", loading: "lazy"});
  const cap = el("div", {class: "cap", text: "Flood (blue) over the after-storm radar image."});
  const bF = el("button", {type: "button", "aria-pressed": "true", text: "Flood"});
  const bR = el("button", {type: "button", "aria-pressed": "false", text: "Risk"});
  bF.onclick = () => { img.src = s.img_flood; bF.setAttribute("aria-pressed", "true"); bR.setAttribute("aria-pressed", "false"); cap.textContent = "Flood (blue) over the after-storm radar image."; };
  bR.onclick = () => { img.src = s.img_risk; bR.setAttribute("aria-pressed", "true"); bF.setAttribute("aria-pressed", "false"); cap.textContent = "Relative flood risk from the pre-storm radar + storm track (a ranking, not a probability)."; };
  maps.append(el("div", {class: "tabs"}, [bF, bR]), img, cap);
  const side = el("div", {style: "display:grid;gap:14px;align-content:start;min-width:0"});
  const tiles = el("div", {class: "tiles"});
  const tile = (v, k) => el("div", {class: "tile"}, [el("div", {class: "v", text: v}), el("div", {class: "k", text: k})]);
  tiles.append(tile(fmt(s.flood_km2, 0) + " km²", "new flood water (cutoff " + s.cut + ")"),
               tile(fmt(s.flood07_km2, 0) + " km²", "high-confidence flood (cutoff 0.7)"),
               tile(fmt(s.water_km2, 0) + " km²", "permanent water (rivers, sea, ponds)"),
               tile(fmt(s.covered_km2, 0) + " km²", "area the radar pair covers"));
  side.append(tiles);
  if (s.eval) {
    const t = el("table");
    t.append(el("tr", {}, [el("th", {text: "Test blocks vs"}), el("th", {class: "n", text: "IoU"}), el("th", {class: "n", text: "Precision"}), el("th", {class: "n", text: "Recall"})]));
    const rows = [["independent (ARIA)", s.eval.independent_test], ["published (Mondal)*", s.eval.published_test]];
    for (const [n, e] of rows) if (e) t.append(el("tr", {}, [el("td", {text: n}), el("td", {class: "n", text: fmt(e.iou, 3)}), el("td", {class: "n", text: fmt(e.precision, 3)}), el("td", {class: "n", text: fmt(e.recall, 3)})]));
    side.append(el("div", {class: "scroll"}, t), el("p", {class: "note", text: "* the map the training labels came from, so this score is optimistic."}));
  }
  side.append(el("p", {class: "note", text: s.notes}));
  side.append(el("p", {class: "note", text: "Models: " + s.models.join(" + ") + " · TTA " + s.tta + " · made " + (s.created || "")}));
  grid.append(maps, side);
  p.append(grid);
  $("#storms").append(p);
}

// training dot plot (one axis, single hue for tests, muted for the old-recipe runs)
(function () {
  const T = D.tests, ns = "http://www.w3.org/2000/svg";
  const W = 760, rowH = 26, top = 18, left = 250, right = 70, H = top + T.length * rowH + 34;
  const x0 = 0.37, x1 = 0.42, X = v => left + (v - x0) / (x1 - x0) * (W - left - right);
  const svg = $("#dots"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.style.maxWidth = "100%"; svg.style.height = "auto";
  const mk = (tag, a) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); svg.append(e); return e; };
  const base = T.length * rowH + top;
  for (const v of [0.37, 0.38, 0.39, 0.40, 0.41, 0.42]) {
    mk("line", {x1: X(v), x2: X(v), y1: top - 6, y2: base, style: "stroke:var(--line)", "stroke-width": 1});
    mk("text", {x: X(v), y: base + 18, "text-anchor": "middle", "font-size": 12, style: "fill:var(--muted)", "font-family": "ui-monospace,Consolas,monospace"}).textContent = v.toFixed(2);
  }
  mk("line", {x1: X(D.base_mean), x2: X(D.base_mean), y1: top - 6, y2: base, style: "stroke:var(--base)", "stroke-width": 2, "stroke-dasharray": "5 4"});
  mk("line", {x1: X(D.bar), x2: X(D.bar), y1: top - 6, y2: base, style: "stroke:var(--warn)", "stroke-width": 2});
  T.forEach((t, i) => {
    const y = top + i * rowH + rowH / 2;
    mk("text", {x: left - 12, y: y + 4, "text-anchor": "end", "font-size": 13, style: "fill:var(--ink)"}).textContent = t.label;
    const c = t.kind === "base" ? "var(--base)" : "var(--water)";
    mk("line", {x1: left, x2: X(t.mean), y1: y, y2: y, style: "stroke:var(--line)", "stroke-width": 1});
    mk("circle", {cx: X(t.mean), cy: y, r: 6, style: `fill:${c};stroke:var(--panel)`, "stroke-width": 2});
    mk("text", {x: W - right + 12, y: y + 4, "font-size": 12, style: "fill:var(--muted)", "font-family": "ui-monospace,Consolas,monospace"}).textContent = t.mean.toFixed(3);
    const hit = mk("rect", {x: left - 240, y: y - rowH / 2, width: W - left + 240, height: rowH, fill: "transparent"});
    const txt = `${t.label}: validation ${t.mean.toFixed(3)} (published ${t.pub}, independent ${t.ind}, cutoff ${t.cut})` +
                (t.kind === "base" ? " · old recipe run" : (t.mean > D.bar ? " · KEPT" : " · dropped: within luck"));
    hit.addEventListener("mousemove", ev => showTip(ev, txt)); hit.addEventListener("mouseleave", hideTip);
  });
  const lg = $("#lg");
  lg.append(el("span", {}, [el("i", {class: "sw", style: "background:var(--base)"}), "old recipe (3 runs)"]),
            el("span", {}, [el("i", {class: "sw", style: "background:var(--water)"}), "recipe change tested"]),
            el("span", {}, [el("i", {class: "sw dash"}), "old mean " + D.base_mean.toFixed(3)]),
            el("span", {}, [el("i", {class: "sw line"}), "bar to beat " + D.bar.toFixed(3) + " (mean + spread)"]));
  const t = el("table");
  t.append(el("tr", {}, ["Run", "Validation", "Published", "Independent", "Cutoff", "Verdict"].map((h, i) => el("th", {class: i ? "n" : "", text: h}))));
  for (const r of T) t.append(el("tr", {}, [el("td", {text: r.label}), el("td", {class: "n", text: r.mean.toFixed(4)}), el("td", {class: "n", text: r.pub}), el("td", {class: "n", text: r.ind}), el("td", {class: "n", text: r.cut}),
    el("td", {class: "n", text: r.kind === "base" ? "old recipe" : (r.mean > D.bar ? "kept" : "dropped")})]));
  $("#tbl").append(t);
  $("#tbtn").onclick = () => { const h = !$("#tbl").hidden; $("#tbl").hidden = h; $("#tbtn").textContent = h ? "Show as table" : "Hide table"; $("#tbtn").setAttribute("aria-expanded", String(!h)); };
  const tested = T.filter(r => r.kind === "test"), best = tested.reduce((a, b) => b.mean > a.mean ? b : a, tested[0]);
  $("#verdict").textContent = tested.some(r => r.mean > D.bar)
    ? "At least one change cleared the bar; see the table."
    : `${tested.length} changes tested; none cleared the bar (best: ${best.label}, ${best.mean.toFixed(3)}). The model in use stays: the limit is the data (one storm, one label map), not the training settings.`;
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build(sys.argv[1])
