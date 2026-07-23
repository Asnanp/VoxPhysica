#!/usr/bin/env python
"""Tiny live dashboard for VocalMorph metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VocalMorph Live</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8f3;
      --ink: #16201d;
      --muted: #607069;
      --panel: #ffffff;
      --line: #d9ded5;
      --accent: #0f766e;
      --accent-2: #b45309;
      --good: #15803d;
      --warn: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 22px clamp(16px, 4vw, 44px);
      border-bottom: 1px solid var(--line);
      background: #fffdf8;
    }
    h1 { margin: 0; font-size: clamp(24px, 3vw, 38px); font-weight: 750; }
    .sub { margin-top: 6px; color: var(--muted); font-size: 14px; }
    .status {
      min-width: 132px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      text-align: center;
      background: var(--panel);
      color: var(--accent);
      font-weight: 700;
    }
    main {
      padding: 22px clamp(16px, 4vw, 44px) 36px;
      display: grid;
      gap: 18px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 12px;
    }
    .tile {
      min-height: 96px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      white-space: nowrap;
    }
    .value {
      margin-top: 10px;
      font-size: clamp(24px, 4vw, 42px);
      font-weight: 800;
      line-height: 1;
    }
    .unit { color: var(--muted); font-size: 15px; margin-left: 3px; }
    .delta { margin-top: 8px; color: var(--muted); font-size: 13px; }
    .chart-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-height: 360px;
    }
    canvas { width: 100%; height: 320px; display: block; }
    .bottom {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr);
      gap: 18px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    th { color: var(--muted); font-weight: 700; background: #fbfbf8; }
    tr:last-child td { border-bottom: 0; }
    pre {
      margin: 0;
      min-height: 210px;
      max-height: 330px;
      overflow: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #101815;
      color: #d8f3e7;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    @media (max-width: 980px) {
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .bottom { grid-template-columns: 1fr; }
      header { align-items: start; flex-direction: column; }
    }
    @media (max-width: 560px) {
      .stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VocalMorph Live</h1>
      <div class="sub" id="source">Waiting for metrics...</div>
    </div>
    <div class="status" id="status">LIVE</div>
  </header>
  <main>
    <section class="stats">
      <div class="tile">
        <div class="label">Epoch</div>
        <div class="value" id="epoch">--</div>
        <div class="delta" id="updated">--</div>
      </div>
      <div class="tile">
        <div class="label">Target MAE</div>
        <div class="value"><span id="metric">--</span><span class="unit">cm</span></div>
        <div class="delta" id="target">target -- cm</div>
      </div>
      <div class="tile">
        <div class="label">Best Target MAE</div>
        <div class="value"><span id="best">--</span><span class="unit">cm</span></div>
        <div class="delta" id="remaining">--</div>
      </div>
      <div class="tile">
        <div class="label">Speaker MAE</div>
        <div class="value"><span id="speaker">--</span><span class="unit">cm</span></div>
        <div class="delta">validation speaker level</div>
      </div>
      <div class="tile">
        <div class="label">Guarded MAE</div>
        <div class="value"><span id="guarded">--</span><span class="unit">cm</span></div>
        <div class="delta">slice-aware monitor</div>
      </div>
    </section>
    <section class="chart-wrap">
      <canvas id="chart" width="1400" height="520"></canvas>
    </section>
    <section class="bottom">
      <table>
        <thead>
          <tr>
            <th>Epoch</th>
            <th>MAE</th>
            <th>Speaker</th>
            <th>Guarded</th>
            <th>Loss</th>
            <th>LR</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <pre id="log"></pre>
    </section>
  </main>
  <script>
    const fmt = (v, digits = 2) => Number.isFinite(v) ? v.toFixed(digits) : "--";

    async function load() {
      try {
        const res = await fetch("/api/metrics", { cache: "no-store" });
        const data = await res.json();
        render(data);
      } catch (err) {
        document.getElementById("status").textContent = "WAIT";
      }
    }

    function render(data) {
      const latest = data.latest || {};
      document.getElementById("status").textContent = latest.reached ? "TARGET" : "LIVE";
      document.getElementById("status").style.color = latest.reached ? "var(--good)" : "var(--accent)";
      document.getElementById("source").textContent = data.metrics_path || "";
      document.getElementById("epoch").textContent = latest.epoch ?? "--";
      document.getElementById("updated").textContent = data.updated_at || "--";
      document.getElementById("metric").textContent = fmt(latest.metric);
      document.getElementById("best").textContent = fmt(data.best_metric);
      document.getElementById("speaker").textContent = fmt(latest.height_mae_speaker);
      document.getElementById("guarded").textContent = fmt(latest.height_mae_speaker_guarded);
      document.getElementById("target").textContent = `${data.metric_name} target ${fmt(data.target)} cm`;
      const remain = Number.isFinite(data.best_metric) && Number.isFinite(data.target)
        ? Math.max(0, data.best_metric - data.target)
        : NaN;
      document.getElementById("remaining").textContent = Number.isFinite(remain)
        ? `${fmt(remain)} cm to go`
        : "--";
      document.getElementById("log").textContent = (data.log_tail || []).join("\\n");
      renderRows(data.records || []);
      drawChart(data.records || [], data.target);
    }

    function renderRows(records) {
      const rows = records.slice(-10).reverse().map((r) => `
        <tr>
          <td>${r.epoch}</td>
          <td>${fmt(r.metric)}</td>
          <td>${fmt(r.height_mae_speaker)}</td>
          <td>${fmt(r.height_mae_speaker_guarded)}</td>
          <td>${fmt(r.val_loss, 4)}</td>
          <td>${fmt(r.lr, 6)}</td>
        </tr>
      `).join("");
      document.getElementById("rows").innerHTML = rows || "<tr><td colspan='6'>Waiting for epochs...</td></tr>";
    }

    function drawChart(records, target) {
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      const pad = { l: 58, r: 28, t: 24, b: 52 };
      const points = records.filter((r) => Number.isFinite(r.metric));
      if (!points.length) {
        ctx.fillStyle = "#607069";
        ctx.font = "24px Segoe UI";
        ctx.fillText("Waiting for metrics", pad.l, h / 2);
        return;
      }
      const xs = points.map((p) => p.epoch);
      const ys = points.map((p) => p.metric).concat(Number.isFinite(target) ? [target] : []);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys) - 0.25;
      const maxY = Math.max(...ys) + 0.25;
      const xOf = (x) => pad.l + ((x - minX) / Math.max(1, maxX - minX)) * (w - pad.l - pad.r);
      const yOf = (y) => pad.t + ((maxY - y) / Math.max(0.01, maxY - minY)) * (h - pad.t - pad.b);

      ctx.strokeStyle = "#d9ded5";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = pad.t + i * (h - pad.t - pad.b) / 5;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
      }

      if (Number.isFinite(target)) {
        const y = yOf(target);
        ctx.strokeStyle = "#b91c1c";
        ctx.setLineDash([10, 8]);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#b91c1c";
        ctx.font = "20px Segoe UI";
        ctx.fillText(`target ${fmt(target)} cm`, pad.l + 10, y - 10);
      }

      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 5;
      ctx.beginPath();
      points.forEach((p, i) => {
        const x = xOf(p.epoch);
        const y = yOf(p.metric);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = "#16201d";
      ctx.font = "18px Segoe UI";
      ctx.fillText(`epoch ${minX}`, pad.l, h - 18);
      ctx.fillText(`epoch ${maxX}`, w - pad.r - 92, h - 18);
      ctx.fillText(`${fmt(maxY)} cm`, 8, pad.t + 8);
      ctx.fillText(`${fmt(minY)} cm`, 8, h - pad.b);
    }

    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live VocalMorph dashboard.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl.")
    parser.add_argument("--log", default=None, help="Optional train stdout log path.")
    parser.add_argument("--metric", default="height_mae", help="Validation metric to track.")
    parser.add_argument("--target", type=float, default=4.0, help="Target value.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records[-limit:]


def metric_from(record: Dict[str, Any], key: str) -> float:
    val = record.get("val", {}).get(key)
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    if record.get("monitor_name") == key:
        val = record.get("monitor_value")
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
    return float("nan")


def finite_min(values: List[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else float("nan")


def clean_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def read_log_tail(path: Path | None, lines: int = 80) -> List[str]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        raw_lines = handle.readlines()[-500:]
    run_markers = [
        idx for idx, line in enumerate(raw_lines) if "[VocalMorph] stdout ->" in line
    ]
    if run_markers:
        raw_lines = raw_lines[run_markers[-1] :]
    return raw_lines[-lines:]


def build_payload(
    metrics_path: Path,
    log_path: Path | None,
    metric_name: str,
    target: float,
    limit: int,
) -> Dict[str, Any]:
    raw_records = read_jsonl(metrics_path, limit)
    records: List[Dict[str, Any]] = []
    for record in raw_records:
        val = record.get("val", {})
        metric = metric_from(record, metric_name)
        item = {
            "epoch": int(record.get("epoch", 0)),
            "metric": clean_number(metric),
            "height_mae": clean_number(metric_from(record, "height_mae")),
            "height_mae_speaker": clean_number(
                metric_from(record, "height_mae_speaker")
            ),
            "height_mae_speaker_guarded": clean_number(
                metric_from(record, "height_mae_speaker_guarded")
            ),
            "val_loss": clean_number(val.get("total", float("nan"))),
            "lr": clean_number(record.get("lr", float("nan"))),
        }
        item["reached"] = item["metric"] is not None and float(item["metric"]) <= target
        records.append(item)

    latest = records[-1] if records else {}
    best_metric = finite_min(
        [float(item["metric"]) for item in records if item["metric"] is not None]
    )
    updated_at = ""
    if metrics_path.exists():
        updated_at = datetime.fromtimestamp(metrics_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return {
        "metrics_path": str(metrics_path),
        "metric_name": metric_name,
        "target": float(target),
        "best_metric": clean_number(best_metric),
        "latest": latest,
        "records": records,
        "updated_at": updated_at,
        "log_tail": [line.rstrip("\n") for line in read_log_tail(log_path)],
    }


class Handler(BaseHTTPRequestHandler):
    metrics_path: Path
    log_path: Path | None
    metric_name: str
    target: float
    limit: int

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/metrics":
            payload = build_payload(
                self.metrics_path,
                self.log_path,
                self.metric_name,
                self.target,
                self.limit,
            )
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> int:
    args = parse_args()
    Handler.metrics_path = Path(args.metrics).resolve()
    Handler.log_path = Path(args.log).resolve() if args.log else None
    Handler.metric_name = str(args.metric)
    Handler.target = float(args.target)
    Handler.limit = int(args.limit)

    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"Live dashboard: http://{args.host}:{args.port}")
    print(f"Metrics: {Handler.metrics_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
