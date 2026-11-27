import { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Search, TrendingDown, TrendingUp } from "lucide-react";
import type { ScanResponse } from "../types";

interface Props {
  scan: ScanResponse | null;
  loading: boolean;
  onSelectTicker: (ticker: string) => void;
  onQuickTrade: (ticker: string) => void;
}

function verdictBadge(rawVerdict: string) {
  const v = rawVerdict.toUpperCase();
  if (v.includes("BULLISH")) return "badge badge-bull";
  if (v.includes("BEARISH")) return "badge badge-bear";
  return "badge badge-neut";
}

export function MarketRadar({ scan, loading, onSelectTicker, onQuickTrade }: Props) {
  const [search, setSearch] = useState("");
  const [segment, setSegment] = useState("ALL");
  const [signalFilter, setSignalFilter] = useState("ALL");

  const segments = useMemo(() => {
    if (!scan) return ["ALL"];
    return ["ALL", ...Array.from(new Set(scan.scan_rows.map((r) => r.market))).sort()];
  }, [scan]);

  const filteredRows = useMemo(() => {
    if (!scan) return [];
    return scan.scan_rows.filter((r) => {
      if (segment !== "ALL" && r.market !== segment) return false;
      if (search && !r.ticker.toUpperCase().includes(search.toUpperCase())) return false;
      if (signalFilter === "BULLISH" && !r.raw_verdict.includes("BULLISH")) return false;
      if (signalFilter === "BEARISH" && !r.raw_verdict.includes("BEARISH")) return false;
      if (signalFilter === "GATED_PASS" && !r.gate_passed) return false;
      return true;
    });
  }, [scan, segment, search, signalFilter]);

  if (loading && !scan) {
    return <div className="panel">Loading market radar &amp; closed-bar forecasts...</div>;
  }
  if (!scan) {
    return <div className="panel">No scan data available for this selection.</div>;
  }

  const { gate_info, pulse, gainers, decliners, metals, scan_rows } = scan;
  const rawBull = scan_rows.filter((r) => r.raw_verdict.includes("BULLISH")).length;
  const rawBear = scan_rows.filter((r) => r.raw_verdict.includes("BEARISH")).length;
  const grossAlloc = scan_rows.reduce((acc, r) => acc + Math.abs(r.alloc_pct || 0), 0);

  return (
    <div>
      {/* Market & Precious Metals Pulse Strip */}
      <div className="pulse-grid">
        {pulse.map((p) => (
          <div className="metric-card" key={p.ticker}>
            <div className="metric-label">{p.ticker}</div>
            <div className="metric-value num">{p.formatted}</div>
            <div className={`metric-sub num ${p.chg_pct >= 0 ? "up" : "dn"}`}>
              {p.chg_pct >= 0 ? "+" : ""}
              {p.chg_pct.toFixed(2)}%
            </div>
          </div>
        ))}
      </div>

      {/* Research / PassGate Status Banner */}
      <div className={gate_info.passed ? "banner-ok" : "banner-warn"}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 700, marginBottom: "0.25rem" }}>
          {gate_info.passed ? <CheckCircle2 size={18} color="#10b981" /> : <AlertTriangle size={18} color="#f59e0b" />}
          <span>{gate_info.headline}</span>
        </div>
        <div style={{ fontSize: "0.84rem", color: "#cbd5e1" }}>
          {gate_info.detail}.{" "}
          {!gate_info.passed &&
            "Click any row below to inspect full WHY justification or log into the Paper Trading Simulator."}
        </div>
      </div>

      {/* Summary KPIs */}
      <div className="kpi-grid">
        <div className="metric-card">
          <div className="metric-label">Assets Scanned</div>
          <div className="metric-value num">{scan_rows.length}</div>
          <div className="metric-sub muted">Skipped (short history): {scan.skipped_count}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Raw Bullish / Bearish</div>
          <div className="metric-value num">
            <span className="up">{rawBull}</span> / <span className="dn">{rawBear}</span>
          </div>
          <div className="metric-sub muted">Ungated closed-bar signals</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Live Portfolio Alloc</div>
          <div className="metric-value num">{grossAlloc.toFixed(1)}%</div>
          <div className="metric-sub warn">{gate_info.passed ? "Portfolio Constrained <= 100%" : "Locked 0% by PassGate"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">OOS Benchmark Alpha</div>
          <div className={`metric-value num ${gate_info.alpha_ann_pct > 0 ? "up" : "dn"}`}>
            {gate_info.alpha_ann_pct >= 0 ? "+" : ""}
            {gate_info.alpha_ann_pct.toFixed(2)}%/yr
          </div>
          <div className="metric-sub muted">HAC t = {gate_info.hac_t_stat.toFixed(2)} | DSR = {gate_info.dsr_prob.toFixed(3)}</div>
        </div>
      </div>

      {/* Side-by-Side Potential Profit vs Potential Loss Watchlists */}
      <div className="grid-2">
        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title" style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
              <TrendingUp size={17} color="#10b981" /> Potential Profit Candidates (Raw Upside)
            </h3>
            <span className="badge badge-bull">Click stock for WHY</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Ref Date</th>
                  <th>Prev Close</th>
                  <th>Model View</th>
                  <th>Est. Move</th>
                  <th>Pred. Close</th>
                  <th>Up Prob</th>
                  <th>Agree</th>
                </tr>
              </thead>
              <tbody>
                {gainers.map((r) => (
                  <tr className="clickable" key={r.Ticker} onClick={() => onSelectTicker(r.Ticker)}>
                    <td style={{ fontWeight: 700, color: "#f8fafc" }}>{r.Ticker}</td>
                    <td className="num muted">{r["Bar Date"]}</td>
                    <td className="num">{Number(r["Last Close"]).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                    <td>
                      <span className={verdictBadge(r["Model View"])}>{r["Model View"]}</span>
                    </td>
                    <td className="num up" style={{ fontWeight: 700 }}>
                      +{Number(r["Est. Move %"]).toFixed(2)}%
                    </td>
                    <td className="num up" style={{ fontWeight: 700 }}>
                      {Number(r["Est. Target"] ?? r["Last Close"]).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    <td className="num">{Number(r["Up Prob %"]).toFixed(1)}%</td>
                    <td className="num">{Number(r["Model Agree %"]).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title" style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
              <TrendingDown size={17} color="#f43f5e" /> Potential Loss / Risk Candidates
            </h3>
            <span className="badge badge-gate">Cash Long-Only Guard</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Ref Date</th>
                  <th>Prev Close</th>
                  <th>Est. Move</th>
                  <th>Pred. Close</th>
                  <th>1-Sigma Band</th>
                  <th>Up Prob</th>
                  <th>Execution Rule</th>
                </tr>
              </thead>
              <tbody>
                {decliners.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="muted">
                      No negative expected-return assets on this closed-bar scan.
                    </td>
                  </tr>
                ) : (
                  decliners.map((r) => (
                    <tr className="clickable" key={r.Ticker} onClick={() => onSelectTicker(r.Ticker)}>
                      <td style={{ fontWeight: 700, color: "#f8fafc" }}>{r.Ticker}</td>
                      <td className="num muted">{r["Bar Date"]}</td>
                      <td className="num">{Number(r["Last Close"]).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                      <td className="num dn" style={{ fontWeight: 700 }}>
                        {Number(r["Est. Move %"]).toFixed(2)}%
                      </td>
                      <td className="num dn" style={{ fontWeight: 700 }}>
                        {Number(r["Est. Target"] ?? r["Last Close"]).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </td>
                      <td className="num muted">{r["Expected Range"]}</td>
                      <td className="num">{Number(r["Up Prob %"]).toFixed(1)}%</td>
                      <td style={{ fontSize: "0.76rem", color: "#fbbf24" }}>{r["Execution Note"]}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Gold & Silver ETFs + Bullion Spotlight */}
      {metals.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title">Gold &amp; Silver ETFs + Global Bullion Spotlight</h3>
            <span className="muted" style={{ fontSize: "0.78rem" }}>
              Cluster exposure capped at 25% when live
            </span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Ref Date</th>
                  <th>Prev Close</th>
                  <th>1-Bar Chg</th>
                  <th>Model View</th>
                  <th>Est. Move</th>
                  <th>Pred. Close</th>
                  <th>1-Sigma Price Band</th>
                  <th>Up Prob</th>
                  <th>Ann Vol</th>
                  <th>Gate / Note</th>
                </tr>
              </thead>
              <tbody>
                {metals.map((m) => (
                  <tr className="clickable" key={m.Ticker} onClick={() => onSelectTicker(m.Ticker)}>
                    <td style={{ fontWeight: 700 }}>{m.Ticker}</td>
                    <td className="num muted">{m["Bar Date"]}</td>
                    <td className="num">{Number(m["Last Close"]).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                    <td className={`num ${m["1-Bar Chg %"] >= 0 ? "up" : "dn"}`}>
                      {m["1-Bar Chg %"] >= 0 ? "+" : ""}
                      {Number(m["1-Bar Chg %"]).toFixed(2)}%
                    </td>
                    <td>
                      <span className={verdictBadge(m["Model View"])}>{m["Model View"]}</span>
                    </td>
                    <td className={`num ${m["Est. Move %"] >= 0 ? "up" : "dn"}`}>
                      {m["Est. Move %"] >= 0 ? "+" : ""}
                      {Number(m["Est. Move %"]).toFixed(2)}%
                    </td>
                    <td className={`num ${m["Est. Move %"] >= 0 ? "up" : "dn"}`} style={{ fontWeight: 700 }}>
                      {Number(m["Est. Target"] ?? m["Last Close"]).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    <td className="num muted">{m["Pred. Range Price"] || m["Expected Range"]}</td>
                    <td className="num">{Number(m["Up Prob %"]).toFixed(1)}%</td>
                    <td className="num">{Number(m["Ann Vol %"]).toFixed(1)}%</td>
                    <td style={{ fontSize: "0.76rem", color: "#fbbf24" }}>{m["Execution Note"]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Full Interactive Universe Screener */}
      <div className="panel">
        <div className="panel-head">
          <h3 className="panel-title">Full Interactive Market Screener ({filteredRows.length} Assets)</h3>
          <div className="controls">
            <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
              <Search size={14} style={{ position: "absolute", left: "9px", color: "#94a3b8" }} />
              <input
                style={{ paddingLeft: "1.8rem", width: "170px" }}
                placeholder="Search symbol..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <select value={segment} onChange={(e) => setSegment(e.target.value)}>
              {segments.map((s) => (
                <option key={s} value={s}>
                  {s === "ALL" ? "All Segments" : s}
                </option>
              ))}
            </select>
            <select value={signalFilter} onChange={(e) => setSignalFilter(e.target.value)}>
              <option value="ALL">All Signals</option>
              <option value="BULLISH">Raw Bullish Only</option>
              <option value="BEARISH">Raw Bearish Only</option>
              <option value="GATED_PASS">Edge-Gate Passed</option>
            </select>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Segment</th>
                <th>Bar Date</th>
                <th>Close</th>
                <th>1-Bar %</th>
                <th>Model View</th>
                <th>Est. Move</th>
                <th>Pred. Close</th>
                <th>Up Prob</th>
                <th>Agree</th>
                <th>GARCH Vol</th>
                <th>Regime</th>
                <th>Live Alloc</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((r) => (
                <tr key={r.ticker}>
                  <td
                    style={{ fontWeight: 700, color: "#38bdf8", cursor: "pointer" }}
                    onClick={() => onSelectTicker(r.ticker)}
                  >
                    {r.ticker}
                  </td>
                  <td className="muted">{r.market}</td>
                  <td className="num muted">{r.as_of}</td>
                  <td className="num">{Number(r.close).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                  <td className={`num ${r.chg_1_pct >= 0 ? "up" : "dn"}`}>
                    {r.chg_1_pct >= 0 ? "+" : ""}
                    {r.chg_1_pct.toFixed(2)}%
                  </td>
                  <td>
                    <span className={verdictBadge(r.raw_verdict)}>{r.raw_verdict}</span>
                  </td>
                  <td className={`num ${r.composite_ret_pct >= 0 ? "up" : "dn"}`}>
                    {r.composite_ret_pct >= 0 ? "+" : ""}
                    {r.composite_ret_pct.toFixed(2)}%
                  </td>
                  <td className={`num ${r.composite_ret_pct >= 0 ? "up" : "dn"}`} style={{ fontWeight: 700 }}>
                    {(r.close * (1 + r.composite_ret_pct / 100)).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className="num">{r.prob_up_pct.toFixed(1)}%</td>
                  <td className="num">{r.consensus_pct.toFixed(0)}%</td>
                  <td className="num">{r.garch_vol_pct.toFixed(1)}%</td>
                  <td style={{ fontSize: "0.76rem" }}>{r.regime}</td>
                  <td className="num">{r.alloc_pct.toFixed(1)}%</td>
                  <td>
                    <div style={{ display: "flex", gap: "0.35rem" }}>
                      <button className="btn" onClick={() => onSelectTicker(r.ticker)}>
                        Why
                      </button>
                      <button className="btn btn-primary" onClick={() => onQuickTrade(r.ticker)}>
                        Paper Trade
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
