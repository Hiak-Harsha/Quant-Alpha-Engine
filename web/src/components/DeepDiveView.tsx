import { useEffect, useState } from "react";
import { ArrowRight, Calculator, HelpCircle } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { AssetDetailResponse } from "../types";

interface Props {
  ticker: string;
  universe: string;
  horizon: string;
  availableTickers: string[];
  onChangeTicker: (t: string) => void;
  onSendToPaper: (ticker: string) => void;
}

function renderFormattedText(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((p, idx) => {
    if (p.startsWith("**") && p.endsWith("**")) {
      return (
        <strong key={idx} style={{ color: "#f8fafc" }}>
          {p.slice(2, -2)}
        </strong>
      );
    }
    if (p.startsWith("`") && p.endsWith("`")) {
      return (
        <code
          key={idx}
          style={{
            background: "#0f172a",
            padding: "1px 5px",
            borderRadius: "4px",
            color: "#38bdf8",
            fontSize: "0.8rem",
          }}
        >
          {p.slice(1, -1)}
        </code>
      );
    }
    return <span key={idx}>{p}</span>;
  });
}

export function DeepDiveView({
  ticker,
  universe,
  horizon,
  availableTickers,
  onChangeTicker,
  onSendToPaper,
}: Props) {
  const [customInput, setCustomInput] = useState("");
  const [capital, setCapital] = useState(100000);
  const [detail, setDetail] = useState<AssetDetailResponse | null>(null);
  const [horizonsMatrix, setHorizonsMatrix] = useState<Array<Record<string, string | number | null>>>([]);
  const [loading, setLoading] = useState(false);
  const [matrixLoading, setMatrixLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setHorizonsMatrix([]);
    api
      .getAssetDetail(ticker, horizon, universe, capital)
      .then((res) => {
        if (!cancelled) setDetail(res);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, horizon, universe]);

  const liveScenario = detail
    ? {
        expected_pnl: (capital * detail.report.composite_expected_ret_pct) / 100,
        low_pnl: (capital * detail.report.expected_range_low_pct) / 100,
        high_pnl: (capital * detail.report.expected_range_high_pct) / 100,
        stop_risk: -(capital * Math.abs(detail.report.stop_loss_price - detail.report.last_close)) / Math.max(detail.report.last_close, 1e-8),
        tp_reward: (capital * Math.abs(detail.report.take_profit_price - detail.report.last_close)) / Math.max(detail.report.last_close, 1e-8),
      }
    : null;

  const loadMatrix = () => {
    setMatrixLoading(true);
    api
      .getAssetHorizons(ticker, universe)
      .then((res) => setHorizonsMatrix(res.rows))
      .catch(() => {})
      .finally(() => setMatrixLoading(false));
  };

  const cur = ticker.endsWith(".NS") || ticker.endsWith(".BO") || ticker.startsWith("^NSE") ? "Rs " : "$";

  return (
    <div>
      {/* Ticker Toolbar */}
      <div className="panel" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "0.75rem" }}>
        <div className="controls">
          <span className="muted" style={{ fontSize: "0.82rem", fontWeight: 700 }}>
            ANALYZE ASSET:
          </span>
          <select value={availableTickers.includes(ticker) ? ticker : ""} onChange={(e) => e.target.value && onChangeTicker(e.target.value)}>
            {!availableTickers.includes(ticker) && <option value="">{ticker}</option>}
            {availableTickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <input
            placeholder="Custom ticker (e.g., HAL.NS, BTC-USD)"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value.toUpperCase())}
            onKeyDown={(e) => {
              if (e.key === "Enter" && customInput.trim()) {
                onChangeTicker(customInput.trim());
                setCustomInput("");
              }
            }}
          />
          <button
            className="btn"
            onClick={() => {
              if (customInput.trim()) {
                onChangeTicker(customInput.trim());
                setCustomInput("");
              }
            }}
          >
            Load Symbol
          </button>
        </div>
        <div className="controls">
          <button className="btn" onClick={loadMatrix} disabled={matrixLoading}>
            {matrixLoading ? "Computing 4 Horizons..." : "Compare All 4 Horizons"}
          </button>
          <button className="btn btn-primary" onClick={() => onSendToPaper(ticker)}>
            Trade in Paper Desk <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {error && <div className="banner-warn">{error}</div>}
      {loading && !detail && <div className="panel">Loading deep-dive forecast &amp; SHAP drivers for {ticker}...</div>}

      {detail && (
        <>
          <div style={{ fontSize: "0.82rem", color: "#94a3b8", marginBottom: "0.7rem" }}>
            Closed Bar Used: <strong style={{ color: "#f8fafc" }}>{detail.report.as_of_timestamp}</strong> | Horizon:{" "}
            <strong style={{ color: "#f8fafc" }}>{detail.report.horizon_display}</strong> | Gate Check:{" "}
            <span className="warn">{detail.report.gate_reason}</span>
          </div>

          <div className="kpi-grid">
            <div className="metric-card">
              <div className="metric-label">Gate / Raw View</div>
              <div className="metric-value" style={{ fontSize: "1.08rem" }}>
                {detail.report.verdict}
              </div>
              <div className="metric-sub muted">Raw Model: {detail.report.raw_verdict}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Est. Expected Move</div>
              <div className={`metric-value num ${detail.report.composite_expected_ret_pct >= 0 ? "up" : "dn"}`}>
                {detail.report.composite_expected_ret_pct >= 0 ? "+" : ""}
                {detail.report.composite_expected_ret_pct.toFixed(2)}%
              </div>
              <div className="metric-sub muted">
                Close: {cur}
                {detail.report.last_close.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Upside Prob / Agree</div>
              <div className="metric-value num">{detail.report.prob_up_pct.toFixed(1)}%</div>
              <div className="metric-sub muted">{detail.report.model_agreement_pct.toFixed(0)}% of 6 model families agree</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">GARCH Annual Vol</div>
              <div className="metric-value num">{detail.report.garch_ann_vol_pct.toFixed(1)}%</div>
              <div className="metric-sub muted">21-Bar RV: {detail.report.realized_vol21_pct.toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">ATR Risk Levels</div>
              <div className="metric-value num" style={{ fontSize: "1.02rem" }}>
                TP {detail.report.take_profit_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
              <div className="metric-sub dn">
                Stop {detail.report.stop_loss_price.toLocaleString(undefined, { maximumFractionDigits: 2 })} | Live Alloc{" "}
                {detail.report.suggested_allocation_pct.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* 5 Plain-English WHY Justification Cards */}
          <div className="panel">
            <div className="panel-head">
              <h3 className="panel-title" style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                <HelpCircle size={17} color="#38bdf8" /> Why This Signal? (Objective Evidence &amp; Risk Justification)
              </h3>
            </div>
            <div className="why-grid">
              {detail.justification.map((c) => (
                <div className="why-card" key={c.title}>
                  <div className="why-title">{c.title}</div>
                  <div className="why-body">{renderFormattedText(c.body)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Interactive Price History + Stop/TP Reference Lines & What-If Scenario Calculator */}
          <div className="grid-2">
            <div className="panel">
              <div className="panel-head">
                <h3 className="panel-title">{detail.report.ticker} — Trailing 90-Bar Price &amp; ATR Guardrails</h3>
              </div>
              <div style={{ width: "100%", height: "270px" }}>
                <ResponsiveContainer>
                  <AreaChart data={detail.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={25} />
                    <YAxis domain={["auto", "auto"]} tick={{ fill: "#94a3b8", fontSize: 11 }} width={62} />
                    <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                    <ReferenceLine y={detail.report.take_profit_price} stroke="#10b981" strokeDasharray="4 4" label={{ value: "Target", fill: "#10b981", fontSize: 11 }} />
                    <ReferenceLine y={detail.report.stop_loss_price} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: "Stop", fill: "#f43f5e", fontSize: 11 }} />
                    <Area type="monotone" dataKey="close" stroke="#38bdf8" fill="rgba(56,189,248,0.16)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <h3 className="panel-title" style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                  <Calculator size={17} color="#38bdf8" /> What-If Position Risk &amp; Return Calculator
                </h3>
                <div className="controls">
                  <span className="muted" style={{ fontSize: "0.78rem" }}>Capital ({cur}):</span>
                  <input
                    type="number"
                    min={1000}
                    step={5000}
                    value={capital}
                    onChange={(e) => setCapital(Math.max(0, Number(e.target.value)))}
                    style={{ width: "120px" }}
                  />
                </div>
              </div>
              {liveScenario && (
                <div className="kpi-grid" style={{ marginBottom: 0 }}>
                  <div className="metric-card">
                    <div className="metric-label">Expected Move</div>
                    <div className={`metric-value num ${liveScenario.expected_pnl >= 0 ? "up" : "dn"}`}>
                      {cur}
                      {liveScenario.expected_pnl >= 0 ? "+" : ""}
                      {liveScenario.expected_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                    <div className="metric-sub muted">{detail.report.composite_expected_ret_pct.toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">1-Sigma Band</div>
                    <div className="metric-value num" style={{ fontSize: "1.05rem" }}>
                      {cur}
                      {liveScenario.low_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })} to +
                      {liveScenario.high_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                    <div className="metric-sub muted">
                      [{detail.report.expected_range_low_pct.toFixed(2)}%, {detail.report.expected_range_high_pct.toFixed(2)}%]
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">ATR Stop Risk</div>
                    <div className="metric-value num dn">
                      {cur}
                      {liveScenario.stop_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                    <div className="metric-sub muted">If stopped @ {detail.report.stop_loss_price}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">ATR Target Reward</div>
                    <div className="metric-value num up">
                      +{cur}
                      {liveScenario.tp_reward.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                    <div className="metric-sub muted">If hit @ {detail.report.take_profit_price}</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 6-Model Breakdown & Local SHAP Feature Drivers */}
          <div className="grid-2">
            <div className="panel">
              <div className="panel-head">
                <h3 className="panel-title">6-Model Family Breakdown</h3>
              </div>
              <div style={{ width: "100%", height: "230px", marginBottom: "0.6rem" }}>
                <ResponsiveContainer>
                  <BarChart data={detail.report.model_breakdown}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="model" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                    <ReferenceLine y={0} stroke="#475569" />
                    <Bar dataKey="expected_ret_pct" name="Expected Move %">
                      {detail.report.model_breakdown.map((entry, idx) => (
                        <Cell key={idx} fill={entry.expected_ret_pct >= 0 ? "#10b981" : "#f43f5e"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Model Family</th>
                      <th>Weight</th>
                      <th>Exp. Return</th>
                      <th>Up Prob</th>
                      <th>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.report.model_breakdown.map((m) => (
                      <tr key={m.model}>
                        <td>{m.model}</td>
                        <td className="num">{m.ensemble_weight.toFixed(1)}%</td>
                        <td className={`num ${m.expected_ret_pct >= 0 ? "up" : "dn"}`}>
                          {m.expected_ret_pct >= 0 ? "+" : ""}
                          {m.expected_ret_pct.toFixed(2)}%
                        </td>
                        <td className="num">{m.prob_up_pct.toFixed(1)}%</td>
                        <td>
                          <span className={`badge ${m.signal === "BULLISH" ? "badge-bull" : m.signal === "BEARISH" ? "badge-bear" : "badge-neut"}`}>
                            {m.signal}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <h3 className="panel-title">Top Local Feature Attribution (SHAP)</h3>
              </div>
              {detail.drivers.length > 0 ? (
                <>
                  <div style={{ width: "100%", height: "230px", marginBottom: "0.6rem" }}>
                    <ResponsiveContainer>
                      <BarChart data={[...detail.drivers].reverse()} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <YAxis type="category" dataKey="label" width={165} tick={{ fill: "#cbd5e1", fontSize: 10 }} />
                        <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                        <ReferenceLine x={0} stroke="#475569" />
                        <Bar dataKey="shap_impact" name="SHAP Impact">
                          {[...detail.drivers].reverse().map((d, idx) => (
                            <Cell key={idx} fill={d.shap_impact >= 0 ? "#10b981" : "#f43f5e"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Driver</th>
                          <th>Value</th>
                          <th>Impact</th>
                          <th>Direction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.drivers.map((d) => (
                          <tr key={d.feature}>
                            <td>{d.label}</td>
                            <td className="num muted">{d.value ?? "N/A"}</td>
                            <td className={`num ${d.shap_impact >= 0 ? "up" : "dn"}`}>
                              {d.shap_impact >= 0 ? "+" : ""}
                              {d.shap_impact.toFixed(4)}
                            </td>
                            <td>{d.direction}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="muted">SHAP drivers unavailable for this model family.</div>
              )}
            </div>
          </div>

          {horizonsMatrix.length > 0 && (
            <div className="panel">
              <div className="panel-head">
                <h3 className="panel-title">4-Horizon Consensus Matrix ({ticker})</h3>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th>Gate Status</th>
                      <th>Model View</th>
                      <th>Est. Move %</th>
                      <th>Prob Up %</th>
                      <th>Consensus %</th>
                      <th>GARCH Vol %</th>
                      <th>OOS Sharpe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {horizonsMatrix.map((row) => (
                      <tr key={String(row.horizon_name)}>
                        <td style={{ fontWeight: 700 }}>{row.horizon}</td>
                        <td>
                          <span className="badge badge-gate">{row.verdict}</span>
                        </td>
                        <td>{row.raw_verdict}</td>
                        <td className={`num ${Number(row.composite_ret_pct) >= 0 ? "up" : "dn"}`}>
                          {Number(row.composite_ret_pct) >= 0 ? "+" : ""}
                          {Number(row.composite_ret_pct).toFixed(2)}%
                        </td>
                        <td className="num">{Number(row.prob_up_pct).toFixed(1)}%</td>
                        <td className="num">{Number(row.consensus_pct).toFixed(0)}%</td>
                        <td className="num">{Number(row.garch_vol_pct).toFixed(1)}%</td>
                        <td className="num">{row.oos_sharpe ?? "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
