import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { BacktestResponse } from "../types";

interface Props {
  horizon: string;
}

const LINE_COLORS = ["#38bdf8", "#10b981", "#f59e0b", "#a78bfa", "#f43f5e", "#34d399", "#fb923c", "#94a3b8", "#e879f9"];

export function BacktestView({ horizon }: Props) {
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getBacktest(horizon)
      .then((res) => {
        if (!cancelled) setData(res);
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
  }, [horizon]);

  if (loading && !data) return <div className="panel">Loading out-of-sample walk-forward proof...</div>;
  if (error) return <div className="banner-warn">{error}</div>;
  if (!data) return null;

  const eqKeys =
    data.equity_curves.length > 0
      ? Object.keys(data.equity_curves[0]).filter((k) => k !== "date" && k !== "Timestamp")
      : [];

  return (
    <div>
      <div className="panel">
        <div className="panel-head">
          <h3 className="panel-title">Out-of-Sample Walk-Forward Leaderboard — {data.display_name}</h3>
          <span className="muted" style={{ fontSize: "0.78rem" }}>
            Benchmarked vs. Equal-Weight &amp; Vol-Matched Buy &amp; Hold (canonical deduplicated exposures)
          </span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>IC</th>
                <th>Hit %</th>
                <th>Long Base %</th>
                <th>Return %</th>
                <th>Vol %</th>
                <th>Sharpe</th>
                <th>Alpha %/yr</th>
                <th>Beta</th>
                <th>HAC t</th>
                <th>DSR</th>
                <th>Max DD %</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {data.leaderboard.map((row) => (
                <tr key={String(row.strategy)}>
                  <td style={{ fontWeight: 700 }}>{row.strategy}</td>
                  <td className="num">{Number(row.spearman_ic ?? 0).toFixed(4)}</td>
                  <td className="num">{Number(row.hit_rate_pct ?? 0).toFixed(2)}%</td>
                  <td className="num muted">{Number(row.always_long_hit_pct ?? 0).toFixed(2)}%</td>
                  <td className={`num ${Number(row.total_return_pct ?? 0) >= 0 ? "up" : "dn"}`}>
                    {Number(row.total_return_pct ?? 0).toFixed(2)}%
                  </td>
                  <td className="num">{Number(row.ann_vol_pct ?? 0).toFixed(2)}%</td>
                  <td className="num" style={{ fontWeight: 700 }}>{Number(row.sharpe ?? 0).toFixed(3)}</td>
                  <td className={`num ${Number(row.alpha_ann_pct ?? 0) > 0 ? "up" : "dn"}`}>
                    {Number(row.alpha_ann_pct ?? 0) >= 0 ? "+" : ""}
                    {Number(row.alpha_ann_pct ?? 0).toFixed(2)}%
                  </td>
                  <td className="num">{Number(row.beta_to_bh ?? 0).toFixed(3)}</td>
                  <td className="num">{Number(row.hac_t_stat ?? 0).toFixed(2)}</td>
                  <td className="num">{Number(row.dsr_prob ?? 0).toFixed(3)}</td>
                  <td className="num dn">{Number(row.max_drawdown_pct ?? 0).toFixed(2)}%</td>
                  <td>
                    <span className={`badge ${row.gate_passed ? "badge-live" : "badge-gate"}`}>
                      {String(row.edge_verdict)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title">OOS Cumulative Equity Curves (Net of Costs)</h3>
          </div>
          <div style={{ width: "100%", height: "300px" }}>
            <ResponsiveContainer>
              <LineChart data={data.equity_curves}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={30} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {eqKeys.slice(0, 6).map((k, idx) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={LINE_COLORS[idx % LINE_COLORS.length]} dot={false} strokeWidth={2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title">Top 15 Global Feature Importance (Mean |SHAP|)</h3>
          </div>
          <div style={{ width: "100%", height: "300px" }}>
            <ResponsiveContainer>
              <BarChart data={[...data.global_shap].reverse()} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis type="category" dataKey="label" width={175} tick={{ fill: "#cbd5e1", fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                <Bar dataKey="mean_abs_shap" fill="#38bdf8" name="Mean |SHAP|" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3 className="panel-title">Per-Asset Ensemble Out-of-Sample Summary</h3>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>OOF Bars</th>
                <th>IC</th>
                <th>Hit %</th>
                <th>Base Long %</th>
                <th>Hit Edge</th>
                <th>Strat Ret %</th>
                <th>B&amp;H Ret %</th>
                <th>Sharpe</th>
                <th>Max DD %</th>
              </tr>
            </thead>
            <tbody>
              {data.per_asset.map((a) => (
                <tr key={String(a.ticker)}>
                  <td style={{ fontWeight: 700 }}>{a.ticker}</td>
                  <td className="num">{a.oof_bars}</td>
                  <td className="num">{Number(a.ensemble_ic ?? 0).toFixed(4)}</td>
                  <td className="num">{Number(a.hit_rate_pct ?? 0).toFixed(1)}%</td>
                  <td className="num muted">{Number(a.always_long_hit_pct ?? 0).toFixed(1)}%</td>
                  <td className={`num ${Number(a.hit_edge_pct ?? 0) >= 0 ? "up" : "dn"}`}>
                    {Number(a.hit_edge_pct ?? 0) >= 0 ? "+" : ""}
                    {Number(a.hit_edge_pct ?? 0).toFixed(2)}%
                  </td>
                  <td className={`num ${Number(a.strategy_ret_pct ?? 0) >= 0 ? "up" : "dn"}`}>
                    {Number(a.strategy_ret_pct ?? 0).toFixed(2)}%
                  </td>
                  <td className="num muted">{Number(a.buy_hold_ret_pct ?? 0).toFixed(2)}%</td>
                  <td className="num">{Number(a.sharpe ?? 0).toFixed(3)}</td>
                  <td className="num dn">{Number(a.max_dd_pct ?? 0).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
