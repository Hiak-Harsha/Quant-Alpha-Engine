import { useEffect, useState } from "react";
import { api } from "../api";

interface Props {
  horizon: string;
  universe: string;
  feeSchedule: Record<string, number>;
}

export function GuideAuditView({ horizon, universe, feeSchedule }: Props) {
  const [audit, setAudit] = useState<Array<Record<string, string | number | null>>>([]);
  const [news, setNews] = useState<Array<Record<string, string | number | null>>>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api
      .getFeedsAudit(horizon, universe)
      .then((res) => {
        setAudit(res.audit);
        setNews(res.news);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [horizon, universe]);

  return (
    <div>
      <div className="grid-2">
        <div className="why-card">
          <div className="why-title">How to Operate Safely as a Beginner</div>
          <div className="why-body">
            1. Start in <strong>Market Radar &amp; Tomorrow</strong>: inspect top raw upside and downside risk stocks for the Next-Day (1D) or Swing (5D) horizon.
            <br />
            2. Click any stock to open <strong>Signal Deep Dive &amp; Why</strong> and read all 5 objective evidence cards before judging a trade.
            <br />
            3. Use <strong>Paper Trading Desk</strong> to test delivery orders with real Indian fees and grade <strong>Pearson / Spearman Correlation</strong> as markets settle.
            <br />
            4. Never bypass the statistical <strong>PassGate</strong>: automated sizing stays at <strong>0.0%</strong> until OOS Alpha &gt; 0, HAC t &gt;= 2.0, and DSR &gt;= 0.95 pass.
          </div>
        </div>

        <div className="why-card">
          <div className="why-title">Exact Delivery Fee Model Used in Paper Trading</div>
          <div className="why-body">
            • <strong>Equity Brokerage:</strong> {(Number(feeSchedule.india_brokerage_rate ?? 0) * 100).toFixed(2)}% (Zero delivery brokerage)
            <br />
            • <strong>STT:</strong> {(Number(feeSchedule.india_equity_stt_rate ?? 0.001) * 100).toFixed(2)}% Buy+Sell Equities / {(Number(feeSchedule.india_etf_sell_stt_rate ?? 0.00001) * 100).toFixed(3)}% Sell ETFs
            <br />
            • <strong>Stamp Duty (Buy):</strong> {(Number(feeSchedule.india_stamp_duty_buy ?? 0.00015) * 100).toFixed(3)}% | <strong>NSE+SEBI:</strong> 0.00307% + 18% GST
            <br />
            • <strong>CDSL DP Charge (Sell):</strong> Rs {Number(feeSchedule.india_dp_sell_charge ?? 15.34).toFixed(2)} flat per ISIN delivery sell
            <br />
            • <strong>Crypto Spot:</strong> {(Number(feeSchedule.crypto_fee_rate ?? 0.001) * 100).toFixed(2)}% per side
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3 className="panel-title">Live News Sentiment Feed (GDELT / RSS / SEC 8-K / Yahoo)</h3>
        </div>
        {loading && news.length === 0 ? (
          <div className="muted">Loading news &amp; feed validation audit...</div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Published</th>
                  <th>Ticker</th>
                  <th>Feed</th>
                  <th>Sentiment</th>
                  <th>Score</th>
                  <th>Headline</th>
                </tr>
              </thead>
              <tbody>
                {news.map((n, i) => (
                  <tr key={i}>
                    <td className="num muted">{String(n.published_at ?? "").slice(0, 16)}</td>
                    <td style={{ fontWeight: 700 }}>{n.ticker}</td>
                    <td className="muted">{n.feed}</td>
                    <td>
                      <span
                        className={`badge ${
                          String(n.sentiment_label).includes("Bull")
                            ? "badge-bull"
                            : String(n.sentiment_label).includes("Bear")
                              ? "badge-bear"
                              : "badge-neut"
                        }`}
                      >
                        {n.sentiment_label}
                      </span>
                    </td>
                    <td className="num">{Number(n.sentiment_score ?? 0).toFixed(3)}</td>
                    <td style={{ whiteSpace: "normal", minWidth: "320px" }}>{n.title}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3 className="panel-title">Multi-Source Data Feed Integrity Audit</h3>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                {audit.length > 0 &&
                  Object.keys(audit[0]).map((k) => <th key={k}>{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {audit.map((row, idx) => (
                <tr key={idx}>
                  {Object.keys(audit[0] ?? {}).map((k) => (
                    <td key={k} className="num">
                      {String(row[k] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
