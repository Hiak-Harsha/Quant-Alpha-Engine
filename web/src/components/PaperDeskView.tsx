import { useEffect, useState } from "react";
import { CheckCircle2, RefreshCw, ShieldAlert, UserPlus, Wallet } from "lucide-react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type {
  CorrelationResponse,
  OrderPreviewResponse,
  PaperAccount,
  PaperPortfolioResponse,
  ScanRow,
} from "../types";

interface Props {
  horizon: string;
  universe: string;
  availableTickers: string[];
  scanRows: ScanRow[];
  initialTicker: string;
}

export function PaperDeskView({ horizon, universe, availableTickers, scanRows, initialTicker }: Props) {
  const [accounts, setAccounts] = useState<PaperAccount[]>([]);
  const [accountId, setAccountId] = useState("default_student");
  const [newOwner, setNewOwner] = useState("");
  const [startCap, setStartCap] = useState(100000);

  const [portfolio, setPortfolio] = useState<PaperPortfolioResponse | null>(null);
  const [corr, setCorr] = useState<CorrelationResponse | null>(null);

  const [ordTicker, setOrdTicker] = useState(initialTicker);
  const [ordSide, setOrdSide] = useState<"BUY" | "SELL">("BUY");
  const [ordQty, setOrdQty] = useState<number>(5);
  const [ordPrice, setOrdPrice] = useState<number>(100);
  const [preview, setPreview] = useState<OrderPreviewResponse | null>(null);

  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setOrdTicker(initialTicker);
  }, [initialTicker]);

  useEffect(() => {
    const match = scanRows.find((r) => r.ticker === ordTicker);
    if (match && match.close > 0) {
      setOrdPrice(Number(match.close.toFixed(2)));
      const isCrypto = ordTicker.endsWith("-USD") || ordTicker.endsWith("-USDT");
      setOrdQty(isCrypto ? 0.05 : Math.max(1, Math.floor(10000 / match.close)));
    }
  }, [ordTicker, scanRows]);

  const reloadDesk = async (acc = accountId) => {
    try {
      const [accRes, portRes, corrRes] = await Promise.all([
        api.listAccounts(),
        api.getPortfolio(acc, horizon, universe),
        api.getCorrelation(),
      ]);
      setAccounts(accRes.accounts);
      setPortfolio(portRes);
      setCorr(corrRes);
    } catch (err) {
      setErrMsg(err instanceof Error ? err.message : "Failed loading paper desk");
    }
  };

  useEffect(() => {
    reloadDesk(accountId);
  }, [accountId, horizon, universe]);

  useEffect(() => {
    if (ordQty > 0 && ordPrice > 0) {
      api
        .previewOrder(accountId, ordTicker, ordSide, ordQty, ordPrice)
        .then(setPreview)
        .catch(() => setPreview(null));
    }
  }, [accountId, ordTicker, ordSide, ordQty, ordPrice, portfolio?.cash_balance]);

  const handleCreateOrReset = async () => {
    setErrMsg(null);
    setStatusMsg(null);
    if (newOwner.trim()) {
      const res = await api.createAccount(newOwner.trim(), startCap);
      setAccounts(res.accounts);
      setAccountId(res.account.account_id);
      setNewOwner("");
      setStatusMsg(`Created simulated account '${res.account.owner_name}' with Rs ${startCap.toLocaleString()}.`);
    } else {
      await api.resetAccount(accountId, startCap);
      setStatusMsg(`Reset '${accountId}' back to Rs ${startCap.toLocaleString()}.`);
      await reloadDesk(accountId);
    }
  };

  const handleExecuteOrder = async () => {
    setErrMsg(null);
    setStatusMsg(null);
    setBusy(true);
    try {
      const match = scanRows.find((r) => r.ticker === ordTicker);
      const res = await api.executeOrder({
        account_id: accountId,
        ticker: ordTicker,
        side: ordSide,
        qty: ordQty,
        price: ordPrice,
        horizon,
        stop_loss: match ? match.stop_loss : ordPrice * 0.96,
        take_profit: match ? match.take_profit : ordPrice * 1.06,
        signal_verdict: match ? match.raw_verdict : "",
        signal_ret_pct: match ? match.composite_ret_pct : 0,
      });
      setStatusMsg(res.message);
      await reloadDesk(accountId);
    } catch (err) {
      setErrMsg(err instanceof Error ? err.message : "Order rejected");
    } finally {
      setBusy(false);
    }
  };

  const handleQuickSellPosition = async (ticker: string, qty: number, livePrice: number) => {
    setErrMsg(null);
    setStatusMsg(null);
    try {
      const res = await api.executeOrder({
        account_id: accountId,
        ticker,
        side: "SELL",
        qty,
        price: livePrice,
        horizon,
        stop_loss: 0,
        take_profit: 0,
        signal_verdict: "MANUAL EXIT",
        signal_ret_pct: 0,
      });
      setStatusMsg(res.message);
      await reloadDesk(accountId);
    } catch (err) {
      setErrMsg(err instanceof Error ? err.message : "Sell failed");
    }
  };

  return (
    <div>
      {/* Account & Tester Switcher */}
      <div className="panel" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "0.75rem" }}>
        <div className="controls">
          <Wallet size={17} color="#38bdf8" />
          <select value={accountId} onChange={(e) => setAccountId(e.target.value)}>
            {accounts.map((a) => (
              <option key={a.account_id} value={a.account_id}>
                {a.owner_name} ({a.account_id})
              </option>
            ))}
          </select>
          <input
            placeholder="Invite tester / new portfolio name"
            value={newOwner}
            onChange={(e) => setNewOwner(e.target.value)}
            style={{ width: "210px" }}
          />
          <input
            type="number"
            min={10000}
            step={10000}
            value={startCap}
            onChange={(e) => setStartCap(Number(e.target.value))}
            style={{ width: "120px" }}
          />
          <button className="btn" onClick={handleCreateOrReset}>
            <UserPlus size={14} /> {newOwner.trim() ? "Create Tester Account" : "Reset Account Cash"}
          </button>
        </div>
        <button className="btn" onClick={() => reloadDesk(accountId)}>
          <RefreshCw size={14} /> Sync Prices
        </button>
      </div>

      {statusMsg && (
        <div className="banner-ok" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <CheckCircle2 size={17} color="#10b981" /> {statusMsg}
        </div>
      )}
      {errMsg && (
        <div className="banner-warn" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <ShieldAlert size={17} color="#f59e0b" /> {errMsg}
        </div>
      )}

      {/* Portfolio KPIs */}
      {portfolio && (
        <div className="kpi-grid">
          <div className="metric-card">
            <div className="metric-label">Total Equity</div>
            <div className="metric-value num">Rs {portfolio.total_equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            <div className={`metric-sub num ${portfolio.net_return_pct >= 0 ? "up" : "dn"}`}>
              {portfolio.net_return_pct >= 0 ? "+" : ""}
              {portfolio.net_return_pct.toFixed(2)}% vs Start
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Available Cash</div>
            <div className="metric-value num">Rs {portfolio.cash_balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            <div className="metric-sub muted">Delivery Long-Only Wallet</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Holdings Value</div>
            <div className="metric-value num">Rs {portfolio.market_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            <div className="metric-sub muted">Invested Cost: Rs {portfolio.invested_cost.toLocaleString()}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Unrealized P&amp;L</div>
            <div className={`metric-value num ${portfolio.unrealized_pnl >= 0 ? "up" : "dn"}`}>
              Rs {portfolio.unrealized_pnl >= 0 ? "+" : ""}
              {portfolio.unrealized_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Realized P&amp;L</div>
            <div className={`metric-value num ${portfolio.realized_pnl >= 0 ? "up" : "dn"}`}>
              Rs {portfolio.realized_pnl >= 0 ? "+" : ""}
              {portfolio.realized_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Real Fees Paid</div>
            <div className="metric-value num warn">Rs {portfolio.total_fees_paid.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            <div className="metric-sub muted">STT + Stamp + NSE + GST + DP</div>
          </div>
        </div>
      )}

      <div className="grid-2">
        {/* Left: Real-World Order Ticket */}
        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title">Simulated Order Ticket (Real Indian Fee Engine)</h3>
          </div>
          <div className="controls" style={{ marginBottom: "0.75rem" }}>
            <select value={ordTicker} onChange={(e) => setOrdTicker(e.target.value)} style={{ flex: 1 }}>
              {availableTickers.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <button
              className={`btn ${ordSide === "BUY" ? "btn-primary" : ""}`}
              onClick={() => setOrdSide("BUY")}
            >
              BUY
            </button>
            <button
              className={`btn ${ordSide === "SELL" ? "btn-primary" : ""}`}
              style={ordSide === "SELL" ? { background: "#e11d48", borderColor: "#f43f5e" } : undefined}
              onClick={() => setOrdSide("SELL")}
            >
              SELL
            </button>
          </div>

          <div className="grid-2" style={{ marginBottom: "0.75rem" }}>
            <div>
              <div className="metric-label">Quantity</div>
              <input
                type="number"
                min={0.001}
                step={ordTicker.endsWith("-USD") ? 0.01 : 1}
                value={ordQty}
                onChange={(e) => setOrdQty(Math.max(0, Number(e.target.value)))}
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <div className="metric-label">Execution Price</div>
              <input
                type="number"
                min={0.01}
                step={0.1}
                value={ordPrice}
                onChange={(e) => setOrdPrice(Math.max(0, Number(e.target.value)))}
                style={{ width: "100%" }}
              />
            </div>
          </div>

          {preview && (
            <div className={preview.valid ? "why-card" : "banner-warn"} style={{ marginBottom: "0.75rem" }}>
              {preview.valid ? (
                <div style={{ fontSize: "0.83rem", lineHeight: 1.55 }}>
                  <strong>Order Value:</strong> Rs {preview.notional.toLocaleString(undefined, { minimumFractionDigits: 2 })} |{" "}
                  <strong>Total Broker Charges:</strong> Rs {preview.fees.total_fee.toFixed(2)} ({preview.fees.fee_bps.toFixed(1)} bps)
                  <br />
                  <span className="muted">
                    STT: {preview.fees.stt.toFixed(2)} | Stamp: {preview.fees.stamp_duty.toFixed(2)} | NSE+SEBI:{" "}
                    {preview.fees.exchange_sebi.toFixed(2)} | GST: {preview.fees.gst.toFixed(2)} | DP Sell:{" "}
                    {preview.fees.dp_charge.toFixed(2)}
                  </span>
                  <br />
                  <strong style={{ color: "#38bdf8" }}>Min Breakeven Move (Buy+Sell): +{preview.breakeven_move_pct.toFixed(3)}%</strong>
                </div>
              ) : (
                <div style={{ fontSize: "0.84rem" }}>{preview.reason}</div>
              )}
            </div>
          )}

          <button
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center", padding: "0.6rem" }}
            disabled={!preview?.valid || busy}
            onClick={handleExecuteOrder}
          >
            {busy ? "Executing..." : `Execute Virtual ${ordSide} (${ordTicker})`}
          </button>

          {/* Open Virtual Positions */}
          <h4 style={{ marginTop: "1rem", marginBottom: "0.5rem", fontSize: "0.88rem" }}>Open Virtual Holdings</h4>
          {portfolio && portfolio.positions.length > 0 ? (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Qty</th>
                    <th>Avg</th>
                    <th>LTP</th>
                    <th>P&amp;L</th>
                    <th>Ret %</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map((pos) => (
                    <tr key={pos.Ticker}>
                      <td style={{ fontWeight: 700 }}>{pos.Ticker}</td>
                      <td className="num">{pos.Qty}</td>
                      <td className="num">{pos["Avg Buy"].toFixed(2)}</td>
                      <td className="num">{pos["Live Price"].toFixed(2)}</td>
                      <td className={`num ${pos["Unrealized P&L"] >= 0 ? "up" : "dn"}`}>
                        {pos["Unrealized P&L"] >= 0 ? "+" : ""}
                        {pos["Unrealized P&L"].toFixed(2)}
                      </td>
                      <td className={`num ${pos["Return %"] >= 0 ? "up" : "dn"}`}>
                        {pos["Return %"] >= 0 ? "+" : ""}
                        {pos["Return %"].toFixed(2)}%
                      </td>
                      <td>
                        <button
                          className="btn"
                          onClick={() => handleQuickSellPosition(pos.Ticker, pos.Qty, pos["Live Price"])}
                        >
                          Sell All
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="muted" style={{ fontSize: "0.83rem" }}>
              No open holdings in this account yet.
            </div>
          )}
        </div>

        {/* Right: Daily Prediction vs Market Correlation Tracker */}
        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title">Daily Prediction vs. Market Variation Correlation</h3>
          </div>
          <div className="controls" style={{ marginBottom: "0.8rem" }}>
            <button
              className="btn btn-primary"
              onClick={async () => {
                const res = await api.logSignals(horizon, universe);
                setStatusMsg(`Logged/verified ${res.count} zero-leakage closed-bar forecasts.`);
                await reloadDesk(accountId);
              }}
            >
              Log Closed-Bar Signals Today
            </button>
            <button
              className="btn"
              onClick={async () => {
                const res = await api.evaluateSignals();
                setStatusMsg(
                  `Evaluated ${res.checked_open} open signals -> ${res.newly_matured} newly matured, ${res.still_open} pending maturity.`,
                );
                await reloadDesk(accountId);
              }}
            >
              Grade Matured Signals vs. Market
            </button>
          </div>

          {corr && (
            <>
              <div className="kpi-grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}>
                <div className="metric-card">
                  <div className="metric-label">Matured</div>
                  <div className="metric-value num">{corr.n_matured}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Pearson r</div>
                  <div className="metric-value num">{corr.pearson_r ?? "Pending"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Spearman IC</div>
                  <div className="metric-value num">{corr.spearman_ic ?? "Pending"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Hit vs Long %</div>
                  <div className="metric-value num">{corr.hit_rate_pct !== null ? `${corr.hit_rate_pct}%` : "Pending"}</div>
                  {corr.always_long_pct !== null && (
                    <div className="metric-sub muted">Base: {corr.always_long_pct}%</div>
                  )}
                </div>
              </div>

              {corr.points.length >= 2 ? (
                <div style={{ width: "100%", height: "260px" }}>
                  <ResponsiveContainer>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis type="number" dataKey="composite_expected_ret_pct" name="Predicted %" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <YAxis type="number" dataKey="actual_asset_ret_pct" name="Actual %" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <ReferenceLine x={0} stroke="#475569" />
                      <ReferenceLine y={0} stroke="#475569" />
                      <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                      <Scatter name="Settled Predictions" data={corr.points} fill="#38bdf8" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="why-card">
                  <div className="why-title">How Forward Accuracy Tracking Works</div>
                  <div className="why-body">
                    Click <strong>Log Closed-Bar Signals Today</strong> to snapshot model predictions into SQLite. After the market horizon completes (1 day for Next-Day, 5 days for Swing), click <strong>Grade Matured Signals vs. Market</strong> to verify real Pearson correlation &amp; hit rate edge.
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Simulated Order Blotter */}
      {portfolio && portfolio.orders.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h3 className="panel-title">Virtual Order Blotter &amp; Broker Fee Audit</h3>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Time (UTC)</th>
                  <th>Ticker</th>
                  <th>Side</th>
                  <th>Qty</th>
                  <th>Fill Price</th>
                  <th>Notional</th>
                  <th>Total Fee</th>
                  <th>Realized P&amp;L</th>
                  <th>Entry Signal</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.orders.map((o) => (
                  <tr key={o.id}>
                    <td className="num muted">{o.timestamp_utc}</td>
                    <td style={{ fontWeight: 700 }}>{o.ticker}</td>
                    <td>
                      <span className={`badge ${o.side === "BUY" ? "badge-bull" : "badge-bear"}`}>{o.side}</span>
                    </td>
                    <td className="num">{o.qty}</td>
                    <td className="num">{o.fill_price.toFixed(2)}</td>
                    <td className="num">{o.notional.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                    <td className="num warn">{o.total_fee.toFixed(2)}</td>
                    <td className={`num ${o.realized_pnl >= 0 ? "up" : "dn"}`}>
                      {o.realized_pnl !== 0 ? `${o.realized_pnl >= 0 ? "+" : ""}${o.realized_pnl.toFixed(2)}` : "-"}
                    </td>
                    <td className="muted">{o.signal_verdict}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
