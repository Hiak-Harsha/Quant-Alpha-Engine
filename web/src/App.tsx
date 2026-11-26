import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  Briefcase,
  Compass,
  Microscope,
  RefreshCw,
} from "lucide-react";
import { api } from "./api";
import { BacktestView } from "./components/BacktestView";
import { DeepDiveView } from "./components/DeepDiveView";
import { GuideAuditView } from "./components/GuideAuditView";
import { MarketRadar } from "./components/MarketRadar";
import { PaperDeskView } from "./components/PaperDeskView";
import type { ConfigResponse, ScanResponse } from "./types";

type TabId = "radar" | "deep_dive" | "paper" | "backtest" | "guide";

export default function App() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [universe, setUniverse] = useState("india_market_gold_silver");
  const [horizon, setHorizon] = useState("next_day");
  const [activeTab, setActiveTab] = useState<TabId>("radar");
  const [selectedTicker, setSelectedTicker] = useState("GOLDBEES.NS");

  const [scan, setScan] = useState<ScanResponse | null>(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getConfig()
      .then((cfg) => {
        setConfig(cfg);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const loadScan = async (refresh = false) => {
    setScanLoading(true);
    setError(null);
    try {
      const data = await api.getScan(horizon, universe, refresh);
      setScan(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load scan");
    } finally {
      setScanLoading(false);
    }
  };

  useEffect(() => {
    loadScan(false);
  }, [horizon, universe]);

  const availableTickers = config?.universes[universe] ?? ["GOLDBEES.NS", "SILVERBEES.NS", "RELIANCE.NS", "HAL.NS", "BTC-USD"];
  const session = scan?.market_session ?? config?.market_session;

  return (
    <div>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <Activity size={20} color="#38bdf8" />
            <span>QUANT ALPHA // TERMINAL</span>
            {session && (
              <span className={`badge ${session.is_open ? "badge-live" : "badge-neut"}`}>
                {session.status_label}
              </span>
            )}
          </div>

          <div className="controls">
            <select value={universe} onChange={(e) => setUniverse(e.target.value)}>
              {Object.keys(config?.universes ?? { india_market_gold_silver: [] }).map((u) => (
                <option key={u} value={u}>
                  {u}
                </option>
              ))}
            </select>

            <select value={horizon} onChange={(e) => setHorizon(e.target.value)}>
              {Object.entries(
                config?.horizons ?? {
                  next_day: { display_name: "Next Day / Tomorrow (1D -> 1D)" },
                  swing: { display_name: "Swing / Short-Term (1D -> 5D)" },
                },
              ).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.display_name}
                </option>
              ))}
            </select>

            <button className="btn btn-primary" onClick={() => loadScan(true)} disabled={scanLoading}>
              <RefreshCw size={14} /> {scanLoading ? "Refreshing..." : "Refresh Live Quotes"}
            </button>
          </div>
        </div>
      </header>

      <nav className="nav-bar">
        <button className={`nav-pill ${activeTab === "radar" ? "active" : ""}`} onClick={() => setActiveTab("radar")}>
          <Compass size={15} /> Market Radar &amp; Tomorrow
        </button>
        <button className={`nav-pill ${activeTab === "deep_dive" ? "active" : ""}`} onClick={() => setActiveTab("deep_dive")}>
          <Microscope size={15} /> Signal Deep Dive &amp; Why
        </button>
        <button className={`nav-pill ${activeTab === "paper" ? "active" : ""}`} onClick={() => setActiveTab("paper")}>
          <Briefcase size={15} /> Paper Trading Simulator
        </button>
        <button className={`nav-pill ${activeTab === "backtest" ? "active" : ""}`} onClick={() => setActiveTab("backtest")}>
          <BarChart3 size={15} /> Alpha Proof &amp; Backtest
        </button>
        <button className={`nav-pill ${activeTab === "guide" ? "active" : ""}`} onClick={() => setActiveTab("guide")}>
          <BookOpen size={15} /> Guide, Fees &amp; Data Health
        </button>
      </nav>

      <main className="main-wrap">
        {error && <div className="banner-warn">{error}</div>}

        {activeTab === "radar" && (
          <MarketRadar
            scan={scan}
            loading={scanLoading}
            onSelectTicker={(t) => {
              setSelectedTicker(t);
              setActiveTab("deep_dive");
            }}
            onQuickTrade={(t) => {
              setSelectedTicker(t);
              setActiveTab("paper");
            }}
          />
        )}

        {activeTab === "deep_dive" && (
          <DeepDiveView
            ticker={selectedTicker}
            universe={universe}
            horizon={horizon}
            availableTickers={availableTickers}
            onChangeTicker={setSelectedTicker}
            onSendToPaper={(t) => {
              setSelectedTicker(t);
              setActiveTab("paper");
            }}
          />
        )}

        {activeTab === "paper" && (
          <PaperDeskView
            horizon={horizon}
            universe={universe}
            availableTickers={availableTickers}
            scanRows={scan?.scan_rows ?? []}
            initialTicker={selectedTicker}
          />
        )}

        {activeTab === "backtest" && <BacktestView horizon={horizon} />}

        {activeTab === "guide" && (
          <GuideAuditView horizon={horizon} universe={universe} feeSchedule={config?.fee_schedule ?? {}} />
        )}
      </main>
    </div>
  );
}
