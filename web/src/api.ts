import type {
  AssetDetailResponse,
  BacktestResponse,
  ConfigResponse,
  CorrelationResponse,
  OrderPreviewResponse,
  PaperAccount,
  PaperPortfolioResponse,
  ScanResponse,
} from "./types";

async function reqJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) msg = body.detail;
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

export const api = {
  getConfig: () => reqJson<ConfigResponse>("/api/config"),
  getScan: (horizon: string, universe: string, refresh = false) =>
    reqJson<ScanResponse>(
      `/api/scan?horizon=${encodeURIComponent(horizon)}&universe=${encodeURIComponent(universe)}&refresh=${refresh}`,
    ),
  getAssetDetail: (ticker: string, horizon: string, universe: string, capital = 100000) =>
    reqJson<AssetDetailResponse>(
      `/api/asset/${encodeURIComponent(ticker)}?horizon=${encodeURIComponent(horizon)}&universe=${encodeURIComponent(universe)}&capital=${capital}`,
    ),
  getAssetHorizons: (ticker: string, universe: string) =>
    reqJson<{ ticker: string; rows: Array<Record<string, string | number | null>> }>(
      `/api/asset/${encodeURIComponent(ticker)}/horizons?universe=${encodeURIComponent(universe)}`,
    ),
  getBacktest: (horizon: string) =>
    reqJson<BacktestResponse>(`/api/backtest?horizon=${encodeURIComponent(horizon)}`),
  getFeedsAudit: (horizon: string, universe: string) =>
    reqJson<{ audit: Array<Record<string, string | number | null>>; news: Array<Record<string, string | number | null>> }>(
      `/api/feeds/audit?horizon=${encodeURIComponent(horizon)}&universe=${encodeURIComponent(universe)}`,
    ),
  listAccounts: () => reqJson<{ accounts: PaperAccount[] }>("/api/paper/accounts"),
  createAccount: (owner_name: string, starting_cash: number) =>
    reqJson<{ account: PaperAccount; accounts: PaperAccount[] }>("/api/paper/accounts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ owner_name, starting_cash }),
    }),
  resetAccount: (account_id: string, starting_cash: number) =>
    reqJson<{ account: PaperAccount }>(
      `/api/paper/accounts/${encodeURIComponent(account_id)}/reset?starting_cash=${starting_cash}`,
      { method: "POST" },
    ),
  getPortfolio: (account_id: string, horizon: string, universe: string) =>
    reqJson<PaperPortfolioResponse>(
      `/api/paper/portfolio?account_id=${encodeURIComponent(account_id)}&horizon=${encodeURIComponent(horizon)}&universe=${encodeURIComponent(universe)}`,
    ),
  previewOrder: (account_id: string, ticker: string, side: string, qty: number, price: number) =>
    reqJson<OrderPreviewResponse>("/api/paper/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ account_id, ticker, side, qty, price }),
    }),
  executeOrder: (payload: {
    account_id: string;
    ticker: string;
    side: string;
    qty: number;
    price: number;
    horizon: string;
    stop_loss: number;
    take_profit: number;
    signal_verdict: string;
    signal_ret_pct: number;
  }) =>
    reqJson<{ ok: boolean; message: string }>("/api/paper/order", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  logSignals: (horizon: string, universe: string) =>
    reqJson<{ count: number }>(
      `/api/paper/log-signals?horizon=${encodeURIComponent(horizon)}&universe=${encodeURIComponent(universe)}`,
      { method: "POST" },
    ),
  evaluateSignals: () =>
    reqJson<{ checked_open: number; newly_matured: number; still_open: number }>("/api/paper/evaluate", {
      method: "POST",
    }),
  getCorrelation: () => reqJson<CorrelationResponse>("/api/paper/correlation"),
};
