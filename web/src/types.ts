export interface MarketSession {
  ist_time: string;
  is_open: boolean;
  status_label: string;
}

export interface HorizonInfo {
  name: string;
  display_name: string;
  interval: string;
  forecast_horizon: number;
  description: string;
}

export interface ConfigResponse {
  universes: Record<string, string[]>;
  horizons: Record<string, HorizonInfo>;
  fee_schedule: Record<string, number>;
  market_session: MarketSession;
}

export interface GateInfo {
  passed: boolean;
  headline: string;
  detail: string;
  alpha_ann_pct: number;
  hac_t_stat: number;
  dsr_prob: number;
  sharpe: number;
}

export interface PulseItem {
  ticker: string;
  value: number;
  formatted: string;
  chg_pct: number;
}

export interface WatchlistRow {
  Ticker: string;
  Segment?: string;
  "Bar Date": string;
  "Last Close": number;
  "1-Bar Chg %": number;
  "Model View": string;
  "Est. Move %": number;
  "Est. Target"?: number;
  "Pred. Range Price"?: string;
  "Expected Range"?: string;
  "Up Prob %": number;
  "Model Agree %": number;
  "Ann Vol %": number;
  "Risk Stop"?: number;
  "Target Price"?: number;
  "Live Alloc %": number;
  "Gate Status": string;
  "Execution Note": string;
}

export interface ScanRow {
  ticker: string;
  market: string;
  as_of: string;
  buzz_score: number;
  news_mentions: number;
  close: number;
  chg_1_pct: number;
  verdict: string;
  raw_verdict: string;
  gate_passed: boolean;
  cash_long_only: boolean;
  composite_ret_pct: number;
  quant_ret_pct: number;
  range_low_pct: number;
  range_high_pct: number;
  prob_up_pct: number;
  consensus_pct: number;
  garch_vol_pct: number;
  regime: string;
  alloc_pct: number;
  stop_loss: number;
  take_profit: number;
}

export interface ScanResponse {
  horizon: string;
  universe: string;
  market_session: MarketSession;
  gate_info: GateInfo;
  pulse: PulseItem[];
  gainers: WatchlistRow[];
  decliners: WatchlistRow[];
  metals: WatchlistRow[];
  scan_rows: ScanRow[];
  skipped_count: number;
}

export interface AssetDetailResponse {
  report: {
    ticker: string;
    horizon_name: string;
    horizon_display: string;
    as_of_timestamp: string;
    last_close: number;
    verdict: string;
    raw_verdict: string;
    gate_passed: boolean;
    gate_reason: string;
    quant_expected_ret_pct: number;
    composite_expected_ret_pct: number;
    expected_range_low_pct: number;
    expected_range_high_pct: number;
    prob_up_pct: number;
    model_agreement_pct: number;
    cost_hurdle_pct: number;
    suggested_allocation_pct: number;
    stop_loss_price: number;
    take_profit_price: number;
    garch_ann_vol_pct: number;
    realized_vol21_pct: number;
    vix_level: number | null;
    regime_label: string;
    regime_probs: Record<string, number>;
    news_summary: Record<string, number | string>;
    model_breakdown: Array<{
      model: string;
      ensemble_weight: number;
      expected_ret_pct: number;
      prob_up_pct: number;
      signal: string;
    }>;
    backtest_track_record: Record<string, number | string | boolean>;
  };
  cash_long_only: boolean;
  justification: Array<{ title: string; body: string }>;
  scenario: {
    capital: number;
    expected_pnl: number;
    range_low_pnl: number;
    range_high_pnl: number;
    stop_loss_risk: number;
    take_profit_gain: number;
    gated_capital: number;
  };
  drivers: Array<{ feature: string; label: string; value: number | null; shap_impact: number; direction: string }>;
  history: Array<{ date: string; close: number; high: number; low: number }>;
}

export interface PaperAccount {
  account_id: string;
  owner_name: string;
  starting_cash: number;
  cash_balance: number;
}

export interface PaperPortfolioResponse {
  account: PaperAccount;
  cash_balance: number;
  invested_cost: number;
  market_value: number;
  total_equity: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_fees_paid: number;
  net_return_pct: number;
  positions: Array<{
    Ticker: string;
    Qty: number;
    "Avg Buy": number;
    "Live Price": number;
    Invested: number;
    "Market Value": number;
    "Unrealized P&L": number;
    "Return %": number;
    "Stop Loss": number;
    "Take Profit": number;
    "Entry View": string;
  }>;
  orders: Array<{
    id: number;
    timestamp_utc: string;
    ticker: string;
    side: string;
    qty: number;
    fill_price: number;
    notional: number;
    total_fee: number;
    realized_pnl: number;
    signal_verdict: string;
  }>;
}

export interface OrderPreviewResponse {
  valid: boolean;
  reason: string;
  notional: number;
  fees: {
    brokerage: number;
    stt: number;
    stamp_duty: number;
    exchange_sebi: number;
    gst: number;
    dp_charge: number;
    total_fee: number;
    fee_bps: number;
  };
  breakeven_move_pct: number;
}

export interface CorrelationResponse {
  n_matured: number;
  pearson_r: number | null;
  spearman_ic: number | null;
  hit_rate_pct: number | null;
  always_long_pct: number | null;
  points: Array<{
    ticker: string;
    horizon_name: string;
    as_of_timestamp: string;
    verdict: string;
    composite_expected_ret_pct: number;
    actual_asset_ret_pct: number;
  }>;
  ledger_summary: Record<string, number | string | null>;
}

export interface BacktestResponse {
  horizon: string;
  display_name: string;
  leaderboard: Array<Record<string, number | string | boolean | null>>;
  equity_curves: Array<Record<string, number | string>>;
  drawdown_curves: Array<Record<string, number | string>>;
  per_asset: Array<Record<string, number | string | boolean | null>>;
  global_shap: Array<{ feature: string; label: string; mean_abs_shap: number }>;
  ensemble_weights: Record<string, number>;
}
