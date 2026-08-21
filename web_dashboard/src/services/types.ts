/**
 * QuantSphereX Institutional Terminal — TypeScript Data Models & DTOs
 */

export interface HealthStatusResponse {
  status: string;
  environment: string;
  version: string;
  timestamp: string;
  uptime_seconds: number;
}

export interface UserProfileResponse {
  id: number;
  username: string;
  email: string;
  role: 'admin' | 'analyst' | 'trader' | 'guest';
  is_active: boolean;
  created_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_in_seconds: number;
  username: string;
  role: string;
}

export interface StockItem {
  symbol: string;
  name: string;
  sector: string;
  universe: string;
}

export interface StockQuoteResponse {
  symbol: string;
  close_price: number;
  change_pct: number;
  volume: number;
  high_52w: number;
  low_52w: number;
  pe_ratio?: number;
  roe?: number;
  market_cap_bn?: number;
}

export interface PositionItem {
  symbol: string;
  shares: number;
  avg_price: number;
  current_price: number;
  market_value: number;
  current_weight: number;
  target_weight: number;
  hysteresis_status: 'KEPT' | 'NEW_ENTRY' | 'EXIT';
}

export interface PortfolioSummaryResponse {
  portfolio_name: string;
  total_value: number;
  cash_balance: number;
  benchmark: string;
  positions_count: number;
  top_5_concentration_pct: number;
  annualized_turnover: number;
  positions: PositionItem[];
}

export interface RebalanceTradeItem {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  shares_delta: number;
  target_weight: number;
  estimated_value: number;
}

export interface RebalanceResponse {
  total_trades: number;
  estimated_turnover_pct: number;
  estimated_transaction_cost: number;
  trades: RebalanceTradeItem[];
}

export interface RiskAuditResponse {
  var_95: number;
  cvar_95: number;
  tail_risk_ratio: number;
  top_5_concentration_pct: number;
  effective_n_positions: number;
  mandate_met: boolean;
  risk_grade: string;
}

export interface PredictionItem {
  symbol: string;
  predicted_return: number;
  confidence_score: number;
  uncertainty_std: number;
  rank_decile: number;
  signal_direction: 'BULLISH' | 'BEARISH';
  shap_top_driver: string;
}

export interface ExplainabilityDetailResponse {
  symbol: string;
  predicted_return: number;
  confidence_score: number;
  shap_values: Record<string, number>;
  key_drivers: string[];
  narrative: string;
}

export interface ResearchAlphaItem {
  symbol: string;
  alpha_score: number;
  conviction_level: string;
  primary_factor: string;
  target_horizon_days: number;
  summary_memo: string;
}

export interface FeatureMeta {
  name: string;
  category: string;
  data_type: string;
  description: string;
  neutralized: boolean;
  rank_transformed: boolean;
}

export interface FeatureStoreSummary {
  total_features: number;
  categories: string[];
  cache_format: string;
  storage_size_mb: number;
  features: FeatureMeta[];
}

export interface ModelVersionItem {
  model_id: string;
  algorithm: string;
  version: string;
  train_ic: number;
  val_ic: number;
  sharpe_net: number;
  created_at: string;
  status: string;
  reproducibility_hash: string;
}

export interface AlertRuleItem {
  id: number;
  metric: string;
  condition: string;
  threshold: number;
  is_active: boolean;
}

export interface AlertHistoryItem {
  id: number;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
  metric: string;
  message: string;
  triggered_at: string;
}

export interface ReportItem {
  report_id: string;
  title: string;
  file_name: string;
  file_type: string;
  created_at: string;
  size_kb: number;
}
