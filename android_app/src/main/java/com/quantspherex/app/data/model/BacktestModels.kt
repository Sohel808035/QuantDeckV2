package com.quantspherex.app.data.model

data class BacktestRequest(
    val initial_capital: Double = 10_000_000.0,
    val transaction_cost_pct: Double = 0.0015,
    val apply_vol_targeting: Boolean = true,
    val signal_lag_days: Int = 1,
    val rebalance_frequency: String = "monthly"
)

data class BacktestResponse(
    val cagr: Double,
    val ann_vol: Double,
    val sharpe_ratio: Double,
    val sortino_ratio: Double,
    val max_drawdown: Double,
    val calmar_ratio: Double,
    val final_equity: Double,
    val total_trades_count: Int,
    val equity_curve: List<Double>
)
