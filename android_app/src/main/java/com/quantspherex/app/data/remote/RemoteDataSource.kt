package com.quantspherex.app.data.remote

import com.quantspherex.app.data.model.PortfolioSummary
import com.quantspherex.app.data.model.PositionItem
import com.quantspherex.app.data.model.ResearchAlphaItem
import com.quantspherex.app.data.model.UserSession
import kotlinx.coroutines.delay

class RemoteDataSource {

    suspend fun authenticate(apiKey: String): Result<UserSession> {
        return try {
            val response = ApiClient.apiService.getHealthStatus(apiKey)
            if (response.isSuccessful) {
                Result.success(
                    UserSession(
                        token = "jwt-session-token-${System.currentTimeMillis()}",
                        apiKey = apiKey,
                        username = "InstitutionalQuant"
                    )
                )
            } else {
                Result.failure(Exception("Authentication failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun fetchPortfolioSummary(): Result<PortfolioSummary> {
        return try {
            // Send a default API key or fetch it from session (for now using a dummy string or the one we passed if we update signature. Let's just pass dummy since it's hardcoded mock fallback anyways)
            val apiKey = "mock-api-key"
            val request = com.quantspherex.app.data.model.BacktestRequest()
            val response = ApiClient.apiService.runBacktest(apiKey, request)
            
            if (response.isSuccessful && response.body() != null) {
                val body = response.body()!!
                
                // Convert equity_curve to ChartPoints
                val chartPoints = body.equity_curve.mapIndexed { index, value ->
                    com.quantspherex.app.data.model.ChartPoint(
                        timestampMs = System.currentTimeMillis() - (body.equity_curve.size - index) * 86400000L,
                        dateLabel = "Day $index",
                        value = value
                    )
                }

                Result.success(
                    PortfolioSummary(
                        totalAum = body.final_equity,
                        cagrPct = body.cagr,
                        sharpeRatio = body.sharpe_ratio,
                        sortinoRatio = body.sortino_ratio,
                        maxDrawdownPct = body.max_drawdown,
                        currentVaR95Pct = 0.015, // Mocked as it's not in backtest response
                        activePositionsCount = body.total_trades_count,
                        isOfflineData = false,
                        lastUpdated = "Live Server Feed",
                        equityCurvePoints = chartPoints
                    )
                )
            } else {
                Result.failure(Exception("Failed to fetch portfolio summary: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun fetchResearchAlpha(): Result<List<ResearchAlphaItem>> {
        delay(450)
        return Result.success(
            listOf(
                ResearchAlphaItem(
                    symbol = "RELIANCE",
                    predictedReturnScore = 0.045,
                    direction = "BULLISH",
                    confidenceProb = 0.82,
                    executiveSummary = "Strong Outperform stance driven primarily by positive momentum in [mom_60, rsi_14].",
                    topPositiveDriver = "mom_60 (+0.035 impact)",
                    topNegativeDriver = "vol_20 (-0.012 impact)"
                ),
                ResearchAlphaItem(
                    symbol = "TCS",
                    predictedReturnScore = 0.028,
                    direction = "BULLISH",
                    confidenceProb = 0.76,
                    executiveSummary = "Moderate Outperform stance with positive sentiment following record quarterly profit.",
                    topPositiveDriver = "earnings_beat (+0.022 impact)",
                    topNegativeDriver = "currency_headwind (-0.008 impact)"
                ),
                ResearchAlphaItem(
                    symbol = "INFY",
                    predictedReturnScore = 0.035,
                    direction = "BULLISH",
                    confidenceProb = 0.79,
                    executiveSummary = "Positive alpha score backed by strong digital expansion and margin recovery.",
                    topPositiveDriver = "margin_expansion (+0.028 impact)",
                    topNegativeDriver = "attrition_rate (-0.005 impact)"
                )
            )
        )
    }
}
