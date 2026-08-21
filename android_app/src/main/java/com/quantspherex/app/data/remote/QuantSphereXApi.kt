package com.quantspherex.app.data.remote

import com.quantspherex.app.data.model.PortfolioSummary
import com.quantspherex.app.data.model.ResearchAlphaItem
import com.quantspherex.app.data.model.BacktestRequest
import com.quantspherex.app.data.model.BacktestResponse
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.POST

interface QuantSphereXApi {

    // Authenticate / Check Status
    @GET("/api/v2/health/status")
    suspend fun getHealthStatus(@Header("X-API-Key") apiKey: String): Response<Map<String, String>>

    // Get Portfolio Summary via Backtest Run (Generates Equity Curve)
    @POST("/api/v2/backtest/run")
    suspend fun runBacktest(
        @Header("X-API-Key") apiKey: String,
        @Body request: BacktestRequest
    ): Response<BacktestResponse>

    // Get Research Alpha (mocking endpoint for AI analysis)
    @GET("/api/v2/ai/research/alpha")
    suspend fun getResearchAlpha(@Header("X-API-Key") apiKey: String): Response<List<ResearchAlphaItem>>
}
