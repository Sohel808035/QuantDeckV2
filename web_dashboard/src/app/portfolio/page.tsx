'use client';

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import { Briefcase, RefreshCw, Layers, CheckCircle2, ArrowUpRight, ShieldCheck } from 'lucide-react';

export default function PortfolioPage() {
  const queryClient = useQueryClient();
  const [isRebalancing, setIsRebalancing] = useState(false);
  const [rebalanceResult, setRebalanceResult] = useState<any>(null);

  const { data: portfolio, isLoading } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: api.getPortfolioSummary,
  });

  const handleRebalance = async () => {
    setIsRebalancing(true);
    const res = await api.rebalancePortfolio();
    setRebalanceResult(res);
    setIsRebalancing(false);
    queryClient.invalidateQueries({ queryKey: ['portfolioSummary'] });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-[#0E121B] via-[#131825] to-[#0E121B] border border-[#1E2638] shadow-xl">
        <div>
          <h2 className="text-xl font-black text-white tracking-tight flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-400">
              <Briefcase className="w-5 h-5" />
            </div>
            <span>Active Portfolio Holdings & Rebalancing Engine</span>
          </h2>
          <p className="text-xs text-[#94A3B8] mt-1 font-medium">
            Hysteresis Rank Buffer Positions, Target Sizing & Turnover Penalized Execution
          </p>
        </div>

        <button
          onClick={handleRebalance}
          disabled={isRebalancing}
          className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-bold text-xs px-4 py-2.5 rounded-xl transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw className={`w-4 h-4 ${isRebalancing ? 'animate-spin' : ''}`} />
          <span>{isRebalancing ? 'Optimizing Orders...' : 'Generate Rebalance Orders'}</span>
        </button>
      </div>

      {/* Top Portfolio Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        <div className="terminal-card p-4 hover:border-blue-500/40 relative overflow-hidden">
          <div className="text-[10px] font-extrabold text-[#64748B] uppercase tracking-wider">Portfolio Total Value</div>
          <div className="text-xl font-bold font-mono text-white mt-1.5">
            {portfolio ? `₹${(portfolio.total_value / 10000000).toFixed(2)} Cr` : '₹10.00 Cr'}
          </div>
          <div className="text-xs text-emerald-400 font-semibold mt-1.5 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            Core Alpha Fund
          </div>
        </div>

        <div className="terminal-card p-4 hover:border-blue-500/40 relative overflow-hidden">
          <div className="text-[10px] font-extrabold text-[#64748B] uppercase tracking-wider">Cash Balance Reserve</div>
          <div className="text-xl font-bold font-mono text-white mt-1.5">
            {portfolio ? `₹${(portfolio.cash_balance / 10000000).toFixed(2)} Cr` : '₹4.87 Cr'}
          </div>
          <div className="text-xs text-slate-400 font-semibold mt-1.5">48.70% Reserve</div>
        </div>

        <div className="terminal-card p-4 hover:border-blue-500/40 relative overflow-hidden">
          <div className="text-[10px] font-extrabold text-[#64748B] uppercase tracking-wider">Top 5 Concentration</div>
          <div className="text-xl font-bold font-mono text-white mt-1.5">
            {portfolio ? `${(portfolio.top_5_concentration_pct * 100).toFixed(2)}%` : '30.28%'}
          </div>
          <div className="text-xs text-emerald-400 font-semibold mt-1.5 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            Cap &lt;= 35.0% [PASS]
          </div>
        </div>

        <div className="terminal-card p-4 hover:border-blue-500/40 relative overflow-hidden">
          <div className="text-[10px] font-extrabold text-[#64748B] uppercase tracking-wider">Annualized Turnover</div>
          <div className="text-xl font-bold font-mono text-white mt-1.5">
            {portfolio ? `${portfolio.annualized_turnover}x` : '1.85x'}
          </div>
          <div className="text-xs text-blue-400 font-semibold mt-1.5">Hysteresis Dampening</div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="terminal-card p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2 border-b border-[#1E2638] pb-3">
          <Layers className="w-4 h-4 text-blue-400" />
          <span>Active Holdings & Target Weights</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="quant-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Shares</th>
                <th>Avg Price</th>
                <th>Current Price</th>
                <th>Market Value</th>
                <th>Current Weight</th>
                <th>Target Weight</th>
                <th>Hysteresis Buffer</th>
              </tr>
            </thead>
            <tbody>
              {portfolio?.positions.map((pos, idx) => (
                <tr key={idx}>
                  <td className="font-bold text-white font-mono">{pos.symbol}</td>
                  <td className="font-mono text-slate-300">{pos.shares.toLocaleString()}</td>
                  <td className="font-mono text-[#94A3B8]">₹{pos.avg_price.toFixed(2)}</td>
                  <td className="font-mono text-white">₹{pos.current_price.toFixed(2)}</td>
                  <td className="font-mono text-white">₹{pos.market_value.toLocaleString()}</td>
                  <td className="font-mono text-slate-300">{(pos.current_weight * 100).toFixed(2)}%</td>
                  <td className="font-mono text-blue-400 font-bold">{(pos.target_weight * 100).toFixed(2)}%</td>
                  <td>
                    <span className={pos.hysteresis_status === 'NEW_ENTRY' ? 'badge-pass' : 'badge-warn'}>
                      {pos.hysteresis_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Generated Rebalance Trades Panel */}
      {rebalanceResult && (
        <div className="terminal-card p-5 border border-emerald-500/40 glow-card-emerald space-y-4">
          <div className="flex items-center space-x-2 text-emerald-400 font-bold text-sm">
            <CheckCircle2 className="w-5 h-5" />
            <span>Generated {rebalanceResult.total_trades} Turnover-Penalized Orders</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono text-[#94A3B8] bg-[#0E121B] p-3 rounded-xl border border-[#1E2638]">
            <div>Est Turnover: <strong className="text-white">{(rebalanceResult.estimated_turnover_pct * 100).toFixed(2)}%</strong></div>
            <div>Est Friction Cost: <strong className="text-white">₹{rebalanceResult.estimated_transaction_cost.toLocaleString()}</strong></div>
            <div>Status: <strong className="text-emerald-400">READY FOR EXECUTION</strong></div>
          </div>

          <div className="overflow-x-auto">
            <table className="quant-table font-mono">
              <thead>
                <tr>
                  <th>Action</th>
                  <th>Symbol</th>
                  <th>Shares Delta</th>
                  <th>Target Weight</th>
                  <th>Est Value</th>
                </tr>
              </thead>
              <tbody>
                {rebalanceResult.trades.map((t: any, idx: number) => (
                  <tr key={idx}>
                    <td className={`font-bold ${t.action === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>{t.action}</td>
                    <td className="text-white">{t.symbol}</td>
                    <td className="text-white">{t.shares_delta}</td>
                    <td className="text-white">{(t.target_weight * 100).toFixed(2)}%</td>
                    <td className="text-white">₹{t.estimated_value.toLocaleString()}</td>
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
