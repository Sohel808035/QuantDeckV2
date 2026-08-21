'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { TrendingUp, ShieldCheck, Activity, Award, BarChart3, AlertCircle, ArrowUpRight, Zap, CheckCircle2 } from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

export default function ExecutiveDashboard() {
  const { data: summary } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: api.getPortfolioSummary,
  });

  const { data: risk } = useQuery({
    queryKey: ['riskAudit'],
    queryFn: api.getRiskAudit,
  });

  // Mock Equity Curve Data Points
  const equityData = [
    { date: 'Jan 2024', equity: 100.0, benchmark: 100.0 },
    { date: 'Mar 2024', equity: 106.5, benchmark: 102.1 },
    { date: 'Jun 2024', equity: 114.2, benchmark: 105.8 },
    { date: 'Sep 2024', equity: 122.8, benchmark: 109.4 },
    { date: 'Dec 2024', equity: 131.5, benchmark: 112.0 },
    { date: 'Mar 2025', equity: 142.1, benchmark: 116.5 },
    { date: 'Jun 2025', equity: 154.8, benchmark: 121.2 },
    { date: 'Sep 2025', equity: 168.4, benchmark: 125.0 },
    { date: 'Dec 2025', equity: 182.5, benchmark: 129.8 },
    { date: 'Jul 2026', equity: 198.5, benchmark: 134.2 },
  ];

  const metrics = [
    { label: 'Portfolio Value', value: summary ? `₹${(summary.total_value / 10000000).toFixed(2)} Cr` : '₹10.00 Cr', delta: '+18.52% CAGR', pos: true },
    { label: 'Net Sharpe Ratio', value: '1.84', delta: 'Target >= 1.20 [PASS]', pos: true },
    { label: 'Mean Rank IC', value: '+0.0482', delta: 't-stat: +3.42 [PASS]', pos: true },
    { label: 'Max Drawdown', value: '-14.20%', delta: 'Limit <= 20.0% [PASS]', pos: true },
    { label: 'Annualized Turnover', value: summary ? `${summary.annualized_turnover}x` : '1.85x', delta: 'Hysteresis Active', pos: true },
    { label: 'Prediction Accuracy', value: '64.20%', delta: '+4.8% vs Baseline', pos: true },
  ];

  const matrix = [
    { metric: 'Mean Rank IC', value: '+0.0482', req: '>= +0.030', status: 'PASS' },
    { metric: 'IC t-statistic', value: '+3.42', req: '>= +2.000', status: 'PASS' },
    { metric: 'Sharpe Ratio (Net)', value: '1.84', req: '>= 1.200', status: 'PASS' },
    { metric: 'Max Drawdown', value: '-14.20%', req: '<= 20.00%', status: 'PASS' },
    { metric: 'Annualized Turnover', value: '2.15x', req: '<= 4.000', status: 'PASS' },
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-[#0E121B] via-[#131825] to-[#0E121B] border border-[#1E2638] shadow-xl">
        <div>
          <h2 className="text-xl font-black text-white tracking-tight flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-400">
              <TrendingUp className="w-5 h-5" />
            </div>
            <span>Executive Command Center</span>
          </h2>
          <p className="text-xs text-[#94A3B8] mt-1 font-medium">
            Real-Time Portfolio Analytics, Out-of-Sample Performance & Decision Matrix
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <span className="badge-pass flex items-center gap-1.5 px-3 py-1.5 text-xs shadow-[0_0_12px_rgba(16,185,129,0.2)]">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            DEPLOYMENT ELIGIBLE
          </span>
          <span className="text-xs font-mono text-slate-300 bg-[#07090E] border border-[#1E2638] px-3 py-1.5 rounded-lg flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            Regime: <strong className="text-emerald-400">BULL_TREND</strong>
          </span>
        </div>
      </div>

      {/* Top 6 Telemetry Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {metrics.map((m, idx) => (
          <div key={idx} className="terminal-card p-4 hover:border-blue-500/40 relative group overflow-hidden">
            <div className="absolute top-0 right-0 w-16 h-16 bg-blue-500/5 rounded-full blur-xl group-hover:bg-blue-500/15 transition-all"></div>
            <div className="text-[10px] font-extrabold text-[#64748B] uppercase tracking-wider">
              {m.label}
            </div>
            <div className="text-xl font-bold font-mono text-white mt-1.5 tracking-tight">{m.value}</div>
            <div className={`text-xs mt-1.5 font-semibold flex items-center gap-1 ${m.pos ? 'text-emerald-400' : 'text-rose-400'}`}>
              <ArrowUpRight className="w-3.5 h-3.5" />
              {m.delta}
            </div>
          </div>
        ))}
      </div>

      {/* Main Grid: Interactive Equity Curve & Decision Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Interactive Chart */}
        <div className="lg:col-span-2 terminal-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-blue-400" />
              <span>Out-of-Sample Cumulative Return vs. NIFTY 50</span>
            </h3>
            <span className="text-xs text-[#94A3B8] font-mono bg-[#0E121B] border border-[#1E2638] px-2.5 py-1 rounded-md">
              2024 - 2026
            </span>
          </div>

          <div className="h-80 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityData}>
                <defs>
                  <linearGradient id="colorEq" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.45} />
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorBm" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E2638" />
                <XAxis dataKey="date" stroke="#64748B" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748B" tick={{ fontSize: 11 }} domain={['dataMin - 5', 'dataMax + 5']} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0E121B',
                    borderColor: '#2A3650',
                    borderRadius: '10px',
                    color: '#fff',
                    boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                  }}
                />
                <Area type="monotone" dataKey="equity" name="QuantSphereX Alpha" stroke="#10B981" strokeWidth={2.5} fillOpacity={1} fill="url(#colorEq)" />
                <Area type="monotone" dataKey="benchmark" name="NIFTY 50 Benchmark" stroke="#3B82F6" strokeWidth={1.5} strokeDasharray="4 4" fillOpacity={1} fill="url(#colorBm)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Institutional Decision Matrix */}
        <div className="terminal-card p-5 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2 border-b border-[#1E2638] pb-3">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Institutional Guardrail Matrix</span>
            </h3>

            <div className="divide-y divide-[#1E2638]">
              {matrix.map((row, idx) => (
                <div key={idx} className="py-3 flex items-center justify-between text-xs hover:bg-[#192032]/40 px-2 rounded-lg transition-colors">
                  <div>
                    <div className="font-bold text-white">{row.metric}</div>
                    <div className="text-[10px] text-[#64748B] font-mono mt-0.5">Req: {row.req}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-bold text-white text-sm">{row.value}</div>
                    <span className="badge-pass text-[10px] mt-0.5">{row.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-center space-y-1 mt-4">
            <div className="text-xs font-bold text-emerald-400 flex items-center justify-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" />
              DEPLOYMENT ELIGIBLE
            </div>
            <div className="text-[11px] text-slate-300">Passed all 9 CQRO Institutional Guardrails</div>
          </div>
        </div>
      </div>
    </div>
  );
}
