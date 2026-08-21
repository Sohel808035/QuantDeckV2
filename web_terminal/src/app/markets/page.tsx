'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { Globe, Layers, BarChart2, TrendingUp, TrendingDown, ArrowUpRight } from 'lucide-react';
import { StockItem } from '@/services/types';

export default function MarketsOverview() {
  const { data: stocks } = useQuery({
    queryKey: ['stockSearch'],
    queryFn: () => api.searchStocks(),
  });

  const sectors = [
    { sector: 'Information Technology', weight: '16.4%', return1m: '+4.20%', return3m: '+12.5%', signal: 'OVERWEIGHT', status: 'pass' },
    { sector: 'Financial Services', weight: '32.1%', return1m: '+2.80%', return3m: '+8.1%', signal: 'NEUTRAL', status: 'pass' },
    { sector: 'Automobile & Transport', weight: '8.5%', return1m: '+5.10%', return3m: '+14.2%', signal: 'OVERWEIGHT', status: 'pass' },
    { sector: 'Oil & Gas / Energy', weight: '11.2%', return1m: '-1.40%', return3m: '-2.5%', signal: 'UNDERWEIGHT', status: 'warn' },
    { sector: 'Consumer Fast Goods', weight: '12.0%', return1m: '+0.90%', return3m: '+3.4%', signal: 'NEUTRAL', status: 'pass' },
    { sector: 'Healthcare & Pharma', weight: '6.8%', return1m: '+3.40%', return3m: '+9.8%', signal: 'OVERWEIGHT', status: 'pass' },
  ];

  const macro = [
    { metric: 'RBI Repo Rate', value: '6.50%', status: 'PAUSED' },
    { metric: 'India 10Y Benchmark Yield', value: '6.98%', status: 'STABLE' },
    { metric: 'US Fed Funds Rate', value: '5.25%', status: 'PAUSED' },
    { metric: 'Brent Crude Oil ($/bbl)', value: '$78.40', status: '-1.2%' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-[#0E121B] via-[#131825] to-[#0E121B] border border-[#1E2638] shadow-xl">
        <div>
          <h2 className="text-xl font-black text-white tracking-tight flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-400">
              <Globe className="w-5 h-5" />
            </div>
            <span>Global & Indian Markets Overview</span>
          </h2>
          <p className="text-xs text-[#94A3B8] mt-1 font-medium">
            Index Breadth, Macroeconomic Indicators, and Sector Relative Performance Heatmaps
          </p>
        </div>
      </div>

      {/* Macro Indicators */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {macro.map((m, idx) => (
          <div key={idx} className="terminal-card p-4 hover:border-blue-500/40 relative overflow-hidden group">
            <div className="text-[10px] font-extrabold text-[#64748B] uppercase tracking-wider">{m.metric}</div>
            <div className="text-xl font-bold font-mono text-white mt-1.5">{m.value}</div>
            <div className="text-xs text-blue-400 font-semibold mt-1.5 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
              {m.status}
            </div>
          </div>
        ))}
      </div>

      {/* Sector Relative Strength Matrix */}
      <div className="terminal-card p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2 border-b border-[#1E2638] pb-3">
          <Layers className="w-4 h-4 text-blue-400" />
          <span>Sector Breakdown & Relative Strength Signals</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="quant-table">
            <thead>
              <tr>
                <th>Sector Name</th>
                <th>Universe Weight</th>
                <th>1-Month Return</th>
                <th>3-Month Return</th>
                <th>Quant Signal</th>
              </tr>
            </thead>
            <tbody>
              {sectors.map((s, idx) => (
                <tr key={idx}>
                  <td className="font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    {s.sector}
                  </td>
                  <td className="font-mono text-slate-300">{s.weight}</td>
                  <td className={`font-mono font-bold ${s.return1m.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {s.return1m}
                  </td>
                  <td className="font-mono text-slate-300">{s.return3m}</td>
                  <td>
                    <span className={s.signal === 'OVERWEIGHT' ? 'badge-pass' : s.signal === 'UNDERWEIGHT' ? 'badge-fail' : 'badge-warn'}>
                      {s.signal}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Stock Universe Coverage */}
      {stocks && stocks.length > 0 && (
        <div className="terminal-card p-5 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2 border-b border-[#1E2638] pb-3">
            <BarChart2 className="w-4 h-4 text-emerald-400" />
            <span>Monitored Stock Universe ({stocks.length} Assets)</span>
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 font-mono text-xs">
            {stocks.map((stk: StockItem) => (
              <div key={stk.symbol} className="bg-[#0E121B] border border-[#1E2638] hover:border-blue-500/40 p-3 rounded-xl transition-all">
                <div className="font-bold text-white flex items-center justify-between">
                  <span>{stk.symbol}</span>
                  <ArrowUpRight className="w-3 h-3 text-blue-400" />
                </div>
                <div className="text-[10px] text-[#94A3B8] truncate mt-0.5">{stk.name}</div>
                <div className="text-[10px] text-blue-400 mt-1 font-semibold">{stk.sector}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
