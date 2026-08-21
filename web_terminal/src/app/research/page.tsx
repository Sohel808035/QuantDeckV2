'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { Cpu, Award, Database, Layers } from 'lucide-react';

export default function ResearchPage() {
  const { data: features } = useQuery({
    queryKey: ['featureStore'],
    queryFn: api.getFeatureStoreSummary,
  });

  const { data: models } = useQuery({
    queryKey: ['modelRegistry'],
    queryFn: api.getModelRegistry,
  });

  const deciles = [
    { decile: 'Decile 1 (Top Alpha)', returnAnn: '+29.40%', sharpe: '2.15' },
    { decile: 'Decile 2', returnAnn: '+24.10%', sharpe: '1.84' },
    { decile: 'Decile 3', returnAnn: '+19.80%', sharpe: '1.55' },
    { decile: 'Decile 4', returnAnn: '+16.20%', sharpe: '1.32' },
    { decile: 'Decile 5', returnAnn: '+13.50%', sharpe: '1.10' },
    { decile: 'Decile 6', returnAnn: '+11.00%', sharpe: '0.92' },
    { decile: 'Decile 7', returnAnn: '+8.40%', sharpe: '0.71' },
    { decile: 'Decile 8', returnAnn: '+5.20%', sharpe: '0.45' },
    { decile: 'Decile 9', returnAnn: '+1.10%', sharpe: '0.12' },
    { decile: 'Decile 10 (Bottom)', returnAnn: '-13.40%', sharpe: '-0.85' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-blue-500" />
            <span>Alpha Signal Diagnostics & Feature Store</span>
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Information Coefficient (IC) Validation, Decile Spreads, and Feature Lineage
          </p>
        </div>
      </div>

      {/* IC Metrics Top Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">Mean Rank IC</div>
          <div className="text-xl font-bold font-mono text-white mt-1">+0.0482</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Req &gt;= +0.030 [PASS]</div>
        </div>

        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">IC t-statistic</div>
          <div className="text-xl font-bold font-mono text-white mt-1">+3.42</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Req &gt;= +2.000 [PASS]</div>
        </div>

        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">Decile 1 Spread</div>
          <div className="text-xl font-bold font-mono text-[#00E676] mt-1">+29.40% /yr</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Long Leg Return</div>
        </div>

        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">Decile 10 Short Leg</div>
          <div className="text-xl font-bold font-mono text-[#FF5252] mt-1">-13.40% /yr</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Monotonic Separation</div>
        </div>
      </div>

      {/* Model Governance Registry */}
      <div className="terminal-card p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
          <Award className="w-4 h-4 text-blue-400" />
          <span>Model Registry & Governance History</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#232A3B] text-[#94A3B8] uppercase font-mono">
                <th className="pb-3">Model ID</th>
                <th className="pb-3">Algorithm</th>
                <th className="pb-3">Version</th>
                <th className="pb-3">Train IC</th>
                <th className="pb-3">Val IC</th>
                <th className="pb-3">Net Sharpe</th>
                <th className="pb-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#232A3B]">
              {models?.map((m, idx) => (
                <tr key={idx} className="hover:bg-[#1E2536] transition">
                  <td className="py-3 font-semibold text-white font-mono">{m.model_id}</td>
                  <td className="py-3 text-white">{m.algorithm}</td>
                  <td className="py-3 font-mono text-[#94A3B8]">{m.version}</td>
                  <td className="py-3 font-mono text-white">+{m.train_ic}</td>
                  <td className="py-3 font-mono font-bold text-[#00E676]">+{m.val_ic}</td>
                  <td className="py-3 font-mono text-white">{m.sharpe_net}</td>
                  <td className="py-3">
                    <span className={m.status === 'PRODUCTION' ? 'badge-pass' : 'badge-warn'}>{m.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
