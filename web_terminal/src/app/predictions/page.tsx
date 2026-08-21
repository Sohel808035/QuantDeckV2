'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { TrendingUp, ShieldCheck, Layers } from 'lucide-react';

export default function PredictionsPage() {
  const { data: predictions } = useQuery({
    queryKey: ['latestPredictions'],
    queryFn: () => api.getLatestPredictions(20),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-blue-500" />
            <span>Cross-Sectional Multi-Period Alpha Forecasts</span>
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Ensemble Model Signal Scores, Confidence Uncertainty Error Bounds, and Top SHAP Feature Drivers
          </p>
        </div>
      </div>

      <div className="terminal-card p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
          <Layers className="w-4 h-4 text-blue-400" />
          <span>Latest Universe Prediction Rankings</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#232A3B] text-[#94A3B8] uppercase font-mono">
                <th className="pb-3">Symbol</th>
                <th className="pb-3">Forecasted Return</th>
                <th className="pb-3">Confidence Score</th>
                <th className="pb-3">Uncertainty Std</th>
                <th className="pb-3">Rank Decile</th>
                <th className="pb-3">Signal Direction</th>
                <th className="pb-3">Top SHAP Driver</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#232A3B]">
              {predictions?.map((pred, idx) => (
                <tr key={idx} className="hover:bg-[#1E2536] transition">
                  <td className="py-3 font-semibold text-white font-mono">{pred.symbol}</td>
                  <td className="py-3 font-mono font-bold text-[#00E676]">
                    +{(pred.predicted_return * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 font-mono text-white">
                    {(pred.confidence_score * 100).toFixed(0)}%
                  </td>
                  <td className="py-3 font-mono text-[#94A3B8]">{pred.uncertainty_std}</td>
                  <td className="py-3 font-mono text-white">Decile {pred.rank_decile}</td>
                  <td className="py-3">
                    <span className="badge-pass">{pred.signal_direction}</span>
                  </td>
                  <td className="py-3 font-mono text-blue-400">{pred.shap_top_driver}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
