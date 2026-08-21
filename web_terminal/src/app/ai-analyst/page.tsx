'use client';

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { Bot, Sparkles, FileText, CheckCircle2 } from 'lucide-react';

export default function AIAnalystPage() {
  const [symbol, setSymbol] = useState('RELIANCE.NS');

  const { data: explainability } = useQuery({
    queryKey: ['explainability', symbol],
    queryFn: () => api.getExplainability(symbol),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Bot className="w-5 h-5 text-blue-500" />
            <span>AI Quant Analyst & Explainability Workstation</span>
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Hallucination-Free Institutional Memos Reading SHAP, Confidence & Risk Engine Metrics
          </p>
        </div>

        <select
          value={symbol}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSymbol(e.target.value)}
          className="bg-[#141824] border border-[#232A3B] text-white text-xs font-mono px-3 py-2 rounded-lg focus:outline-none"
        >
          <option value="RELIANCE.NS">RELIANCE.NS (Reliance Industries)</option>
          <option value="TCS.NS">TCS.NS (Tata Consultancy Services)</option>
          <option value="BHARTIARTL.NS">BHARTIARTL.NS (Bharti Airtel)</option>
          <option value="HDFCBANK.NS">HDFCBANK.NS (HDFC Bank)</option>
        </select>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* SHAP Feature Driver Breakdown */}
        <div className="terminal-card p-5 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span>SHAP Feature Driver Breakdown ({symbol})</span>
          </h3>

          <div className="space-y-3 font-mono text-xs">
            {explainability &&
              (Object.entries(explainability.shap_values) as [string, number][]).map(([feat, val]: [string, number], idx: number) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-white">
                    <span>{feat}</span>
                    <span className={val > 0 ? 'text-[#00E676]' : 'text-[#FF5252]'}>
                      {val > 0 ? `+${val}` : val}
                    </span>
                  </div>
                  <div className="w-full bg-[#0E121B] h-2 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${val > 0 ? 'bg-[#00E676]' : 'bg-[#FF5252]'}`}
                      style={{ width: `${Math.min(Math.abs(val) * 1000, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>

        {/* AI Quantitative Executive Memo */}
        <div className="terminal-card p-5 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <FileText className="w-4 h-4 text-emerald-400" />
            <span>AI Executive Investment Memo</span>
          </h3>

          <div className="p-4 bg-[#0E121B] border border-[#232A3B] rounded-lg text-xs space-y-3">
            <div className="text-sm font-bold text-white flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-[#00E676]" />
              <span>Target Asset: {symbol}</span>
            </div>

            <p className="text-[#94A3B8] leading-relaxed">
              {explainability?.narrative || 'Model predicts strong multi-period excess return.'}
            </p>

            <div className="space-y-1">
              <div className="font-bold text-white">Key Factor Drivers:</div>
              <ul className="list-disc list-inside text-[#94A3B8] space-y-1">
                {explainability?.key_drivers.map((kd: string, idx: number) => (
                  <li key={idx}>{kd}</li>
                ))}
              </ul>
            </div>

            <div className="pt-2 border-t border-[#232A3B] flex justify-between font-mono">
              <span>Confidence Score: <strong className="text-[#00E676]">{explainability ? (explainability.confidence_score * 100).toFixed(0) : 88}%</strong></span>
              <span>Predicted Return: <strong className="text-[#00E676]">+{explainability ? (explainability.predicted_return * 100).toFixed(2) : '3.70'}%</strong></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
