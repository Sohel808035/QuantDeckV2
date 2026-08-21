'use client';

import React from 'react';
import { Settings, Sliders, Shield } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Settings className="w-5 h-5 text-blue-500" />
            <span>Institutional Engine Configuration</span>
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Rebalancing Frequency, Target Volatility, Hysteresis N, and Risk Controls
          </p>
        </div>
      </div>

      <div className="terminal-card p-6 max-w-2xl space-y-6">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2 border-b border-[#232A3B] pb-3">
          <Sliders className="w-4 h-4 text-blue-400" />
          <span>Execution Parameters & Model Thresholds</span>
        </h3>

        <form className="space-y-4 text-xs font-mono" onSubmit={(e) => e.preventDefault()}>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-[#94A3B8] mb-1">Target Annual Volatility</label>
              <input type="text" defaultValue="0.14 (14%)" className="w-full bg-[#0E121B] border border-[#232A3B] p-2 rounded text-white" />
            </div>

            <div>
              <label className="block text-[#94A3B8] mb-1">Top N Asset Selection</label>
              <input type="text" defaultValue="45 Assets" className="w-full bg-[#0E121B] border border-[#232A3B] p-2 rounded text-white" />
            </div>

            <div>
              <label className="block text-[#94A3B8] mb-1">Buffer N (Hysteresis Rank)</label>
              <input type="text" defaultValue="65 Assets" className="w-full bg-[#0E121B] border border-[#232A3B] p-2 rounded text-white" />
            </div>

            <div>
              <label className="block text-[#94A3B8] mb-1">Transaction Cost (bps)</label>
              <input type="text" defaultValue="15 bps (0.0015)" className="w-full bg-[#0E121B] border border-[#232A3B] p-2 rounded text-white" />
            </div>
          </div>

          <div className="pt-4 flex justify-end">
            <button type="submit" className="bg-blue-600 hover:bg-blue-500 text-white font-bold px-4 py-2 rounded-lg transition">
              Save Configuration
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
