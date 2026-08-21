'use client';

import React from 'react';
import { Search, Bell, ShieldCheck, Activity, User, Terminal, Wifi, ChevronDown } from 'lucide-react';

interface TopNavProps {
  onOpenCommandPalette: () => void;
}

export default function TopNav({ onOpenCommandPalette }: TopNavProps) {
  return (
    <header className="bg-[#0E121B]/95 backdrop-blur-md border-b border-[#1E2638] sticky top-0 z-40 shadow-lg">
      {/* Live Market Telemetry Ticker Tape */}
      <div className="bg-[#07090E] border-b border-[#1A2130] px-4 py-1.5 text-[11px] font-mono flex items-center justify-between text-[#94A3B8]">
        <div className="flex items-center space-x-4 overflow-x-auto whitespace-nowrap scrollbar-none">
          <span className="flex items-center space-x-1.5 font-extrabold text-white tracking-wider">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-live"></span>
            <Terminal className="w-3.5 h-3.5 text-blue-500" />
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">QUANTDECK PRO</span>
          </span>
          <span className="text-[#1A2130]">|</span>
          <span>NIFTY 50: <strong className="text-[#10B981]">24,850.40 (+0.65%)</strong></span>
          <span>INDIA VIX: <strong className="text-[#F59E0B]">12.45 (-2.10%)</strong></span>
          <span>REGIME: <strong className="text-[#10B981] bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">BULL_TREND</strong></span>
          <span>NET SHARPE: <strong className="text-[#10B981]">1.84</strong></span>
          <span>ALPHA GENERATION: <strong className="text-blue-400">+4.82%</strong></span>
        </div>

        <div className="flex items-center space-x-3 text-[11px]">
          <span className="badge-pass flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            PROD_ONLINE
          </span>
          <span className="flex items-center text-[#94A3B8] gap-1">
            <Wifi className="w-3 h-3 text-emerald-400" />
            12ms
          </span>
        </div>
      </div>

      {/* Main Top Header Controls */}
      <div className="px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
              <span>QuantDeck Institutional Terminal</span>
            </h1>
            <span className="text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-md font-mono font-semibold">
              v2.1.0 CQRO
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={onOpenCommandPalette}
            className="flex items-center space-x-3 bg-[#131825] hover:bg-[#192032] border border-[#1E2638] hover:border-blue-500/40 px-3.5 py-1.5 rounded-lg text-xs text-[#94A3B8] transition-all shadow-sm group"
          >
            <Search className="w-3.5 h-3.5 text-blue-400 group-hover:scale-110 transition-transform" />
            <span>Search equities, models, signals...</span>
            <kbd className="bg-[#1E2638] text-[10px] px-1.5 py-0.5 rounded text-slate-300 font-mono border border-slate-700">⌘K</kbd>
          </button>

          <button className="relative p-2 bg-[#131825] hover:bg-[#192032] border border-[#1E2638] hover:border-slate-700 rounded-lg text-[#94A3B8] hover:text-white transition-all shadow-sm">
            <Bell className="w-4 h-4" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-emerald-500 rounded-full ring-2 ring-[#0E121B]"></span>
          </button>

          <div className="flex items-center space-x-2 bg-[#131825] border border-[#1E2638] hover:border-slate-700 px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer">
            <div className="w-5 h-5 rounded-full bg-blue-600/30 border border-blue-400/40 flex items-center justify-center text-blue-400 font-bold text-[10px]">
              A
            </div>
            <span className="font-semibold text-white">Admin Account</span>
            <span className="text-[10px] bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded font-mono font-bold">PRO</span>
          </div>
        </div>
      </div>
    </header>
  );
}
