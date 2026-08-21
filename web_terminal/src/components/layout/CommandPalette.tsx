'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Search, X, TrendingUp, ShieldAlert, Cpu, Globe, Briefcase, Bot, Command, ArrowRight } from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        onClose();
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const quickActions = [
    { label: 'Executive Dashboard', href: '/', category: 'Navigation', icon: TrendingUp },
    { label: 'Markets Overview & Sectors', href: '/markets', category: 'Markets', icon: Globe },
    { label: 'Active Portfolio Holdings', href: '/portfolio', category: 'Portfolio', icon: Briefcase },
    { label: 'Risk Engine & Stress Tests', href: '/risk', category: 'Risk', icon: ShieldAlert },
    { label: 'Alpha Signal Diagnostics', href: '/research', category: 'ML Research', icon: Cpu },
    { label: 'AI Quant Workstation', href: '/ai-analyst', category: 'AI Workstation', icon: Bot },
    { label: 'RELIANCE.NS - Reliance Industries', href: '/ai-analyst', category: 'Equities', icon: TrendingUp },
    { label: 'TCS.NS - Tata Consultancy Services', href: '/ai-analyst', category: 'Equities', icon: TrendingUp },
    { label: 'HDFCBANK.NS - HDFC Bank Ltd.', href: '/ai-analyst', category: 'Equities', icon: TrendingUp },
  ];

  const filtered = quickActions.filter((a) =>
    a.label.toLowerCase().includes(query.toLowerCase()) || a.category.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (href: string) => {
    router.push(href);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-start justify-center pt-20 animate-in fade-in duration-200">
      <div className="bg-[#131825] border border-[#2A3650] w-full max-w-xl rounded-2xl shadow-2xl overflow-hidden ring-1 ring-white/10">
        {/* Search Header */}
        <div className="p-4 border-b border-[#1E2638] flex items-center space-x-3 bg-[#0E121B]">
          <Search className="w-5 h-5 text-blue-400 shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a command, stock symbol (e.g. RELIANCE), or page..."
            className="w-full bg-transparent text-white placeholder-[#64748B] text-sm focus:outline-none font-medium"
            autoFocus
          />
          <button
            onClick={onClose}
            className="p-1 text-[#94A3B8] hover:text-white hover:bg-[#1E2638] rounded-lg transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-80 overflow-y-auto p-2 space-y-1">
          {filtered.length > 0 ? (
            filtered.map((item, idx) => {
              const Icon = item.icon;
              return (
                <button
                  key={idx}
                  onClick={() => handleSelect(item.href)}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs hover:bg-[#192032] hover:border hover:border-[#2A3650] transition group text-left text-white"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 group-hover:bg-blue-500/20 transition">
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="font-medium group-hover:text-blue-300 transition">{item.label}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] bg-[#0E121B] text-[#94A3B8] border border-[#1E2638] px-2 py-0.5 rounded-md font-mono">
                      {item.category}
                    </span>
                    <ArrowRight className="w-3.5 h-3.5 text-[#64748B] opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>
              );
            })
          ) : (
            <div className="p-8 text-center text-xs text-[#94A3B8]">
              No matching assets or actions found.
            </div>
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="p-3 border-t border-[#1E2638] bg-[#0E121B] flex items-center justify-between text-[11px] text-[#64748B]">
          <div className="flex items-center space-x-3">
            <span><kbd className="bg-[#192032] text-slate-300 px-1.5 py-0.5 rounded text-[10px] font-mono border border-[#2A3650]">↑↓</kbd> Navigate</span>
            <span><kbd className="bg-[#192032] text-slate-300 px-1.5 py-0.5 rounded text-[10px] font-mono border border-[#2A3650]">↵</kbd> Select</span>
            <span><kbd className="bg-[#192032] text-slate-300 px-1.5 py-0.5 rounded text-[10px] font-mono border border-[#2A3650]">ESC</kbd> Close</span>
          </div>
          <span className="font-mono text-blue-400">QuantSphereX Command</span>
        </div>
      </div>
    </div>
  );
}
