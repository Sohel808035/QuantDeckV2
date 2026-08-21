'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Globe,
  Briefcase,
  ShieldAlert,
  Cpu,
  TrendingUp,
  Bot,
  Activity,
  FileSpreadsheet,
  Settings,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function Sidebar({ collapsed, onToggleCollapse }: SidebarProps) {
  const pathname = usePathname();

  const navGroups = [
    {
      label: 'OVERVIEW & MARKETS',
      items: [
        { href: '/', label: 'Executive Dashboard', icon: LayoutDashboard },
        { href: '/markets', label: 'Markets Overview', icon: Globe },
      ],
    },
    {
      label: 'PORTFOLIO & EXECUTION',
      items: [
        { href: '/portfolio', label: 'Active Portfolio', icon: Briefcase },
        { href: '/risk', label: 'Risk & Stress Tests', icon: ShieldAlert },
      ],
    },
    {
      label: 'ALPHA RESEARCH & ML',
      items: [
        { href: '/research', label: 'Alpha Signal Research', icon: Cpu },
        { href: '/predictions', label: 'Multi-Period Forecasts', icon: TrendingUp },
        { href: '/ai-analyst', label: 'AI Quant Workstation', icon: Bot },
      ],
    },
    {
      label: 'TELEMETRY & OPERATIONS',
      items: [
        { href: '/monitoring', label: 'Telemetry & Drift', icon: Activity },
        { href: '/reports', label: 'Teardown Reports', icon: FileSpreadsheet },
        { href: '/settings', label: 'Engine Settings', icon: Settings },
        { href: '/admin', label: 'System Admin (RBAC)', icon: ShieldCheck },
      ],
    },
  ];

  return (
    <aside
      className={`bg-[#0E121B] border-r border-[#1E2638] flex flex-col justify-between transition-all duration-300 z-30 select-none ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      <div className="py-4">
        {/* Collapse Header */}
        <div className="px-4 mb-4 flex justify-between items-center">
          {!collapsed && (
            <span className="text-[10px] font-extrabold tracking-widest text-[#64748B] uppercase">
              Terminal Navigation
            </span>
          )}
          <button
            onClick={onToggleCollapse}
            className="p-1.5 hover:bg-[#192032] text-[#94A3B8] hover:text-white rounded-lg border border-transparent hover:border-[#1E2638] transition-all"
            title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Nav Items Grouped */}
        <div className="space-y-5 px-2.5">
          {navGroups.map((group, idx) => (
            <div key={idx}>
              {!collapsed && (
                <div className="px-3 mb-1.5 text-[9px] font-black text-[#475569] tracking-widest uppercase">
                  {group.label}
                </div>
              )}
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href;

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`relative flex items-center ${collapsed ? 'justify-center' : 'space-x-3'} px-3 py-2 rounded-xl text-xs font-semibold transition-all ${
                        isActive
                          ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-[0_0_12px_rgba(59,130,246,0.15)]'
                          : 'text-[#94A3B8] hover:bg-[#131825] hover:text-white hover:border hover:border-[#1E2638]'
                      }`}
                      title={collapsed ? item.label : undefined}
                    >
                      {isActive && (
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-blue-500 rounded-r-full shadow-[0_0_8px_#3B82F6]"></div>
                      )}
                      <Icon className={`w-4 h-4 transition-transform group-hover:scale-110 ${isActive ? 'text-blue-400' : 'text-[#64748B]'}`} />
                      {!collapsed && <span className="truncate">{item.label}</span>}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer Branding */}
      {!collapsed && (
        <div className="p-4 border-t border-[#1E2638] bg-[#0A0D14]/50 text-[11px] text-[#64748B]">
          <div className="flex items-center justify-between">
            <span className="font-bold text-white">QuantSphereX</span>
            <span className="text-[9px] bg-blue-500/10 text-blue-400 px-1.5 py-0.5 rounded font-mono border border-blue-500/20">PRO</span>
          </div>
          <p className="text-[10px] text-[#64748B] mt-0.5">Institutional Engine v2.1</p>
        </div>
      )}
    </aside>
  );
}
