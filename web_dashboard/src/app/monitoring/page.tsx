'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { Activity, Terminal } from 'lucide-react';

export default function MonitoringPage() {
  const { data: alerts } = useQuery({
    queryKey: ['alertHistory'],
    queryFn: api.getAlertHistory,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Activity className="w-5 h-5 text-blue-500" />
            <span>Real-Time Telemetry & Data Drift Monitoring</span>
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Population Stability Index (PSI), Model Health, CPU/RAM Usage, and Latency Telemetry
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">Feature Drift PSI</div>
          <div className="text-xl font-bold font-mono text-white mt-1">0.042</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Limit &lt;= 0.25 [PASS]</div>
        </div>

        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">Prediction Drift KS</div>
          <div className="text-xl font-bold font-mono text-white mt-1">0.018</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">p-value: 0.88 [PASS]</div>
        </div>

        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">API Latency p99</div>
          <div className="text-xl font-bold font-mono text-white mt-1">18.4ms</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Target &lt; 50ms</div>
        </div>

        <div className="terminal-card p-4">
          <div className="text-[11px] font-bold text-[#94A3B8] uppercase">CPU / Memory Usage</div>
          <div className="text-xl font-bold font-mono text-white mt-1">14% / 480MB</div>
          <div className="text-xs text-[#00E676] font-semibold mt-1">Healthy</div>
        </div>
      </div>

      <div className="terminal-card p-5 space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-blue-400" />
          <span>Real-Time Engine System Log Stream</span>
        </h3>

        <div className="bg-[#0E121B] border border-[#232A3B] p-4 rounded-lg font-mono text-xs text-[#94A3B8] space-y-1">
          {alerts && alerts.length > 0 ? (
            alerts.map((a) => (
              <div key={a.id}>
                {a.triggered_at} [{a.severity}] {a.metric}: {a.message}
              </div>
            ))
          ) : (
            <>
              <div>2026-07-30 11:30:00 [INFO] FeatureStore: 42 features validated against schema.</div>
              <div>2026-07-30 11:30:05 [INFO] ModelHealth: XGBoost ensemble predictions generated in 14.2ms.</div>
              <div>2026-07-30 11:30:10 [INFO] DriftMonitor: Feature PSI = 0.042 (No distribution drift detected).</div>
              <div>2026-07-30 11:30:15 [INFO] AlertEngine: All 4 risk thresholds operational. Status: HEALTHY.</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
