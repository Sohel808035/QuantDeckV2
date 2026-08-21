'use client';

import React, { useState } from 'react';
import './globals.css';
import Providers from '@/components/Providers';
import TopNav from '@/components/layout/TopNav';
import Sidebar from '@/components/layout/Sidebar';
import CommandPalette from '@/components/layout/CommandPalette';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  return (
    <html lang="en">
      <head>
        <title>QuantSphereX Institutional Research Terminal</title>
        <meta name="description" content="Institutional Quantitative Research & Trading Platform" />
      </head>
      <body className="bg-[#0B0E14] text-[#F8FAFC] antialiased">
        <Providers>
          <div className="min-h-screen flex flex-col">
            <TopNav onOpenCommandPalette={() => setCommandPaletteOpen(true)} />

            <div className="flex-1 flex overflow-hidden">
              <Sidebar
                collapsed={sidebarCollapsed}
                onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
              />

              <main className="flex-1 overflow-y-auto p-6 bg-[#0B0E14]">
                {children}
              </main>
            </div>
          </div>

          <CommandPalette
            isOpen={commandPaletteOpen}
            onClose={() => setCommandPaletteOpen(false)}
          />
        </Providers>
      </body>
    </html>
  );
}
