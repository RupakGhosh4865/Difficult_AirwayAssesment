import React from 'react';
import Link from 'next/link';
import { Home, BarChart2, FileText, Upload, Activity } from 'lucide-react';

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-slate-50 dark:bg-slate-900">
      {/* Sidebar */}
      <aside className="w-full md:w-64 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 p-6 flex flex-col">
        <div className="flex items-center gap-3 mb-10">
          <div className="bg-primary text-white p-2 rounded-lg">
            <Activity size={24} />
          </div>
          <h1 className="font-bold text-lg tracking-tight leading-tight">
            AirwayAI <br />
            <span className="text-xs font-normal text-slate-500">Intubation Predictor</span>
          </h1>
        </div>

        <nav className="flex-1 space-y-2">
          <NavItem href="/" icon={<Home size={20} />} label="Home" />
          <NavItem href="/predict" icon={<Upload size={20} />} label="Predict Risk" />
          <NavItem href="/analytics" icon={<BarChart2 size={20} />} label="Analytics" />
          <NavItem href="/research" icon={<FileText size={20} />} label="Research" />
        </nav>

        <div className="mt-auto pt-6 border-t border-slate-100 dark:border-slate-700">
          <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-xl text-xs text-slate-500">
            <p className="font-medium text-slate-700 dark:text-slate-300 mb-1">Status</p>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              API Connected
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto p-4 md:p-10">
        <div className="max-w-6xl mx-auto h-full">
          {children}
        </div>
      </main>
    </div>
  );
}

function NavItem({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <Link 
      href={href} 
      className="flex items-center gap-3 px-4 py-3 rounded-xl text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-primary transition-all duration-200 group"
    >
      <span className="group-hover:scale-110 transition-transform">{icon}</span>
      <span className="font-medium">{label}</span>
    </Link>
  );
}
