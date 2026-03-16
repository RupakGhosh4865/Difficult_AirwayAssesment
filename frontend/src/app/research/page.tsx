"use client";

import React, { useState } from 'react';
import { FileText, Download, ExternalLink, Loader2, AlertCircle } from 'lucide-react';

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export default function ResearchPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  return (
    <div className="h-full flex flex-col space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold mb-2">Research Paper</h2>
          <p className="text-slate-500 dark:text-slate-400">Technical details and methodology of AirwayAI.</p>
        </div>
        <div className="flex gap-3">
          <a 
            href={`${API_BASE_URL}/research-paper`}
            download
            className="px-5 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl font-bold text-sm flex items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
          >
            <Download size={18} />
            Download PDF
          </a>
           <a 
            href={`${API_BASE_URL}/research-paper`}
            target="_blank"
            className="px-5 py-2.5 bg-primary text-white rounded-xl font-bold text-sm flex items-center gap-2 hover:bg-primary-hover transition-colors shadow-lg shadow-primary/20"
          >
            <ExternalLink size={18} />
            Open in New Tab
          </a>
        </div>
      </div>

      <div className="flex-1 bg-white dark:bg-slate-800 rounded-3xl border border-slate-200 dark:border-slate-700 overflow-hidden relative shadow-sm">
        {loading && (
           <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-white dark:bg-slate-800 z-10">
            <Loader2 className="animate-spin text-primary" size={32} />
            <p className="text-slate-400 text-sm font-medium">Loading PDF viewer...</p>
          </div>
        )}
        
        {error ? (
          <div className="flex flex-col items-center justify-center h-full p-10 text-center">
            <div className="w-16 h-16 bg-red-50 text-red-500 rounded-2xl flex items-center justify-center mb-4">
              <AlertCircle size={32} />
            </div>
            <h3 className="text-xl font-bold mb-2">Failed to load PDF</h3>
            <p className="text-slate-500 max-w-sm">
              Please ensure you have placed your research paper PDF file inside the <code>utils</code> folder in the project root.
            </p>
          </div>
        ) : (
          <iframe 
            src={`${API_BASE_URL}/research-paper#toolbar=0`} 
            className="w-full h-[700px] border-none"
            onLoad={() => setLoading(false)}
            onError={() => {
              setLoading(false);
              setError(true);
            }}
          />
        )}
      </div>

      <div className="p-6 bg-slate-50 dark:bg-slate-800/50 rounded-2xl border border-slate-100 dark:border-slate-700">
        <h4 className="font-bold flex items-center gap-2 mb-2 text-sm">
           <FileText size={16} className="text-primary" />
           About the Methodology
        </h4>
        <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
          The paper details our approach using a ResNet18 convolutional neural network. Subjects were screened using standardized airway 
          positioning and images were processed through a custom augmentation pipeline to improve model robustness across diverse patient populations.
        </p>
      </div>
    </div>
  );
}
