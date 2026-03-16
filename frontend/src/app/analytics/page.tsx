"use client";

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  Cell, PieChart, Pie, Legend 
} from 'recharts';
import { TrendingUp, Target, Activity, Zap, Loader2, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');

type MetricsData = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
};

export default function AnalyticsPage() {
  const [data, setData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/metrics`);
        setData(response.data);
      } catch (err) {
        setError("Failed to load metrics. Ensure the backend is running and training files exist.");
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) return (
    <div className="h-[60vh] flex flex-col items-center justify-center gap-4">
      <Loader2 className="animate-spin text-primary" size={48} />
      <p className="text-slate-500 font-medium">Loading model performance data...</p>
    </div>
  );

  if (error) return (
    <div className="h-[60vh] flex flex-col items-center justify-center gap-4 text-center max-w-md mx-auto">
      <div className="w-16 h-16 bg-red-50 text-red-500 rounded-2xl flex items-center justify-center mb-2">
        <AlertCircle size={32} />
      </div>
      <h3 className="text-xl font-bold">Metrics Unavailable</h3>
      <p className="text-slate-500">{error}</p>
    </div>
  );

  const stats = [
    { label: 'Accuracy', value: data?.accuracy, icon: <Target />, color: 'text-blue-500', bg: 'bg-blue-50' },
    { label: 'Precision', value: data?.precision, icon: <TrendingUp />, color: 'text-green-500', bg: 'bg-green-50' },
    { label: 'Recall', value: data?.recall, icon: <Activity />, color: 'text-purple-500', bg: 'bg-purple-50' },
    { label: 'F1 Score', value: data?.f1, icon: <Zap />, color: 'text-yellow-500', bg: 'bg-yellow-50' },
  ];

  const mainMetrics = [
    { name: 'Accuracy', value: (data?.accuracy || 0) * 100 },
    { name: 'Precision', value: (data?.precision || 0) * 100 },
    { name: 'Recall', value: (data?.recall || 0) * 100 },
    { name: 'F1 Score', value: (data?.f1 || 0) * 100 },
  ];

  return (
    <div className="space-y-10">
      <div>
        <h2 className="text-3xl font-bold mb-2">Model Performance</h2>
        <p className="text-slate-500 dark:text-slate-400">Analysis of the ResNet18 model trained on clinical airway data.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, i) => (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            key={stat.label} 
            className="bg-white dark:bg-slate-800 p-6 rounded-3xl border border-slate-100 dark:border-slate-700 shadow-sm"
          >
            <div className={`w-12 h-12 ${stat.bg} ${stat.color} rounded-2xl flex items-center justify-center mb-4`}>
              {stat.icon}
            </div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">{stat.label}</p>
            <p className="text-2xl font-black">{((stat.value || 0) * 100).toFixed(1)}%</p>
          </motion.div>
        ))}
      </div>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Main Chart */}
        <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl border border-slate-100 dark:border-slate-700 shadow-sm">
          <h3 className="font-bold text-lg mb-8">Performance Overview</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mainMetrics} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fontWeight: 500 }} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12 }} domain={[0, 100]} />
                <Tooltip 
                  cursor={{ fill: 'transparent' }}
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={40}>
                   {mainMetrics.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#3b82f6', '#22c55e', '#a855f7', '#eab308'][index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Visualization */}
        <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl border border-slate-100 dark:border-slate-700 shadow-sm">
          <h3 className="font-bold text-lg mb-8">Confusion Matrix</h3>
          {data?.confusion_matrix ? (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <MatrixItem label="True Negative" sub="Easy -> Easy" value={data.confusion_matrix[1][1]} color="bg-green-100 text-green-700" />
                <MatrixItem label="False Positive" sub="Easy -> Diff" value={data.confusion_matrix[1][0]} color="bg-red-50 text-red-600" />
                <MatrixItem label="False Negative" sub="Diff -> Easy" value={data.confusion_matrix[0][1]} color="bg-orange-50 text-orange-600" />
                <MatrixItem label="True Positive" sub="Diff -> Diff" value={data.confusion_matrix[0][0]} color="bg-green-100 text-green-700" />
              </div>
              <p className="text-xs text-slate-400 text-center italic">
                * Note: Indices might vary based on training class order (Easy/Difficult).
              </p>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[200px] text-slate-400 italic">
               Confusion matrix data not available.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function MatrixItem({ label, sub, value, color }: any) {
  return (
    <div className={cn("p-4 rounded-2xl flex flex-col items-center justify-center text-center", color)}>
      <span className="text-xs font-bold uppercase tracking-tight opacity-70 mb-1">{label}</span>
      <span className="text-2xl font-black">{value}</span>
      <span className="text-[10px] uppercase mt-1 opacity-60">{sub}</span>
    </div>
  );
}
