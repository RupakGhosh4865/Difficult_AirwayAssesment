"use client";

import React, { useState } from 'react';
import { Upload, Camera, AlertCircle, CheckCircle2, ChevronRight, Loader2, Zap } from 'lucide-react';
import axios from 'axios';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');

type PredictionResult = {
  prediction: string;
  confidence: number;
  probabilities: {
    Easy: number;
    Difficult: number;
  };
};

export default function PredictPage() {
  const [images, setImages] = useState<{ [key: string]: File | null }>({
    neutral: null,
    tongue: null,
    headup: null,
  });
  const [previews, setPreviews] = useState<{ [key: string]: string | null }>({
    neutral: null,
    tongue: null,
    headup: null,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, key: string) => {
    const file = e.target.files?.[0];
    if (file) {
      setImages(prev => ({ ...prev, [key]: file }));
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviews(prev => ({ ...prev, [key]: reader.result as string }));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!images.neutral || !images.tongue || !images.headup) {
      setError("Please upload all three required photos.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('neutral', images.neutral);
    formData.append('tongue', images.tongue);
    formData.append('headup', images.headup);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData);
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Prediction failed. Is the backend server running?");
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClick = async (type: string, url: string, fileName: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const file = new File([blob], fileName, { type: blob.type });
      
      setImages(prev => ({ ...prev, [type]: file }));
      setPreviews(prev => ({ ...prev, [type]: url }));
    } catch (err) {
      console.error("Failed to load sample image", err);
    }
  };

  const isUploadComplete = images.neutral && images.tongue && images.headup;

  const samples = [
    { type: 'neutral', url: '/test-images/neutral_1.jpg', label: 'Sample 1', cat: 'Neutral' },
    { type: 'neutral', url: '/test-images/neutral_2.jpg', label: 'Sample 2', cat: 'Neutral' },
    { type: 'tongue', url: '/test-images/tongue_1.jpg', label: 'Sample 1', cat: 'Tongue' },
    { type: 'tongue', url: '/test-images/tongue_2.jpg', label: 'Sample 2', cat: 'Tongue' },
    { type: 'headup', url: '/test-images/headup_1.jpg', label: 'Sample 1', cat: 'Head Up' },
    { type: 'headup', url: '/test-images/headup_2.jpg', label: 'Sample 2', cat: 'Head Up' },
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-10">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h2 className="text-3xl font-bold mb-2">Predict Intubation Risk</h2>
          <p className="text-slate-500 dark:text-slate-400">Upload clinical photos for AI-assisted assessment.</p>
        </div>
        <div className="flex gap-2 text-xs font-medium text-slate-500 bg-slate-100 dark:bg-slate-800 p-2 rounded-lg">
           <div className="flex items-center gap-1">
             <div className={cn("w-2 h-2 rounded-full", images.neutral ? "bg-green-500" : "bg-slate-300")} /> Neutral
           </div>
           <div className="flex items-center gap-1">
             <div className={cn("w-2 h-2 rounded-full", images.tongue ? "bg-green-500" : "bg-slate-300")} /> Tongue
           </div>
           <div className="flex items-center gap-1">
             <div className={cn("w-2 h-2 rounded-full", images.headup ? "bg-green-500" : "bg-slate-300")} /> Head Up
           </div>
        </div>
      </div>

      {/* Test Samples Gallery */}
      <section className="bg-white dark:bg-slate-800 p-6 rounded-3xl border border-slate-100 dark:border-slate-700 shadow-sm space-y-4">
        <h3 className="font-bold text-sm flex items-center gap-2 text-slate-400 uppercase tracking-wider">
           <Zap size={16} className="text-yellow-500" />
           Quick Test Samples
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          {samples.map((sample, i) => (
            <button
              key={i}
              type="button"
              onClick={() => handleSampleClick(sample.type, sample.url, `${sample.type}_test.jpg`)}
              className="group relative h-20 rounded-xl overflow-hidden border-2 border-transparent hover:border-primary transition-all active:scale-95"
            >
              <img src={sample.url} alt={sample.label} className="w-full h-full object-cover opacity-80 group-hover:opacity-100" />
              <div className="absolute inset-x-0 bottom-0 bg-black/40 text-[8px] text-white py-1">
                {sample.cat}
              </div>
            </button>
          ))}
        </div>
      </section>

      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <UploadBox 
            id="neutral"
            label="Neutral Face" 
            sublabel="Facing camera, mouth closed"
            preview={previews.neutral}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleFileChange(e, 'neutral')}
          />
          <UploadBox 
            id="tongue"
            label="Tongue Extended" 
            sublabel="Mouth open, tongue out"
            preview={previews.tongue}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleFileChange(e, 'tongue')}
          />
          <UploadBox 
            id="headup"
            label="Head Up" 
            sublabel="Sniffing position (Neck up)"
            preview={previews.headup}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleFileChange(e, 'headup')}
          />
        </div>

        <button
          type="submit"
          disabled={!isUploadComplete || loading}
          className={cn(
            "w-full py-5 rounded-2xl font-bold text-lg transition-all flex items-center justify-center gap-3",
            isUploadComplete && !loading
              ? "bg-primary text-white hover:bg-primary-hover shadow-lg shadow-primary/20"
              : "bg-slate-200 text-slate-400 cursor-not-allowed"
          )}
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" />
              Analyzing Airway Anatomy...
            </>
          ) : (
            <>
              Predict Intubation Risk
              <ChevronRight size={20} />
            </>
          )}
        </button>
      </form>

      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-6 bg-red-50 text-red-700 border border-red-100 rounded-2xl flex items-center gap-4"
          >
            <AlertCircle className="shrink-0" />
            <p className="font-medium">{error}</p>
          </motion.div>
        )}

        {result && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-8 bg-white dark:bg-slate-800 border-2 border-slate-100 dark:border-slate-700 rounded-3xl shadow-xl space-y-8"
          >
            <div className="flex items-center justify-between border-b border-slate-50 dark:border-slate-700 pb-6">
              <h3 className="font-bold text-xl flex items-center gap-2">
                <CheckCircle2 className="text-green-500" />
                Prediction Result
              </h3>
              <div className="text-right">
                <span className="text-sm text-slate-500 block uppercase tracking-wider font-bold">Confidence Score</span>
                <span className="text-3xl font-black text-primary">{result.confidence.toFixed(1)}%</span>
              </div>
            </div>

            <div className="flex flex-col items-center py-4">
              <div className={cn(
                "px-8 py-3 rounded-2xl text-2xl font-black mb-8",
                result.prediction === "Easy" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
              )}>
                {result.prediction} Intubation
              </div>

              <div className="w-full grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-bold">
                    <span>Easy</span>
                    <span>{result.probabilities.Easy.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }} 
                      animate={{ width: `${result.probabilities.Easy}%` }}
                      className="h-full bg-green-500" 
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-bold">
                    <span>Difficult</span>
                    <span>{result.probabilities.Difficult.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }} 
                      animate={{ width: `${result.probabilities.Difficult}%` }}
                      className="h-full bg-red-400" 
                    />
                  </div>
                </div>
              </div>
            </div>

            <p className="text-center text-xs text-slate-500">
               * Interpret results in context of bedside airway exams, patient history, and professional judgment.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="p-6 bg-blue-50/50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800 rounded-3xl">
        <h4 className="font-bold flex items-center gap-2 mb-2">
           <AlertCircle size={18} className="text-blue-500" />
           Important Clarification
        </h4>
        <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
          Capture images in good lighting with the entire face, jaw, tongue, and neck clearly visible. Use a neutral background when possible. This tool is for training and education—it does not replace comprehensive clinical assessment.
        </p>
      </div>
    </div>
  );
}

function UploadBox({ id, label, sublabel, preview, onChange }: any) {
  return (
    <div className="space-y-3">
      <label htmlFor={id} className="block group cursor-pointer h-full">
        <div className={cn(
          "relative h-56 rounded-3xl border-2 border-dashed transition-all flex flex-col items-center justify-center p-4 text-center overflow-hidden",
          preview 
            ? "border-primary bg-blue-50/10" 
            : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-primary hover:bg-blue-50 dark:hover:bg-blue-900/10"
        )}>
          {preview ? (
            <>
              <img src={preview} alt={label} className="absolute inset-0 w-full h-full object-cover" />
              <div className="absolute inset-x-0 bottom-0 bg-black/60 p-3 text-white backdrop-blur-sm">
                 <p className="text-xs font-bold uppercase tracking-wider">{label}</p>
                 <p className="text-[10px] opacity-70">Click to change</p>
              </div>
            </>
          ) : (
            <>
              <div className="w-12 h-12 rounded-2xl bg-slate-100 dark:bg-slate-700 flex items-center justify-center text-slate-400 mb-4 group-hover:scale-110 transition-transform">
                <Camera size={24} />
              </div>
              <p className="font-bold text-sm mb-1">{label}</p>
              <p className="text-[10px] text-slate-400">{sublabel}</p>
            </>
          )}
        </div>
        <input type="file" id={id} className="hidden" accept="image/*" onChange={onChange} />
      </label>
    </div>
  );
}
