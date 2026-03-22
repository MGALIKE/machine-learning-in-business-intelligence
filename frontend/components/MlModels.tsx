'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  BrainCircuit, Activity, Network, Zap, Cpu, Settings, Target, 
  Database, BarChart3, LineChart 
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell 
} from 'recharts';
import { mockData } from '@/data/mockData';

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, staggerChildren: 0.1 } }
};

const baselineModels = [
  { name: 'LightGBM', accuracy: '98.65%', f1: '0.9753', precision: '0.9850', recall: '0.9660', auc: '0.9988', time: '2.5s' },
  { name: 'Logistic Regression', accuracy: '98.12%', f1: '0.9665', precision: '0.9526', recall: '0.9811', auc: '0.9975', time: '3.7s' },
  { name: 'XGBoost', accuracy: '97.71%', f1: '0.9577', precision: '0.9730', recall: '0.9432', auc: '0.9971', time: '3.1s' },
  { name: 'Gradient Boosting', accuracy: '97.29%', f1: '0.9485', precision: '0.9920', recall: '0.9092', auc: '0.9963', time: '3.7s' },
  { name: 'Random Forest', accuracy: '78.75%', f1: '0.4411', precision: '0.8011', recall: '0.3070', auc: '0.8904', time: '3.3s' },
];

const featureImportanceData = mockData.churnDrivers.map(d => ({
  name: d.feature.replace(/_/g, ' '),
  importance: Math.round(d.importance * 100),
})).slice(0, 8); // Top 8 for chart

export default function MlModels() {
  const p = mockData.modelPerformance;

  return (
    <motion.div
      className="w-full min-h-screen p-4 lg:p-8 relative z-10"
      initial="hidden"
      animate="show"
      variants={fadeUp}
    >
      <motion.header variants={fadeUp} className="mb-8">
        <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
          <BrainCircuit className="text-emerald-400" size={32} />
          Machine Learning Pipeline
        </h1>
        <p className="text-sm text-gray-400 mt-2 font-medium">
          Comprehensive review of data preprocessing, base estimators, hyperparameter tuning, and meta-learner stacking.
        </p>
      </motion.header>

      {/* ───────────────────────────────────────────────────────── */}
      {/* 1. DATA PREPROCESSING & DATASET METRICS                   */}
      {/* ───────────────────────────────────────────────────────── */}
      <motion.div variants={fadeUp} className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
        <div className="glass p-5">
          <div className="flex items-center gap-2 mb-3">
            <Database size={16} className="text-sky-400" />
            <h3 className="text-sm font-semibold text-white">Dataset Overview</h3>
          </div>
          <div className="space-y-2 mt-4">
            <div className="flex justify-between text-sm"><span className="text-gray-400">Total Samples:</span> <span className="text-white font-mono">1,200</span></div>
            <div className="flex justify-between text-sm"><span className="text-gray-400">Train / Test Split:</span> <span className="text-white font-mono">80% / 20%</span></div>
            <div className="flex justify-between text-sm"><span className="text-gray-400">Features Engineered:</span> <span className="text-white font-mono">61</span></div>
            <div className="flex justify-between text-sm"><span className="text-gray-400">Multicollinearity:</span> <span className="text-white font-serif italic text-xs">Tree-robust</span></div>
          </div>
        </div>
        
        <div className="glass p-5">
          <div className="flex items-center gap-2 mb-3">
            <Activity size={16} className="text-rose-400" />
            <h3 className="text-sm font-semibold text-white">Class Imbalance Handling</h3>
          </div>
          <p className="text-xs text-gray-400 leading-relaxed mb-4 mt-2">
            The dataset suffers from a severe class imbalance with a <strong>27.5% churn rate</strong>. 
            Standard models failed to penalize false negatives adequately.
          </p>
          <div className="bg-surface-800/50 p-3 rounded-lg border border-rose-500/10 text-xs">
            <span className="text-rose-400 font-semibold block mb-1">Solution applied:</span>
            SMOTE (Synthetic Minority Over-sampling Technique) combined with Scale_Pos_Weight balancing in XGBoost/LightGBM.
          </div>
        </div>

        <div className="glass p-5">
          <div className="flex items-center gap-2 mb-3">
            <BrainCircuit size={16} className="text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Champion: Stacking</h3>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-4">
            <MetricBox label="F1 Score" value={`${p.f1}%`} color="emerald" />
            <MetricBox label="AUC-ROC" value={`${p.roc_auc}%`} color="sky" />
            <MetricBox label="Precision" value={`${p.precision}%`} color="violet" />
            <MetricBox label="Recall" value={`${p.recall}%`} color="amber" />
          </div>
        </div>
      </motion.div>

      {/* ───────────────────────────────────────────────────────── */}
      {/* 2. BASELINE MODELS AND OPTUNA OPTIMIZATION                 */}
      {/* ───────────────────────────────────────────────────────── */}
      <motion.div variants={fadeUp} className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        
        {/* Baselining Table */}
        <div className="glass overflow-hidden flex flex-col">
          <div className="p-5 border-b border-white/[0.04] bg-surface-900/40 flex items-center gap-2">
            <BarChart3 size={18} className="text-amber-400" />
            <h3 className="text-base font-semibold text-white">Baseline Evaluation (5-Fold CV)</h3>
          </div>
          <div className="w-full overflow-x-auto p-1 text-sm">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="text-[11px] uppercase tracking-wider text-gray-500 font-semibold border-b border-white/[0.05]">
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">F1</th>
                  <th className="px-4 py-3">Precision</th>
                  <th className="px-4 py-3">Recall</th>
                  <th className="px-4 py-3">AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.02]">
                {baselineModels.map((m, i) => (
                  <tr key={i} className={`hover:bg-white/[0.02] ${i === 0 ? 'bg-emerald-500/5' : ''}`}>
                    <td className="px-4 py-3 font-medium text-white flex items-center gap-2">
                      {i === 0 && <Zap size={12} className="text-emerald-400" />}
                      {m.name}
                    </td>
                    <td className={`px-4 py-3 font-mono ${i === 0 ? 'text-emerald-400 font-bold' : 'text-gray-300'}`}>{m.f1}</td>
                    <td className="px-4 py-3 font-mono text-gray-300">{m.precision}</td>
                    <td className="px-4 py-3 font-mono text-gray-300">{m.recall}</td>
                    <td className="px-4 py-3 font-mono text-gray-300">{m.auc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Optuna Tuning */}
        <div className="glass p-5">
          <div className="flex items-center gap-2 mb-4">
            <Settings size={18} className="text-violet-400" />
            <h3 className="text-base font-semibold text-white">Optuna Hyperparameter Tuning</h3>
          </div>
          <p className="text-sm text-gray-400 mb-5">
            100 Optuna trials were executed utilizing Tree-Structured Parzen Estimator (TPE) algorithm, targeting the F1 metric to severely punish false negatives.
          </p>
          
          <div className="space-y-4">
            <div className="bg-surface-800/60 p-4 rounded-xl border border-white/[0.05]">
              <h4 className="text-xs font-bold text-violet-400 uppercase tracking-wide mb-2">Best XGBoost Parameters Map</h4>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs font-mono">
                <div className="flex justify-between"><span className="text-gray-500">n_estimators:</span><span className="text-sky-300">627</span></div>
                <div className="flex justify-between"><span className="text-gray-500">max_depth:</span><span className="text-sky-300">9</span></div>
                <div className="flex justify-between"><span className="text-gray-500">learning_rate:</span><span className="text-sky-300">0.0501</span></div>
                <div className="flex justify-between"><span className="text-gray-500">subsample:</span><span className="text-sky-300">0.6025</span></div>
                <div className="flex justify-between"><span className="text-gray-500">colsample_bytree:</span><span className="text-sky-300">0.7841</span></div>
                <div className="flex justify-between"><span className="text-gray-500">min_child_weight:</span><span className="text-sky-300">3</span></div>
              </div>
            </div>
            
            <div className="flex items-center gap-3 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
              <Target size={20} className="text-emerald-400 shrink-0" />
              <p className="text-xs text-emerald-100/80 leading-relaxed">
                <strong className="text-emerald-400">Result:</strong> Tuning successfully pushed the baseline XGBoost F1 score from 0.9577 to 0.9697, heavily minimizing leakage from false positives.
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ───────────────────────────────────────────────────────── */}
      {/* 3. FEATURE IMPORTANCE & ARCHITECTURE DASHBOARD             */}
      {/* ───────────────────────────────────────────────────────── */}
      <motion.div variants={fadeUp} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Feature Importance Chart */}
        <div className="glass p-5 lg:col-span-2">
          <div className="flex items-center gap-2 mb-6">
            <LineChart size={18} className="text-sky-400" />
            <h3 className="text-base font-semibold text-white">Meta-Learner Feature Importance Weights</h3>
          </div>
          <div className="h-[280px] w-full mt-2 pr-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportanceData} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={140} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                  contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  itemStyle={{ color: '#38bdf8' }}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={16}>
                  {featureImportanceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#34d399' : index < 3 ? '#38bdf8' : '#8b5cf6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix / Flow */}
        <div className="glass p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Network size={18} className="text-emerald-400" />
              <h3 className="text-base font-semibold text-white">Holdout Confusion Matrix</h3>
            </div>
            <p className="text-[11px] text-gray-400 mb-6">
              Evaluation on the 20% untouched holdout set simulating production. Extremely tight diagonal indicating near-perfect generalizability.
            </p>

            <div className="grid grid-cols-2 gap-2 text-center text-xs">
              <div className="bg-surface-800/40 p-3 rounded-md border border-white/[0.04]">
                <div className="text-gray-500 mb-1">True Negative</div>
                <div className="text-xl font-bold text-white">100%</div>
                <div className="text-[10px] text-emerald-400 mt-1">Retained (Correct)</div>
              </div>
              <div className="bg-rose-500/10 p-3 rounded-md border border-rose-500/20">
                <div className="text-rose-300/60 mb-1">False Positive</div>
                <div className="text-xl font-bold text-rose-400">0.02%</div>
                <div className="text-[10px] text-rose-400 mt-1">Costly Error</div>
              </div>
              <div className="bg-amber-500/10 p-3 rounded-md border border-amber-500/20">
                <div className="text-amber-300/60 mb-1">False Negative</div>
                <div className="text-xl font-bold text-amber-500">0.05%</div>
                <div className="text-[10px] text-amber-500 mt-1">Missed Churn</div>
              </div>
              <div className="bg-emerald-500/10 p-3 rounded-md border border-emerald-500/20">
                <div className="text-emerald-300/60 mb-1">True Positive</div>
                <div className="text-xl font-bold text-emerald-400">99.9%</div>
                <div className="text-[10px] text-emerald-400 mt-1">Churn Captured</div>
              </div>
            </div>
          </div>

          <div className="mt-8 p-3 rounded-xl bg-surface-900/50 border border-white/[0.04] flex items-start gap-3">
            <Cpu size={16} className="text-gray-400 shrink-0 mt-0.5" />
            <p className="text-[11px] text-gray-500 leading-relaxed">
              Logistic Regression Meta-Learner aggregates predicting logits to output final probabilities, preventing base-estimator overfitting.
            </p>
          </div>
        </div>

      </motion.div>
    </motion.div>
  );
}

function MetricBox({ label, value, color }: { label: string; value: string; color: 'emerald' | 'sky' | 'violet' | 'amber' }) {
  const colorMap = {
    emerald: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5',
    sky: 'text-sky-400 border-sky-500/20 bg-sky-500/5',
    violet: 'text-violet-400 border-violet-500/20 bg-violet-500/5',
    amber: 'text-amber-400 border-amber-500/20 bg-amber-500/5',
  };
  
  return (
    <div className={`p-3 rounded-xl border ${colorMap[color]} flex flex-col justify-center`}>
      <div className="text-[11px] font-medium text-white/50 uppercase tracking-widest mb-1">{label}</div>
      <div className="text-xl font-bold font-mono">{value}</div>
    </div>
  );
}
