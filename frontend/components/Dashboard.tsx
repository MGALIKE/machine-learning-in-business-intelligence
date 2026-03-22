'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, PieChart, Pie, Cell, AreaChart, Area, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import {
  TrendingDown, Users, DollarSign, AlertTriangle, Activity,
  Shield, Target, Zap, ArrowUpRight, ArrowDownRight, BarChart3,
  PieChart as PieIcon, Clock, Lightbulb, ChevronRight
} from 'lucide-react';
import { mockData } from '@/data/mockData';

/* ── animation variants ─────────────────────────────────── */
const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } }
};

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' as const } }
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.92 },
  show: { opacity: 1, scale: 1, transition: { duration: 0.5, ease: 'easeOut' as const } }
};

/* ── animated counter ───────────────────────────────────── */
function AnimatedNumber({ value, prefix = '', suffix = '', duration = 1.2 }: {
  value: number; prefix?: string; suffix?: string; duration?: number;
}) {
  const [display, setDisplay] = useState(0);
  
  useEffect(() => {
    let animationFrameId: number;
    const startTime = performance.now();
    
    const step = (now: number) => {
      const progress = Math.min((now - startTime) / (duration * 1000), 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(Math.round(eased * value));
      
      if (progress < 1) {
        animationFrameId = requestAnimationFrame(step);
      }
    };
    
    animationFrameId = requestAnimationFrame(step);
    
    // Crucial cleanup to prevent orphaned RAF loops that freeze the browser
    return () => cancelAnimationFrame(animationFrameId);
  }, [value, duration]);
  
  return <>{prefix}{display.toLocaleString()}{suffix}</>;
}

/* ── tooltip style ──────────────────────────────────────── */
const tooltipStyle = {
  backgroundColor: 'rgba(12,12,16,0.95)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '10px',
  color: '#e2e8f0',
  fontSize: '13px',
  boxShadow: '0 8px 32px rgba(0,0,0,0.5)'
};

/* ── main dashboard ─────────────────────────────────────── */
export default function Dashboard() {
  return (
    <motion.div
      className="w-full min-h-screen p-8 relative z-10"
      variants={container}
      initial="hidden"
      animate="show"
    >
      {/* Header */}
      <motion.header variants={fadeUp} className="mb-10">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">
              Churn Intelligence
            </h1>
            <p className="text-sm text-gray-500 mt-1 font-medium tracking-wide">
              Real-time ML-powered retention analytics
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full glass-subtle">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
              </span>
              <span className="text-xs font-medium text-emerald-400 uppercase tracking-widest">Pipeline Active</span>
            </div>
          </div>
        </div>
      </motion.header>

      {/* KPI Row */}
      <motion.div variants={fadeUp} className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5 mb-8">
        <KpiCard
          icon={<TrendingDown size={20} />}
          title="Churn Rate"
          value={<AnimatedNumber value={mockData.kpis.overallChurnRate} suffix="%" />}
          subtitle="of total customer base"
          accentColor="rose"
          trend={{ direction: 'down', label: '2.1% vs baseline' }}
        />
        <KpiCard
          icon={<DollarSign size={20} />}
          title="Avg. Lifetime Value"
          value={<AnimatedNumber value={mockData.kpis.avgClv} prefix="$" />}
          subtitle="per retained customer"
          accentColor="emerald"
          trend={{ direction: 'up', label: 'Top predictor' }}
        />
        <KpiCard
          icon={<AlertTriangle size={20} />}
          title="At-Risk Customers"
          value={<AnimatedNumber value={mockData.kpis.highRiskCount} />}
          subtitle="predicted by ML model"
          accentColor="amber"
          trend={{ direction: 'down', label: 'Action required' }}
        />
        <KpiCard
          icon={<Shield size={20} />}
          title="Revenue at Risk"
          value={<AnimatedNumber value={mockData.kpis.revenueAtRisk} prefix="$" />}
          subtitle="recoverable via retention"
          accentColor="violet"
          trend={{ direction: 'down', label: 'Preventable loss' }}
        />
      </motion.div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-8">
        {/* Feature Importance */}
        <motion.div variants={scaleIn} className="lg:col-span-7 glass p-6">
          <div className="flex items-center gap-2 mb-6">
            <BarChart3 size={18} className="text-emerald-400" />
            <h2 className="text-base font-semibold text-white">Top Churn Drivers</h2>
            <span className="ml-auto text-xs text-gray-500 font-medium">ML Feature Importance</span>
          </div>
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockData.churnDrivers} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" horizontal={false} />
                <XAxis type="number" stroke="#555" tick={{ fontSize: 11 }} tickFormatter={(v) => v.toFixed(2)} />
                <YAxis dataKey="feature" type="category" width={130} stroke="transparent" tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.02)' }} contentStyle={tooltipStyle} />
                <Bar dataKey="importance" radius={[0, 6, 6, 0]} barSize={18} fill="url(#barGrad)" />
                <defs>
                  <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#059669" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#34d399" stopOpacity={1} />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Risk Segmentation */}
        <motion.div variants={scaleIn} className="lg:col-span-5 glass p-6">
          <div className="flex items-center gap-2 mb-6">
            <PieIcon size={18} className="text-violet-400" />
            <h2 className="text-base font-semibold text-white">Risk Distribution</h2>
          </div>
          <div className="h-[240px]">
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={mockData.riskSegments}
                  cx="50%" cy="50%"
                  innerRadius={65} outerRadius={90}
                  paddingAngle={4}
                  dataKey="value"
                  stroke="none"
                  animationBegin={300}
                  animationDuration={1200}
                >
                  {mockData.riskSegments.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-3 mt-2">
            {mockData.riskSegments.map((seg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 + i * 0.15 }}
                className="text-center"
              >
                <div className="text-xl font-bold" style={{ color: seg.color }}>{seg.value}</div>
                <div className="text-[11px] text-gray-500 mt-0.5">{seg.name}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Second Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-8">
        {/* Tenure Impact */}
        <motion.div variants={scaleIn} className="lg:col-span-7 glass p-6">
          <div className="flex items-center gap-2 mb-6">
            <Clock size={18} className="text-sky-400" />
            <h2 className="text-base font-semibold text-white">Churn by Customer Tenure</h2>
            <span className="ml-auto text-xs text-gray-500 font-medium">Correlation Analysis</span>
          </div>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockData.tenureImpact} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                <XAxis dataKey="tenure" stroke="#555" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <YAxis stroke="#555" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(val: any) => [`${val}%`, 'Churn Rate']}
                />
                <Area
                  type="monotone" dataKey="churnRate"
                  stroke="#8b5cf6" strokeWidth={2.5}
                  fill="url(#areaGrad)"
                  dot={{ r: 4, fill: '#8b5cf6', stroke: '#050507', strokeWidth: 2 }}
                  activeDot={{ r: 6, fill: '#a78bfa', stroke: '#050507', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Model Performance */}
        <motion.div variants={scaleIn} className="lg:col-span-5 glass p-6">
          <div className="flex items-center gap-2 mb-6">
            <Activity size={18} className="text-emerald-400" />
            <h2 className="text-base font-semibold text-white">Model Performance</h2>
          </div>
          <div className="space-y-4">
            {[
              { label: 'F1 Score', value: mockData.modelPerformance.f1, color: '#34d399' },
              { label: 'Precision', value: mockData.modelPerformance.precision, color: '#38bdf8' },
              { label: 'Recall', value: mockData.modelPerformance.recall, color: '#a78bfa' },
              { label: 'AUC-ROC', value: mockData.modelPerformance.roc_auc, color: '#fbbf24' },
            ].map((metric, i) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.12 }}
              >
                <div className="flex justify-between mb-1.5">
                  <span className="text-sm text-gray-400">{metric.label}</span>
                  <span className="text-sm font-semibold text-white">{metric.value}%</span>
                </div>
                <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: metric.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${metric.value}%` }}
                    transition={{ duration: 1.2, delay: 0.6 + i * 0.12 }}
                  />
                </div>
              </motion.div>
            ))}
          </div>
          <div className="mt-6 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/10">
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-emerald-400" />
              <span className="text-xs font-medium text-emerald-400">Champion: {mockData.modelPerformance.champion_model} Classifier</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* ═══════════════════════════════════════════════════════ */}
      {/*  SECTION: CHURN REDUCTION STRATEGIES                  */}
      {/* ═══════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp} className="mb-8">
        <div className="flex items-center gap-2 mb-2">
          <Shield size={18} className="text-emerald-400" />
          <h2 className="text-lg font-semibold text-white">How to Reduce Churn</h2>
        </div>
        <p className="text-sm text-gray-500 mb-5">Data-driven strategies ranked by ML feature importance</p>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {mockData.churnReduction.map((item, i) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.1, duration: 0.45 }}
              whileHover={{ y: -3, transition: { duration: 0.2 } }}
              className="glass p-5 group cursor-default relative overflow-hidden"
            >
              {/* Top accent */}
              <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-60" />

              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <h3 className="text-sm font-semibold text-white leading-tight pr-2">{item.title}</h3>
                <div className="flex gap-1.5 shrink-0">
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                    item.impact === 'High' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-sky-500/15 text-sky-400'
                  }`}>{item.impact} Impact</span>
                </div>
              </div>

              {/* Description */}
              <p className="text-[13px] text-gray-400 leading-relaxed mb-4">{item.description}</p>

              {/* Data source */}
              <div className="text-[11px] text-gray-600 mb-4 font-mono bg-surface-800/50 px-2.5 py-1.5 rounded-md border border-white/[0.03]">
                {item.dataSource}
              </div>

              {/* Footer */}
              <div className="flex items-end justify-between pt-3 border-t border-white/5">
                <div>
                  <div className="text-lg font-bold text-emerald-400">{item.kpi}</div>
                  <div className="text-[10px] text-gray-500">{item.kpiLabel}</div>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-gray-500">
                  <Clock size={12} />
                  <span>{item.timeline}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════ */}
      {/*  SECTION: WIN-BACK CAMPAIGNS                          */}
      {/* ═══════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp} className="mb-8">
        <div className="flex items-center gap-2 mb-2">
          <Target size={18} className="text-rose-400" />
          <h2 className="text-lg font-semibold text-white">Win-Back Campaigns</h2>
        </div>
        <p className="text-sm text-gray-500 mb-5">Re-engagement strategies for the 64 predicted churners</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {mockData.winBackCampaigns.map((camp, i) => (
            <motion.div
              key={camp.id}
              initial={{ opacity: 0, x: i % 2 === 0 ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.2 + i * 0.12, duration: 0.5 }}
              whileHover={{ y: -2, transition: { duration: 0.2 } }}
              className="glass p-5 relative overflow-hidden group cursor-default"
            >
              {/* Left accent bar */}
              <div className="absolute top-0 left-0 w-[3px] h-full bg-gradient-to-b from-rose-500 via-rose-400 to-transparent" />

              <div className="pl-3">
                <h3 className="text-sm font-semibold text-white mb-2">{camp.title}</h3>
                <p className="text-[13px] text-gray-400 leading-relaxed mb-4">{camp.description}</p>

                {/* Campaign details grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-surface-800/40 rounded-lg p-2.5 border border-white/[0.03]">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Target</div>
                    <div className="text-xs font-medium text-white">{camp.segment}</div>
                  </div>
                  <div className="bg-surface-800/40 rounded-lg p-2.5 border border-white/[0.03]">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Channel</div>
                    <div className="text-xs font-medium text-white">{camp.channel}</div>
                  </div>
                  <div className="bg-surface-800/40 rounded-lg p-2.5 border border-white/[0.03]">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Expected ROI</div>
                    <div className="text-xs font-bold text-emerald-400">{camp.expectedROI}</div>
                  </div>
                  <div className="bg-surface-800/40 rounded-lg p-2.5 border border-white/[0.03]">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Success Rate</div>
                    <div className="text-xs font-bold text-sky-400">{camp.successRate}</div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* ═══════════════════════════════════════════════════════ */}
      {/*  SECTION: DATA-DRIVEN INSIGHTS                        */}
      {/* ═══════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp}>
        <div className="flex items-center gap-2 mb-2">
          <Lightbulb size={18} className="text-amber-400" />
          <h2 className="text-lg font-semibold text-white">Additional Data Insights</h2>
        </div>
        <p className="text-sm text-gray-500 mb-5">Key patterns discovered by the ML pipeline from 61 engineered features</p>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
          {mockData.additionalInsights.map((insight, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.5 + i * 0.1, duration: 0.4 }}
              whileHover={{ y: -2, transition: { duration: 0.2 } }}
              className="glass p-5 cursor-default group"
            >
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded-md bg-amber-500/10 flex items-center justify-center">
                  <Zap size={12} className="text-amber-400" />
                </div>
                <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">{insight.metricLabel}: {insight.metric}</span>
              </div>
              <h3 className="text-sm font-semibold text-white mb-2">{insight.title}</h3>
              <p className="text-[12px] text-gray-400 leading-relaxed">{insight.insight}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── KPI Card ───────────────────────────────────────────── */
function KpiCard({ icon, title, value, subtitle, accentColor, trend }: {
  icon: React.ReactNode; title: string; value: React.ReactNode;
  subtitle: string; accentColor: string;
  trend: { direction: 'up' | 'down'; label: string };
}) {
  const colorMap: Record<string, string> = {
    emerald: '#34d399', rose: '#f43f5e', amber: '#f59e0b', violet: '#8b5cf6', sky: '#38bdf8'
  };
  const glowMap: Record<string, string> = {
    emerald: 'glow-emerald', rose: 'glow-rose', amber: '', violet: 'glow-violet', sky: 'glow-sky'
  };
  const color = colorMap[accentColor] || '#34d399';

  return (
    <motion.div
      variants={fadeUp}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
      className={`glass p-5 cursor-default group ${glowMap[accentColor] || ''}`}
      style={{ borderColor: `${color}10` }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}10` }}>
          <div style={{ color }}>{icon}</div>
        </div>
        <div className="flex items-center gap-1 text-xs font-medium" style={{ color }}>
          {trend.direction === 'up' ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
          <span>{trend.label}</span>
        </div>
      </div>
      <div className="text-2xl font-bold text-white mb-1 tracking-tight">{value}</div>
      <div className="text-xs text-gray-500">{title} &middot; {subtitle}</div>
    </motion.div>
  );
}

