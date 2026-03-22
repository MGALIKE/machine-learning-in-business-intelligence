'use client';

import React, { useState, useMemo, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, MoreHorizontal, User, AlertCircle, CheckCircle2 } from 'lucide-react';
import { mockData } from '@/data/mockData';

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' as const } }
};

export default function Customers() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [visibleCount, setVisibleCount] = useState(15);
  const observerTarget = useRef<HTMLDivElement>(null);

  const allCustomers = mockData.customerDirectory;

  // Filtered array
  const filteredCustomers = useMemo(() => {
    return allCustomers.filter((c: any) => {
      const matchesSearch = c.id.toLowerCase().includes(searchQuery.toLowerCase()) || 
                            c.status.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFilter = statusFilter === 'All' || c.status === statusFilter;
      return matchesSearch && matchesFilter;
    });
  }, [searchQuery, statusFilter, allCustomers]);

  // Infinite Scroll Observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisibleCount((prev) => Math.min(prev + 15, filteredCustomers.length));
        }
      },
      { threshold: 1.0 }
    );
    
    if (observerTarget.current) {
      observer.observe(observerTarget.current);
    }
    
    return () => observer.disconnect();
  }, [filteredCustomers.length]);

  return (
    <motion.div
      className="w-full min-h-screen p-4 lg:p-8 relative z-10"
      initial="hidden"
      animate="show"
      variants={{
        hidden: { opacity: 0 },
        show: { opacity: 1, transition: { staggerChildren: 0.08 } }
      }}
    >
      <motion.header variants={fadeUp} className="mb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Customer Directory</h1>
          <p className="text-sm text-gray-500 mt-1 font-medium tracking-wide">
            Detailed scoring for {allCustomers.length} profiled holdout accounts
          </p>
        </div>
      </motion.header>

      <motion.div variants={fadeUp} className="glass overflow-hidden">
        {/* Toolbar */}
        <div className="p-4 border-b border-white/[0.04] flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-surface-900/40">
          <div className="relative w-full sm:w-72">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input 
              type="text" 
              placeholder="Search by ID or Status..." 
              value={searchQuery}
              onChange={(e) => { setSearchQuery(e.target.value); setVisibleCount(15); }}
              className="w-full bg-surface-800/50 border border-white/[0.05] rounded-xl py-2 pl-9 pr-4 text-sm text-white focus:outline-none focus:border-emerald-500/50 transition-colors"
            />
          </div>
          
          <div className="flex items-center gap-2">
            <select 
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setVisibleCount(15); }}
              className="bg-surface-800/50 border border-white/[0.05] rounded-xl py-2 px-3 text-sm text-gray-300 focus:outline-none focus:border-emerald-500/50 appearance-none h-full outline-none"
            >
              <option value="All">All Risk Levels</option>
              <option value="At Risk">High Risk</option>
              <option value="Monitoring">Medium Risk</option>
              <option value="Healthy">Low Risk</option>
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="w-full overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[800px]">
            <thead>
              <tr className="border-b border-white/[0.04] bg-surface-900/20 text-[11px] uppercase tracking-wider text-gray-500 font-semibold">
                <th className="px-6 py-4">Customer ID</th>
                <th className="px-6 py-4">Est. CLV</th>
                <th className="px-6 py-4">Tenure (Mos)</th>
                <th className="px-6 py-4">Risk Metric (Prob)</th>
                <th className="px-6 py-4">Model Status</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.02]">
              {filteredCustomers.slice(0, visibleCount).map((c: any) => (
                <tr key={c.id} className="hover:bg-white/[0.02] transition-colors group">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-surface-800 flex items-center justify-center border border-white/[0.05]">
                        <User size={14} className="text-gray-400" />
                      </div>
                      <span className="text-sm font-medium text-white font-mono tracking-tight">{c.id}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-300 font-mono">${c.clv.toLocaleString()}</td>
                  <td className="px-6 py-4 text-sm text-gray-300 font-mono">{c.tenure}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-surface-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full rounded-full" 
                          style={{ 
                            width: `${c.riskScore}%`,
                            backgroundColor: c.riskScore > 75 ? '#f43f5e' : c.riskScore > 35 ? '#f59e0b' : '#34d399'
                          }} 
                        />
                      </div>
                      <span className="text-xs text-gray-400 font-mono">{c.riskScore}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border
                      ${c.status === 'At Risk' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' : 
                        c.status === 'Monitoring' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' : 
                        'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'}`
                    }>
                      {c.status === 'At Risk' || c.status === 'Monitoring' ? <AlertCircle size={10} /> : <CheckCircle2 size={10} />}
                      {c.status}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button className="text-gray-500 hover:text-white transition-colors p-1 opacity-0 group-hover:opacity-100">
                      <MoreHorizontal size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredCustomers.length === 0 && (
            <div className="p-8 text-center text-sm text-gray-500">
              No customers found matching "{searchQuery}" and "{statusFilter}".
            </div>
          )}
        </div>
        
        {/* Infinite Scroll Trigger */}
        {visibleCount < filteredCustomers.length && (
          <div ref={observerTarget} className="h-12 w-full flex items-center justify-center p-4">
            <span className="text-xs text-gray-500">Loading more customers...</span>
          </div>
        )}
        
        <div className="p-4 border-t border-white/[0.04] flex items-center justify-between text-xs text-gray-500 bg-surface-900/40">
          <span>Showing {Math.min(visibleCount, filteredCustomers.length)} of {filteredCustomers.length} filtered customers</span>
        </div>
      </motion.div>
    </motion.div>
  );
}
