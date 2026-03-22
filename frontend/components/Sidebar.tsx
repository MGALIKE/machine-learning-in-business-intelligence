'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Users, Brain } from 'lucide-react';

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <motion.nav
      initial={{ x: -80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="w-[72px] lg:w-[240px] border-r border-white/[0.04] bg-surface-900/60 backdrop-blur-2xl flex flex-col h-screen sticky top-0 z-20 shrink-0"
    >
      {/* Logo */}
      <div className="p-4 lg:px-5 lg:py-6 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center font-bold text-white text-sm shadow-lg shadow-emerald-500/20 shrink-0">
          B
        </div>
        <div className="hidden lg:block">
          <span className="font-bold text-lg text-white tracking-tight">BELEK</span>
          <p className="text-[10px] text-gray-500 -mt-0.5 uppercase tracking-[0.2em]">Analytics</p>
        </div>
      </div>

      {/* Nav Items */}
      <div className="flex-1 py-4 flex flex-col gap-1 px-3">
        <NavItem 
          href="/"
          icon={<LayoutDashboard size={18} />} 
          label="Dashboard" 
          active={pathname === '/'} 
        />
        <NavItem 
          href="/customers"
          icon={<Users size={18} />} 
          label="Customers" 
          active={pathname === '/customers'} 
        />
        <NavItem 
          href="/models"
          icon={<Brain size={18} />} 
          label="ML Models" 
          active={pathname === '/models'} 
        />
      </div>

      {/* User */}
      <div className="p-4 border-t border-white/[0.04]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shrink-0">
            <span className="text-xs font-bold text-white">CTO</span>
          </div>
          <div className="hidden lg:block flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">Team Belek</p>
            <p className="text-[11px] text-gray-500">Hackathon 2026</p>
          </div>
        </div>
      </div>
    </motion.nav>
  );
}

function NavItem({ href, icon, label, active = false }: {
  href: string; icon: React.ReactNode; label: string; active?: boolean;
}) {
  return (
    <Link 
      href={href}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group relative
      ${active
        ? 'bg-white/[0.06] text-white'
        : 'text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]'}
    `}>
      {active && (
        <motion.div
          layoutId="activeNav"
          className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 bg-emerald-400 rounded-r-full"
          transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }}
        />
      )}
      <span className={`shrink-0 ${active ? 'text-emerald-400' : ''}`}>{icon}</span>
      <span className="text-sm font-medium hidden lg:block">{label}</span>
    </Link>
  );
}
