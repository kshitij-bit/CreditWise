import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

export default function Header() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [open, setOpen] = useState(false);

  const displayName = user?.name || user?.email || 'User';
  const initials = displayName
    .split(' ')
    .map((p) => p[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();
  const pageNames = {
    '/dashboard': 'Portfolio Risk Dashboard',
    '/assessment': 'New Assessment',
    '/history': 'Prediction History',
    '/model-info': 'Model Insights',
    '/model-detail': 'Model Details',
  };

  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-[#e1e5f0] bg-white px-4 sm:px-6">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#8b91a3]">CreditWise AI</p>
        <p className="text-sm font-semibold text-[#181c20]">
          {pageNames[location.pathname] || 'Credit Risk Platform'}
        </p>
      </div>

      <div className="relative">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-3 rounded-md px-2 py-1 transition hover:bg-[#f3f5fb]"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[#2457d6] text-xs font-semibold text-white">
            {initials}
          </span>
          <span className="hidden text-sm text-slate-700 sm:block">{displayName}</span>
          <span className="text-xs text-[#8b91a3]">⌄</span>
        </button>

        {open && (
          <div className="absolute right-0 mt-2 w-56 rounded-md border border-slate-200 bg-white py-1 shadow-lg">
            <div className="border-b border-slate-100 px-4 py-2">
              <p className="text-sm font-medium text-slate-800">{displayName}</p>
              <p className="text-xs text-slate-500">{user?.email}</p>
              <span className="mt-1 inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-slate-600">
                {user?.role}
              </span>
            </div>
            <Link
              to="/model-info"
              className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50"
            >
              Model Information
            </Link>
            <button
              onClick={logout}
              className="block w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50"
            >
              Sign Out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
