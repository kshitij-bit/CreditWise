import { NavLink } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: '📊' },
  { to: '/assessment', label: 'New Assessment', icon: '🔍' },
  { to: '/history', label: 'Prediction History', icon: '🕘' },
  { to: '/model-info', label: 'Model Info', icon: '🧠' },
];

export default function Sidebar() {
  const { user } = useAuth();
  const isAdmin = user && user.role === 'ADMIN';

  return (
    <aside className="flex w-64 flex-col bg-slate-900 text-slate-100">
      <div className="flex items-center gap-2 border-b border-slate-700 px-5 py-4">
        <span className="text-xl">🛡️</span>
        <div>
          <p className="text-sm font-semibold leading-tight">CreditWise AI</p>
          <p className="text-xs text-slate-400">Credit Risk Platform</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <span>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}

        {isAdmin && (
          <NavLink
            to="/admin/users"
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <span>👥</span>
            User Management
          </NavLink>
        )}
      </nav>

      <div className="border-t border-slate-700 px-5 py-3 text-xs text-slate-400">
        creditwise-xgb-v1 · {new Date().getFullYear()}
      </div>
    </aside>
  );
}
