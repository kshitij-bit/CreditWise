import { NavLink } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: '▦', group: 'Overview' },
  { to: '/assessment', label: 'New Assessment', icon: '+', group: 'Assessment' },
  { to: '/history', label: 'Prediction History', icon: '↺', group: 'Assessment' },
  { to: '/model-info', label: 'Model Insights', icon: '⌁', group: 'Insights' },
];

export default function Sidebar() {
  const { user } = useAuth();
  const isAdmin = user && user.role === 'ADMIN';

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-[#dfe3f0] bg-white text-[#181c20] lg:flex">
      <div className="flex h-16 items-center gap-2 border-b border-[#e7e9f1] px-5">
        <span className="flex h-7 w-7 items-center justify-center rounded-md bg-[#2457d6] text-sm font-bold text-white">C</span>
        <div>
          <p className="text-sm font-bold leading-tight text-[#181c20]">CreditWise AI</p>
          <p className="text-[10px] text-[#6d7486]">Credit Risk Platform</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-5">
        {['Overview', 'Assessment', 'Insights'].map((group) => (
          <div key={group} className="mb-5">
            <p className="mb-2 px-3 text-[9px] font-bold uppercase tracking-[0.12em] text-[#8b91a3]">{group}</p>
            {navItems.filter((item) => item.group === group).map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `mb-1 flex items-center gap-3 rounded-md px-3 py-2 text-xs font-medium transition ${
                    isActive ? 'bg-[#2457d6] text-white shadow-sm' : 'text-[#565e71] hover:bg-[#f0f3fb] hover:text-[#2457d6]'
                  }`
                }
              >
                <span className="w-4 text-center text-sm">{item.icon}</span>
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}

        {isAdmin && (
          <NavLink
            to="/admin/users"
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-xs transition ${
                isActive
                  ? 'bg-[#2457d6] text-white'
                  : 'text-[#565e71] hover:bg-[#f0f3fb] hover:text-[#2457d6]'
              }`
            }
          >
            <span className="w-4 text-center">◎</span>
            User Management
          </NavLink>
        )}
      </nav>

      <div className="border-t border-[#e7e9f1] px-5 py-4 text-[10px] text-[#6d7486]">
        <p className="font-semibold text-[#565e71]">Model status</p>
        <p className="mt-1"><span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-[#12b76a]" /> creditwise-xgb-v1</p>
      </div>
    </aside>
  );
}
