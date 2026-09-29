export default function StatCard({ label, value, hint, tone = 'default' }) {
  const tones = {
    default: 'border-slate-200 bg-white',
    success: 'border-emerald-200 bg-emerald-50',
    danger: 'border-red-200 bg-red-50',
    warning: 'border-amber-200 bg-amber-50',
    info: 'border-sky-200 bg-sky-50',
  };

  return (
    <div className={`rounded-lg border p-4 ${tones[tone] || tones.default}`}>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-slate-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}
