export default function LoadingSpinner({ label = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12 text-slate-500">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
      <p className="text-sm">{label}</p>
    </div>
  );
}
