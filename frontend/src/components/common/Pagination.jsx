export default function Pagination({ page, totalPages, onPageChange }) {
  if (totalPages <= 1) return null;

  const pages = [];
  for (let p = 1; p <= totalPages; p++) {
    if (p === 1 || p === totalPages || Math.abs(p - page) <= 1) {
      pages.push(p);
    } else if (pages[pages.length - 1] !== '…') {
      pages.push('…');
    }
  }

  return (
    <nav className="mt-4 flex items-center justify-between border-t border-slate-200 pt-4">
      <button
        type="button"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
        className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
      >
        ← Prev
      </button>

      <div className="flex items-center gap-1">
        {pages.map((p, idx) =>
          p === '…' ? (
            <span key={`ellipsis-${idx}`} className="px-1 text-slate-400">…</span>
          ) : (
            <button
              key={p}
              type="button"
              onClick={() => onPageChange(p)}
              className={`rounded-md px-3 py-1.5 text-sm transition ${
                p === page
                  ? 'bg-indigo-600 font-semibold text-white'
                  : 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50'
              }`}
            >
              {p}
            </button>
          )
        )}
      </div>

      <button
        type="button"
        disabled={page >= totalPages}
        onClick={() => onPageChange(page + 1)}
        className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Next →
      </button>
    </nav>
  );
}
