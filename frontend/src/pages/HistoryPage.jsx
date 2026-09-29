import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { fetchPredictions } from '../api/predictions';
import { formatDate, formatScore } from '../lib/format';
import { getRiskCategoryStyle, getDecisionStyle } from '../lib/riskStyles';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';
import Pagination from '../components/common/Pagination';

const PAGE_SIZE = 10;

export default function HistoryPage() {
  const { user } = useAuth();
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async (targetPage) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPredictions({
        page: targetPage,
        page_size: PAGE_SIZE,
      });
      const rows = Array.isArray(data) ? data : data.items || [];
      setItems(rows);
      setTotal(data?.total ?? rows.length);
    } catch (err) {
      setError(err.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(page);
  }, [page]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Prediction History</h1>
          <p className="text-sm text-slate-500">
            Past credit risk assessments for {user?.email}
          </p>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {loading ? (
        <LoadingSpinner label="Loading predictions…" />
      ) : items.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-10 text-center">
          <p className="text-3xl">📭</p>
          <p className="mt-2 text-sm font-medium text-slate-700">No predictions yet</p>
          <p className="text-sm text-slate-500">
            Run your first assessment to see results here.
          </p>
          <Link
            to="/assessment"
            className="mt-4 inline-block rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
          >
            New Assessment
          </Link>
        </div>
      ) : (
        <>
          <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Date</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Score</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Risk</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Decision</th>
                  <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wide text-slate-500">View</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {items.map((item) => {
                  const risk = getRiskCategoryStyle(item.risk_category);
                  const decision = getDecisionStyle(item.decision);
                  return (
                    <tr key={item.id} className="hover:bg-slate-50">
                      <td className="whitespace-nowrap px-4 py-2.5 text-slate-600">
                        {formatDate(item.created_at)}
                      </td>
                      <td className="px-4 py-2.5 font-semibold text-slate-900">
                        {formatScore(item.credit_score)}
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${risk.bg} ${risk.text}`}>
                          {item.risk_category}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${decision.bg} ${decision.text}`}>
                          {item.decision}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <Link
                          to={`/history/${item.id}`}
                          className="text-indigo-600 hover:underline"
                        >
                          Details →
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <Pagination
              page={page}
              totalPages={totalPages}
              onPageChange={setPage}
            />
          )}
        </>
      )}
    </div>
  );
}
