import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchModelExplain } from '../api/model';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000/api').replace(/\/$/, '');

function DirectionBadge({ direction }) {
  if (direction === 1) {
    return (
      <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-700">
        Risk-increasing contribution
      </span>
    );
  }
  if (direction === -1) {
    return (
      <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">
        Risk-reducing contribution
      </span>
    );
  }
  return (
    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600">
      Mixed / constant
    </span>
  );
}

export default function ModelDetailPage() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchModelExplain()
      .then((data) => {
        if (active) setInfo(data);
      })
      .catch((err) => {
        if (active) setError(err.response?.data?.error || err.response?.data?.message || err.message || 'Failed to load explainability');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  if (loading) return <LoadingSpinner label="Loading explainability…" />;
  if (error) return <ErrorMessage message={error} />;
  if (!info || !Array.isArray(info.top_features) || info.top_features.length === 0) {
    return <ErrorMessage message="No explainability data available." />;
  }

  const features = info.top_features;
  const maxImportance = Math.max(...features.map((f) => Number(f.importance) || 0));
  const plots = Object.values(info.plots || {});
  const disclaimer = info.disclaimer || '';

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Model Explainability</h1>
          <p className="text-sm text-slate-500">
            SHAP feature contributions for model version <b>{info.model_version || '—'}</b>
          </p>
        </div>
        <Link
          to="/model-info"
          className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
        >
          ← Back
        </Link>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="mb-2 text-sm font-semibold text-slate-700">Top global feature importance (model importance)</h2>
        <p className="mb-4 text-xs text-slate-500">
          Features ranked by mean absolute SHAP value. Each SHAP value describes a feature's contribution to an individual prediction.
        </p>

        <ul className="space-y-3">
          {features.map((f, idx) => {
            const importance = Number(f.importance) || 0;
            const pct = maxImportance > 0 ? Math.round((importance / maxImportance) * 100) : 0;
            return (
              <li key={f.feature ?? idx} className="flex items-center gap-3">
                <span className="w-4 shrink-0 text-right text-xs font-semibold text-slate-400">
                  {idx + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-slate-800">{f.feature}</span>
                    {f.direction != null && <DirectionBadge direction={f.direction} />}
                  </div>
                  <div className="mt-1 h-2 w-full overflow-hidden rounded bg-slate-100">
                    <div
                      className="h-full rounded bg-indigo-600"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
                <span className="w-20 shrink-0 text-right text-sm tabular-nums text-slate-700">
                  {importance.toFixed(4)}
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="mb-2 text-sm font-semibold text-slate-700">Interpreting SHAP contributions</h2>
        <ul className="space-y-1.5 text-sm text-slate-600">
          <li>A positive SHAP value means the feature pushes the model toward higher default risk.</li>
          <li>A negative SHAP value means the feature pushes the model toward lower default risk.</li>
          <li>Values describe the model's prediction behavior; they are not causal.</li>
        </ul>
      </div>

      {plots.length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-white p-6">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">SHAP plots</h2>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {plots.map((src) => (
              <img
                key={src}
                src={`${API_BASE}${src}`}
                alt="SHAP explanation plot"
                className="w-full rounded border border-slate-200"
                loading="lazy"
              />
            ))}
          </div>
        </div>
      )}

      {disclaimer && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-xs text-amber-800">
          <b>Disclaimer:</b> {disclaimer}
        </div>
      )}
    </div>
  );
}
