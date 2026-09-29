import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { fetchHistoryItem } from '../api/predictions';
import { useAuth } from '../auth/AuthContext';
import { formatDate, formatCurrency, formatPercent, formatScore } from '../lib/format';
import { getRiskCategoryStyle, getDecisionStyle } from '../lib/riskStyles';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';
import StatCard from '../components/common/StatCard';

function Field({ label, value }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</span>
      <span className="text-sm font-semibold text-slate-800">{value}</span>
    </div>
  );
}

export default function HistoryDetailPage() {
  const { id } = useParams();
  const { user } = useAuth();
  const [item, setItem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchHistoryItem(id)
      .then((data) => {
        if (active) setItem(data);
      })
      .catch((err) => {
        if (active) setError(err.message || 'Failed to load prediction');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [id]);

  if (loading) return <LoadingSpinner label="Loading prediction…" />;
  if (error) return <ErrorMessage message={error} />;
  if (!item) return <ErrorMessage message="Prediction not found" />;

  const risk = getRiskCategoryStyle(item.risk_category);
  const decision = getDecisionStyle(item.decision);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Assessment #{item.id}</h1>
          <p className="text-sm text-slate-500">
            {formatDate(item.created_at)} · {user?.email}
          </p>
        </div>
        <Link
          to="/history"
          className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
        >
          ← Back
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          label="Risk Category"
          value={item.risk_category}
          tone={risk.tone}
        />
        <StatCard
          label="Decision"
          value={item.decision}
          tone={decision.tone}
        />
        <StatCard
          label="Default Probability"
          value={formatPercent(item.default_probability)}
          hint={`Score ${formatScore(item.credit_score)}/100`}
        />
      </div>

      {item.input && (
        <div className="rounded-lg border border-slate-200 bg-white p-6">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Input Snapshot</h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            <StatCard label="Age" value={item.input.person_age ?? '—'} />
            <StatCard label="Annual Income" value={item.input.person_income ? formatCurrency(item.input.person_income) : '—'} />
            <StatCard label="Home Ownership" value={item.input.person_home_ownership ?? '—'} />
            <StatCard label="Employment (yrs)" value={item.input.person_emp_length ?? '—'} />
            <StatCard label="Loan Intent" value={item.input.loan_intent ?? '—'} />
            <StatCard label="Loan Grade" value={item.input.loan_grade ?? '—'} />
            <StatCard label="Loan Amount" value={item.input.loan_amnt ? formatCurrency(item.input.loan_amnt) : '—'} />
            <StatCard label="Interest Rate" value={item.input.loan_int_rate ?? '—'} />
            <StatCard label="Income Share" value={item.input.loan_percent_income ?? '—'} />
            <StatCard label="Default on File" value={item.input.cb_person_default_on_file ?? '—'} />
            <StatCard label="Credit Hist (yrs)" value={item.input.cb_person_cred_hist_length ?? '—'} />
          </div>
        </div>
      )}

      {item.shap && item.shap.length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-white p-6">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Top Factors (SHAP)</h2>
          <ul className="space-y-3">
            {item.shap.map((s) => (
              <li key={s.feature} className="flex items-center justify-between">
                <span className="text-sm text-slate-700">{formatFeatureName(s.feature)}</span>
                <span className="text-sm font-semibold text-slate-900">
                  {s.value > 0 ? '+' : ''}{Number(s.value).toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate-400">
            Positive values push the prediction toward higher default probability.
          </p>
        </div>
      )}
    </div>
  );
}

function formatFeatureName(feature) {
  return feature
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
