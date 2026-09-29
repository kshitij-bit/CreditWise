import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { fetchAnalytics } from '../api/predictions';
import { formatScore, formatPercent } from '../lib/format';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import StatCard from '../components/common/StatCard';

export default function DashboardPage() {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchAnalytics()
      .then((data) => {
        if (active) setAnalytics(data);
      })
      .catch((err) => {
        if (active) setError(err.message || 'Failed to load analytics');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const greeting = `Good ${new Date().getHours() < 12 ? 'morning' : new Date().getHours() < 17 ? 'afternoon' : 'evening'}`;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          {greeting}, {user?.name?.split(' ')[0] || 'there'} 👋
        </h1>
        <p className="text-sm text-slate-500">
          Credit risk assessment platform — predict default probability from applicant data.
        </p>
      </div>

      {error && <ErrorMessage message={error} />}

      {loading ? (
        <LoadingSpinner label="Loading analytics…" />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Predictions"
            value={analytics?.total_predictions ?? 0}
            tone="info"
          />
          <StatCard
            label="Approval Rate"
            value={formatPercent(analytics?.approval_rate)}
            hint="Share of predictions approved"
            tone="success"
          />
          <StatCard
            label="Avg Risk Score"
            value={formatScore(analytics?.avg_score)}
            hint="0–100, higher = riskier"
            tone="warning"
          />
          <StatCard
            label="Default Rate"
            value={formatPercent(analytics?.default_rate)}
            tone="danger"
          />
        </div>
      )}

      <div className="rounded-lg border border-slate-200 bg-white p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Quick Start</h2>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Link
            to="/assessment"
            className="rounded-md border border-slate-200 bg-white p-4 transition hover:border-indigo-400 hover:shadow-sm"
          >
            <span className="text-2xl">🔍</span>
            <p className="mt-2 font-medium text-slate-900">New Assessment</p>
            <p className="text-sm text-slate-500">
              Enter applicant data to predict credit risk
            </p>
          </Link>
          <Link
            to="/history"
            className="rounded-md border border-slate-200 bg-white p-4 transition hover:border-indigo-400 hover:shadow-sm"
          >
            <span className="text-2xl">🕘</span>
            <p className="mt-2 font-medium text-slate-900">Prediction History</p>
            <p className="text-sm text-slate-500">Review past assessments and decisions</p>
          </Link>
          <Link
            to="/model-info"
            className="rounded-md border border-slate-200 bg-white p-4 transition hover:border-indigo-400 hover:shadow-sm"
          >
            <span className="text-2xl">🧠</span>
            <p className="mt-2 font-medium text-slate-900">Model Info</p>
            <p className="text-sm text-slate-500">Understand the XGBoost model &amp; SHAP</p>
          </Link>
        </div>
      </div>
    </div>
  );
}
