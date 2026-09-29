import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchModelInfo } from '../api/model';
import { formatPercent, formatScore } from '../lib/format';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import StatCard from '../components/common/StatCard';

export default function ModelInfoPage() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchModelInfo()
      .then((data) => {
        if (active) setInfo(data);
      })
      .catch((err) => {
        if (active) setError(err.response?.data?.error || err.response?.data?.message || err.message || 'Failed to load model info');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  if (loading) return <LoadingSpinner label="Loading model info…" />;
  if (error) return <ErrorMessage message={error} />;

  const cv = info?.cross_validation;
  const metrics = info?.evaluation_metrics;
  const classMetrics = metrics?.classification_metrics;
  const ranking = metrics?.ranking_metrics;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Model Information</h1>
        <p className="text-sm text-slate-500">CreditWise AI credit risk model</p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Model Version" value={info?.model_version || '—'} hint={info?.model_name} />
        <StatCard label="Algorithm" value={info?.algorithm || '—'} hint={info?.feature_configuration} />
        <StatCard label="Classification Threshold" value={info?.classification_threshold ?? '—'} hint={info?.probability_representation} />
      </div>

      {metrics && (
        <div className="rounded-lg border border-slate-200 bg-white p-6">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">Holdout Test Metrics{info?.evaluation_source ? ` — ${info.evaluation_source}` : ''}</h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            <StatCard label="ROC-AUC" value={ranking?.roc_auc != null ? formatScore(ranking.roc_auc, 4) : '—'} />
            <StatCard label="PR-AUC" value={ranking?.pr_auc != null ? formatScore(ranking.pr_auc, 4) : '—'} />
            <StatCard label="Precision" value={classMetrics?.precision != null ? formatPercent(classMetrics.precision, 2) : '—'} />
            <StatCard label="Recall" value={classMetrics?.recall != null ? formatPercent(classMetrics.recall, 2) : '—'} />
            <StatCard label="F1 Score" value={classMetrics?.f1 != null ? formatPercent(classMetrics.f1, 2) : '—'} />
          </div>
        </div>
      )}

      {cv && (
        <div className="rounded-lg border border-slate-200 bg-white p-6">
          <h2 className="mb-4 text-sm font-semibold text-slate-700">5-Fold Cross-Validation</h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Mean ROC-AUC" value={formatScore(cv.mean_roc_auc)} />
            <StatCard label="± Std" value={formatScore(cv.std_roc_auc)} hint="across folds" />
            <StatCard label="Mean Accuracy" value={formatPercent(cv.mean_accuracy)} />
            <StatCard label="± Std" value={formatPercent(cv.std_accuracy)} hint="across folds" />
          </div>
        </div>
      )}

      <div className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="mb-4 text-sm font-semibold text-slate-700">How Predictions Work</h2>
        <ol className="list-decimal space-y-2 pl-5 text-sm text-slate-600">
          <li>Your assessment inputs are scaled using the featurizer saved with the model.</li>
          <li>The gradient-boosted model predicts a raw default probability.</li>
          <li>The probability is converted to a 0–100 credit score and a risk band (Low/Moderate/High/Very High).</li>
          <li>A transparent rule maps score + risk band to a decision: Approve / Review / Reject.</li>
          <li>SHAP values explain which features pushed the prediction up or down.</li>
        </ol>
        <Link
          to="/model-detail"
          className="mt-4 inline-block text-sm font-semibold text-indigo-600 hover:underline"
        >
          View feature importance &amp; SHAP →
        </Link>
      </div>
    </div>
  );
}
