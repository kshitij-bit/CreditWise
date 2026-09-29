import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createPrediction } from '../api/predictions';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';
import StatCard from '../components/common/StatCard';
import { formatPercent } from '../lib/format';
import {
  HOME_OWNERSHIP_OPTIONS,
  LOAN_INTENT_OPTIONS,
  DEFAULT_ON_FILE_OPTIONS,
  getRiskCategoryStyle,
  getDecisionStyle,
} from '../lib/riskStyles';

const emptyForm = {
  person_age: '',
  person_income: '',
  person_home_ownership: 'RENT',
  person_emp_length: '',
  loan_intent: 'PERSONAL',
  loan_amnt: '',
  cb_person_default_on_file: 'N',
  cb_person_cred_hist_length: '',
};

export default function AssessmentPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState(emptyForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setForm((prev) => ({ ...prev, [name]: type === 'number' ? value : value }));
  };

  const handleNumberField = (e) => {
    const { name, value } = e.target;
    if (value === '' || !Number.isNaN(Number(value))) {
      setForm((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const payload = {
        person_age: Number(form.person_age),
        person_income: Number(form.person_income),
        person_home_ownership: form.person_home_ownership,
        person_emp_length: form.person_emp_length === '' ? null : Number(form.person_emp_length),
        loan_intent: form.loan_intent,
        loan_amnt: Number(form.loan_amnt),
        cb_person_default_on_file: form.cb_person_default_on_file,
        cb_person_cred_hist_length: Number(form.cb_person_cred_hist_length),
      };
      const data = await createPrediction(payload);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Prediction failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">New Assessment</h1>
        <p className="text-sm text-slate-500">
          Enter applicant information to predict default probability.
        </p>
      </div>

      {error && <ErrorMessage message={error} />}

      {result ? (
        <ResultPanel result={result} onReset={() => { setResult(null); setForm(emptyForm); }} />
      ) : (
        <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-6 rounded-xl border border-slate-200 bg-white p-6 lg:grid-cols-2">
          <section className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-700">Personal</h2>

            <div>
              <label htmlFor="person_age" className="mb-1 block text-sm font-medium text-slate-700">
                Age <span className="text-red-500">*</span>
              </label>
              <input
                id="person_age"
                name="person_age"
                type="number"
                min="18"
                max="100"
                required
                value={form.person_age}
                onChange={handleNumberField}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                placeholder="e.g. 35"
              />
            </div>

            <div>
              <label htmlFor="person_income" className="mb-1 block text-sm font-medium text-slate-700">
                Annual Income (USD) <span className="text-red-500">*</span>
              </label>
              <input
                id="person_income"
                name="person_income"
                type="number"
                min="0"
                required
                value={form.person_income}
                onChange={handleNumberField}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                placeholder="e.g. 65000"
              />
            </div>

            <div>
              <label htmlFor="person_home_ownership" className="mb-1 block text-sm font-medium text-slate-700">
                Home Ownership
              </label>
              <select
                id="person_home_ownership"
                name="person_home_ownership"
                value={form.person_home_ownership}
                onChange={handleChange}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              >
                {HOME_OWNERSHIP_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="person_emp_length" className="mb-1 block text-sm font-medium text-slate-700">
                Employment Length (years)
              </label>
              <input
                id="person_emp_length"
                name="person_emp_length"
                type="number"
                min="0"
                max="45"
                value={form.person_emp_length}
                onChange={handleNumberField}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                placeholder="e.g. 7"
              />
            </div>
          </section>

          <section className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-700">Loan</h2>

            <div>
              <label htmlFor="loan_amnt" className="mb-1 block text-sm font-medium text-slate-700">
                Loan Amount (USD) <span className="text-red-500">*</span>
              </label>
              <input
                id="loan_amnt"
                name="loan_amnt"
                type="number"
                min="0"
                required
                value={form.loan_amnt}
                onChange={handleNumberField}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                placeholder="e.g. 12000"
              />
            </div>

            <div>
              <label htmlFor="loan_intent" className="mb-1 block text-sm font-medium text-slate-700">
                Loan Intent
              </label>
              <select
                id="loan_intent"
                name="loan_intent"
                value={form.loan_intent}
                onChange={handleChange}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              >
                {LOAN_INTENT_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>

            </section>

          <section className="space-y-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-slate-700">Credit History</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="cb_person_default_on_file" className="mb-1 block text-sm font-medium text-slate-700">
                  Default on File (credit bureau)
                </label>
                <select
                  id="cb_person_default_on_file"
                  name="cb_person_default_on_file"
                  value={form.cb_person_default_on_file}
                  onChange={handleChange}
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                >
                  {DEFAULT_ON_FILE_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="cb_person_cred_hist_length" className="mb-1 block text-sm font-medium text-slate-700">
                  Credit History Length (years)
                </label>
                <input
                  id="cb_person_cred_hist_length"
                  name="cb_person_cred_hist_length"
                  type="number"
                  min="0"
                  max="40"
                  value={form.cb_person_cred_hist_length}
                  onChange={handleNumberField}
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                  placeholder="e.g. 5"
                />
              </div>
            </div>
          </section>

          <div className="lg:col-span-2">
            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-50 lg:w-auto lg:px-8"
            >
              {submitting ? 'Predicting…' : 'Run Prediction'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

function ResultPanel({ result, onReset }) {
  const riskStyle = getRiskCategoryStyle(result.risk_category);
  const decisionStyle = getDecisionStyle(result.decision);
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${riskStyle.bg} ${riskStyle.text}`}>
          {result.risk_category}
        </span>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${decisionStyle.bg} ${decisionStyle.text}`}>
          {result.decision}
        </span>
        <span className="ml-auto text-sm text-slate-500">
          Risk Score: <b>{result.risk_score}</b>/100
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Default-Risk Probability" value={formatPercent(result.default_risk_probability, 2)} />
        <StatCard label="CreditWise Risk Score" value={`${result.risk_score} / 100`} hint="0–100, higher = greater predicted default risk" />
        <StatCard label="Model Version" value={result.model_version || '—'} />
      </div>

      {result.message && (
        <p className="mt-4 text-xs text-slate-400">{result.message}</p>
      )}

      {result.classification_threshold != null && (
        <p className="mt-1 text-xs text-slate-400">
          Classification Threshold: {result.classification_threshold}
        </p>
      )}

      <div className="mt-6 flex gap-3">
        <button
          onClick={onReset}
          className="rounded-md bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-200"
        >
          ↺ New Assessment
        </button>
      </div>
    </div>
  );
}
