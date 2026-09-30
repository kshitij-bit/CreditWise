import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { fetchHistoryItem } from '../api/predictions';
import { formatCurrency, formatDate, formatPercent, formatScore, titleCase } from '../lib/format';
import { getDecisionStyle, getRiskCategoryStyle } from '../lib/riskStyles';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';

function DetailCard({ title, eyebrow, children }) {
  return (
    <section className="rounded-xl border border-[#dfe3f0] bg-white p-5 shadow-sm">
      {eyebrow && <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#2457d6]">{eyebrow}</p>}
      <h2 className="mt-1 text-base font-semibold text-[#181c20]">{title}</h2>
      {children}
    </section>
  );
}

function DetailField({ label, value }) {
  return (
    <div className="rounded-lg bg-[#f1f4f9] px-3 py-2.5">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-[#747686]">{label}</p>
      <p className="mt-1 text-sm font-semibold text-[#181c20]">{value ?? 'Not provided'}</p>
    </div>
  );
}

function buildIndicators(item) {
  const indicators = [];
  const income = Number(item.person_income);
  const loan = Number(item.loan_amnt);
  const loanToIncome = income > 0 ? loan / income : null;

  if (loanToIncome !== null) {
    indicators.push({
      label: 'Loan-to-income context',
      value: `${(loanToIncome * 100).toFixed(1)}%`,
      detail: 'Calculated from the submitted loan amount and annual income.',
      tone: loanToIncome > 0.3 ? 'risk' : 'positive',
    });
  }
  if (item.cb_person_default_on_file === 'Y') {
    indicators.push({ label: 'Prior default recorded', value: 'Yes', detail: 'The submitted bureau flag indicates a prior default.', tone: 'risk' });
  } else {
    indicators.push({ label: 'Prior default recorded', value: 'No', detail: 'The submitted bureau flag does not indicate a prior default.', tone: 'positive' });
  }
  if (item.person_emp_length != null) {
    indicators.push({ label: 'Employment history', value: `${item.person_emp_length} years`, detail: 'Employment length supplied in the application.', tone: Number(item.person_emp_length) >= 5 ? 'positive' : 'neutral' });
  }
  if (item.cb_person_cred_hist_length != null) {
    indicators.push({ label: 'Credit history length', value: `${item.cb_person_cred_hist_length} years`, detail: 'Credit history length supplied in the application.', tone: Number(item.cb_person_cred_hist_length) >= 5 ? 'positive' : 'neutral' });
  }
  return indicators;
}

function RiskSpectrum({ category }) {
  const categories = [
    { key: 'LOW', label: 'Low', color: '#12b76a' },
    { key: 'MODERATE', label: 'Moderate', color: '#f79009' },
    { key: 'HIGH', label: 'High', color: '#f04438' },
    { key: 'VERY_HIGH', label: 'Very high', color: '#ba1a1a' },
  ];
  const activeIndex = Math.max(0, categories.findIndex((entry) => entry.key === category));

  return (
    <div className="mt-5 rounded-lg border border-[#e7e9f1] bg-[#fbfcff] p-3">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wide text-[#747686]">
        <span>Risk spectrum</span><span>{categories[activeIndex].label} band</span>
      </div>
      <div className="mt-2 grid grid-cols-4 gap-1" aria-label={`Risk spectrum: ${categories[activeIndex].label}`}>
        {categories.map((entry, index) => <div key={entry.key} className="relative"><div className="h-2" style={{ backgroundColor: entry.color, opacity: index === activeIndex ? 1 : 0.25 }} />{index === activeIndex && <span className="absolute left-1/2 top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow" style={{ backgroundColor: entry.color }} />}</div>)}
      </div>
      <div className="mt-2 grid grid-cols-4 text-[10px] text-[#747686]"><span>Low</span><span className="text-center">Moderate</span><span className="text-center">High</span><span className="text-right">Very high</span></div>
    </div>
  );
}

export default function HistoryDetailPage() {
  const { id } = useParams();
  const [item, setItem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchHistoryItem(id)
      .then((data) => { if (active) setItem(data); })
      .catch((err) => { if (active) setError(err.response?.data?.error || err.message || 'Failed to load prediction'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [id]);

  const indicators = useMemo(() => (item ? buildIndicators(item) : []), [item]);

  if (loading) return <LoadingSpinner label="Loading assessment report..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!item) return <ErrorMessage message="Prediction not found" />;

  const risk = getRiskCategoryStyle(item.risk_category);
  const decision = getDecisionStyle(item.decision);
  const probability = Number(item.default_risk_probability || 0);
  const threshold = Number(item.classification_threshold || 0);
  const decisionReason = probability >= threshold
    ? `The default probability of ${formatPercent(probability, 2)} is at or above the model classification threshold of ${formatPercent(threshold, 1)}.`
    : `The default probability of ${formatPercent(probability, 2)} is below the model classification threshold of ${formatPercent(threshold, 1)}.`;

  return (
    <div className="report-page mx-auto max-w-6xl space-y-5">
      <div className="report-header flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-2 text-xs text-[#747686]"><Link to="/history" className="hover:text-[#2457d6]">Prediction History</Link><span className="mx-2">/</span><span>Assessment Report #{item.id}</span></p>
          <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#2457d6]">Assessment report · #{item.id}</p>
          <h1 className="mt-1 text-2xl font-bold text-[#181c20]">Underwriting Decision Report</h1>
          <p className="mt-1 text-sm text-[#565e71]">Generated {formatDate(item.created_at)} from the stored prediction record.</p>
        </div>
        <div className="flex gap-2">
          <button type="button" onClick={() => window.print()} className="rounded-md border border-[#c3c5d7] bg-white px-3 py-2 text-sm font-medium text-[#565e71] hover:bg-[#f1f4f9]">Print report</button>
          <Link to="/history" className="rounded-md bg-[#2457d6] px-3 py-2 text-sm font-semibold text-white hover:bg-[#003fb3]">Back to history</Link>
        </div>
      </div>

      <section className="report-section grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-xl border border-[#dfe3f0] bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div><p className="text-xs font-semibold uppercase tracking-wide text-[#747686]">Final model outcome</p><p className="mt-1 text-lg font-bold text-[#181c20]">{decision.label}</p></div>
            <span className={`rounded-full px-3 py-1 text-xs font-bold ${risk.badge}`}>{risk.label}</span>
          </div>
          <p className="mt-4 text-sm leading-6 text-[#565e71]">{decisionReason} The resulting risk band is <strong>{risk.label}</strong>, and the recorded recommendation is <strong>{decision.label}</strong>.</p>
          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg bg-[#f1f4f9] p-3"><p className="text-[10px] font-semibold uppercase text-[#747686]">Risk score</p><p className="mt-1 text-2xl font-bold text-[#181c20]">{formatScore(item.risk_score, 0)}<span className="text-sm font-medium text-[#747686]"> / 100</span></p></div>
            <div className="rounded-lg bg-[#f1f4f9] p-3"><p className="text-[10px] font-semibold uppercase text-[#747686]">Default probability</p><p className="mt-1 text-2xl font-bold text-[#181c20]">{formatPercent(probability, 2)}</p></div>
            <div className="rounded-lg bg-[#f1f4f9] p-3"><p className="text-[10px] font-semibold uppercase text-[#747686]">Threshold</p><p className="mt-1 text-2xl font-bold text-[#181c20]">{formatPercent(threshold, 1)}</p></div>
          </div>
          <RiskSpectrum category={item.risk_category} />
        </div>
        <DetailCard title="Model execution" eyebrow="Traceability">
          <div className="mt-4 grid gap-3 text-sm"><DetailField label="Model version" value={item.model_version} /><DetailField label="Inference status" value="Completed and persisted" /><DetailField label="Record timestamp" value={formatDate(item.created_at)} /></div>
        </DetailCard>
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <DetailCard title="Applicant and loan snapshot" eyebrow="Inputs used by the model">
          <div className="mt-4 grid grid-cols-2 gap-3"><DetailField label="Age" value={item.person_age} /><DetailField label="Annual income" value={formatCurrency(item.person_income)} /><DetailField label="Home ownership" value={titleCase(item.person_home_ownership)} /><DetailField label="Employment length" value={item.person_emp_length == null ? 'Not provided' : `${item.person_emp_length} years`} /><DetailField label="Loan amount" value={formatCurrency(item.loan_amnt)} /><DetailField label="Loan intent" value={titleCase(item.loan_intent)} /><DetailField label="Prior default" value={item.cb_person_default_on_file === 'Y' ? 'Yes' : 'No'} /><DetailField label="Credit history" value={`${item.cb_person_cred_hist_length} years`} /></div>
        </DetailCard>
        <DetailCard title="Decision context" eyebrow="Transparent input indicators">
          <p className="mt-3 text-xs leading-5 text-[#747686]">These indicators are calculated from the stored application fields. They explain the record in context; they are not separate model attribution values.</p>
          <div className="mt-4 space-y-2">{indicators.map((indicator) => <div key={indicator.label} className="flex items-start justify-between gap-4 rounded-lg border border-[#e7e9f1] px-3 py-2.5"><div><p className="text-sm font-semibold text-[#181c20]">{indicator.label}</p><p className="text-xs text-[#747686]">{indicator.detail}</p></div><span className={`whitespace-nowrap text-sm font-bold ${indicator.tone === 'risk' ? 'text-[#ba1a1a]' : indicator.tone === 'positive' ? 'text-[#087443]' : 'text-[#565e71]'}`}>{indicator.value}</span></div>)}</div>
        </DetailCard>
      </div>

      <section className="report-section rounded-xl border border-[#dfe3f0] bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-end justify-between gap-2"><div><p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#2457d6]">Decision audit</p><h2 className="mt-1 text-base font-semibold text-[#181c20]">Formal decision record</h2></div><span className="text-xs font-semibold text-[#087443]">✓ Persisted</span></div>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6"><DetailField label="Assessment ID" value={`#${item.id}`} /><DetailField label="Recommendation" value={decision.label} /><DetailField label="Risk classification" value={risk.label} /><DetailField label="Model cutoff" value={formatPercent(threshold, 1)} /><DetailField label="Inferred probability" value={formatPercent(probability, 2)} /><DetailField label="Verification" value="Stored record" /></div>
      </section>

      <DetailCard title="How to interpret this report" eyebrow="Model decision guidance">
        <div className="mt-3 grid gap-3 text-sm leading-6 text-[#565e71] md:grid-cols-3"><p><strong className="text-[#181c20]">Probability:</strong> the model-estimated likelihood of default for the submitted profile.</p><p><strong className="text-[#181c20]">Risk score:</strong> CreditWise's model-derived 0-100 risk indicator, not a bureau score.</p><p><strong className="text-[#181c20]">Decision:</strong> the recorded recommendation generated using the probability threshold and risk engine.</p></div>
      </DetailCard>
      <footer className="report-footer flex flex-wrap items-center justify-between gap-2 border-t border-[#dfe3f0] px-1 pt-4 text-xs text-[#747686]"><span><strong className="text-[#565e71]">CreditWise AI</strong> · Credit Risk Platform</span><span>Assessment #{item.id} · {item.model_version} · {formatDate(item.created_at)}</span></footer>
    </div>
  );
}