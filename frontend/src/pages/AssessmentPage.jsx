import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createPrediction } from '../api/predictions';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { formatPercent } from '../lib/format';
import './AssessmentPage.css';
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

  const resetForm = () => {
    setResult(null);
    setError(null);
    setForm(emptyForm);
  };

  return (
    <div className="assessment-page">
      <header className="assessment-page-header">
        <div>
          <p className="assessment-eyebrow">Underwriting suite <span>•</span> Production environment</p>
          <h1>New Credit Assessment</h1>
          <p>Evaluate an applicant&apos;s probability of default using the CreditWise AI risk model.</p>
        </div>
        <div className="assessment-model-card">
          <span className="assessment-model-icon">◈</span>
          <div><strong>creditwise-xgb-v1</strong><small>XGBoost production model</small></div>
        </div>
      </header>

      {error && <ErrorMessage message={error} />}

      <div className="assessment-layout">
        <form onSubmit={handleSubmit} className="assessment-form">
          <AssessmentSection number="01" title="Applicant Profile" subtitle="Primary financial, demographic and employment characteristics" tag="Demographics">
            <Field label="Applicant Age (years)" name="person_age" value={form.person_age} onChange={handleNumberField} type="number" min="18" max="100" required placeholder="e.g. 35" suffix="YRS" />
            <Field label="Annual Income (USD)" name="person_income" value={form.person_income} onChange={handleNumberField} type="number" min="0" required placeholder="e.g. 65000" prefix="$" />
            <SelectField label="Home Ownership Status" name="person_home_ownership" value={form.person_home_ownership} onChange={handleChange} options={HOME_OWNERSHIP_OPTIONS} />
            <Field label="Employment Length" name="person_emp_length" value={form.person_emp_length} onChange={handleNumberField} type="number" min="0" max="45" placeholder="e.g. 7" suffix="Years" />
          </AssessmentSection>

          <AssessmentSection number="02" title="Loan Specifications" subtitle="Requested credit facility terms and stated financing intent" tag="Exposure">
            <Field label="Loan Amount (USD)" name="loan_amnt" value={form.loan_amnt} onChange={handleNumberField} type="number" min="0" required placeholder="e.g. 12000" prefix="$" />
            <SelectField label="Loan Intent Category" name="loan_intent" value={form.loan_intent} onChange={handleChange} options={LOAN_INTENT_OPTIONS} />
          </AssessmentSection>

          <AssessmentSection number="03" title="Bureau & Credit History" subtitle="Historical bureau delinquency indicators and trade line seasoning" tag="Bureau">
            <SelectField label="Default on File (credit bureau)" name="cb_person_default_on_file" value={form.cb_person_default_on_file} onChange={handleChange} options={DEFAULT_ON_FILE_OPTIONS} />
            <Field label="Credit History Length (years)" name="cb_person_cred_hist_length" value={form.cb_person_cred_hist_length} onChange={handleNumberField} type="number" min="0" max="40" placeholder="e.g. 5" suffix="Years" />
          </AssessmentSection>

          <div className="assessment-actions">
            <button type="submit" disabled={submitting} className="assessment-primary-button">
              {submitting ? <><span className="assessment-spinner" /> Running Risk Assessment...</> : <>Run Risk Assessment <span>→</span></>}
            </button>
            <button type="button" onClick={resetForm} className="assessment-secondary-button">Reset Form</button>
            <span className="assessment-action-note">Inference via creditwise-xgb-v1</span>
          </div>
        </form>

        <ResultPanel result={result} submitting={submitting} onReset={resetForm} />
      </div>

      <div className="assessment-info-grid">
        <InfoCard icon="▣" title="Feature Coverage" value={`${Object.keys(emptyForm).length} application inputs`} />
        <InfoCard icon="◌" title="Model Version" value={result?.model_version || 'creditwise-xgb-v1'} />
        <InfoCard icon="✓" title="Inference Status" value={submitting ? 'Processing request' : result ? 'Assessment complete' : 'Ready for assessment'} />
      </div>
    </div>
  );
}

function AssessmentSection({ number, title, subtitle, tag, children }) {
  return <section className="assessment-section"><div className="assessment-section-heading"><span className="assessment-section-number">{number}</span><div><h2>{title}</h2><p>{subtitle}</p></div><span className="assessment-section-tag">{tag}</span></div><div className="assessment-fields">{children}</div></section>;
}

function Field({ label, name, value, onChange, required, prefix, suffix, ...props }) {
  return <label className="assessment-field"><span>{label} {required && <b>*</b>}</span><div className="assessment-input-wrap">{prefix && <i>{prefix}</i>}<input id={name} name={name} value={value} onChange={onChange} required={required} {...props} />{suffix && <em>{suffix}</em>}</div></label>;
}

function SelectField({ label, name, value, onChange, options, required }) {
  return <label className="assessment-field"><span>{label} {required && <b>*</b>}</span><select id={name} name={name} value={value} onChange={onChange} required={required}>{options.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>;
}

function InfoCard({ icon, title, value }) {
  return <div className="assessment-info-card"><span>{icon}</span><div><small>{title}</small><strong>{value}</strong></div></div>;
}

function ResultPanel({ result, submitting, onReset }) {
  if (submitting) return <aside className="assessment-result-panel assessment-result-loading"><div className="result-empty-icon">◌</div><h2>Running assessment</h2><p>The production model is evaluating the submitted application.</p><span className="assessment-spinner" /></aside>;
  if (!result) return <aside className="assessment-result-panel"><div className="result-empty-icon">◈</div><p className="assessment-eyebrow">Assessment result</p><h2>Ready for Assessment</h2><p>Complete the applicant information and run the risk assessment to generate model results.</p><div className="result-empty-line"><span />Awaiting application data</div></aside>;

  const riskStyle = getRiskCategoryStyle(result.risk_category);
  const decisionStyle = getDecisionStyle(result.decision);
  const probability = Number(result.default_risk_probability || 0);
  const riskPosition = Math.max(0, Math.min(100, Number(result.risk_score || probability * 100)));
  return (
    <aside className="assessment-result-panel"><div className="result-heading"><div><p className="assessment-eyebrow">Assessment result</p><h2>Risk assessment complete</h2><small>Execution timestamp: {new Date().toLocaleString()}</small></div><span className={`result-badge ${riskStyle.bg} ${riskStyle.text}`}>{result.risk_category}</span></div><div className="result-metrics"><div><small>Risk score</small><strong>{result.risk_score}<i>/100</i></strong></div><div><small>Default prob. (PD)</small><strong>{formatPercent(result.default_risk_probability, 2)}</strong></div></div><div className="risk-meter"><div className="risk-meter-labels"><span>Low (0-40)</span><span>Moderate (41-75)</span><span>High (76-100)</span></div><div className="risk-meter-bar"><span className="low" /><span className="moderate" /><span className="high" /><b style={{ left: `${riskPosition}%` }} /></div></div><div className={`decision-block ${decisionStyle.bg}`}><strong>{result.decision}</strong><p>{result.message || 'Outcome generated from the production risk model.'}</p></div><div className="result-factors"><div className="result-subheading"><strong>Assessment factors</strong><small>Observed inputs</small></div><Factor label="Application data validated" value="Complete" positive /><Factor label="Classification threshold" value={result.classification_threshold ?? '—'} /><Factor label="Model version" value={result.model_version || '—'} /></div><div className="result-trace"><strong>Data verification trace</strong><p><span>Prediction API</span><b>Successful</b></p><p><span>Input validation</span><b>Passed</b></p><p><span>Inference model</span><b>{result.model_version || 'Production'}</b></p></div><button type="button" onClick={onReset} className="assessment-secondary-button result-reset">Start New Assessment</button></aside>
  );
}

function Factor({ label, value, positive }) { return <div className="factor-row"><span className={positive ? 'factor-positive' : ''}>{positive ? '✓' : '•'} {label}</span><b>{value}</b></div>; }
