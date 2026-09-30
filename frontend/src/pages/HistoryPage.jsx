import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchPredictions } from '../api/predictions';
import { formatCurrency, formatDate, formatPercent, formatScore, titleCase } from '../lib/format';
import { getRiskCategoryStyle } from '../lib/riskStyles';
import './HistoryPage.css';

const PAGE_SIZE = 10;
const RISK_FILTERS = [
  { value: '', label: 'All risk levels' },
  { value: 'LOW', label: 'Low risk' },
  { value: 'MODERATE', label: 'Moderate' },
  { value: 'HIGH', label: 'High risk' },
  { value: 'VERY_HIGH', label: 'Very high' },
];

function SkeletonRows() {
  return <div className="history-skeleton" aria-label="Loading prediction history">{[1, 2, 3, 4, 5].map((row) => <div key={row} className="history-skeleton-row" />)}</div>;
}

function MetricCard({ label, value, note, tone = 'blue' }) {
  return <article className="history-metric"><div className="history-metric-top"><p>{label}</p><span className={`history-metric-icon ${tone}`}>•</span></div><strong>{value}</strong><small>{note}</small></article>;
}

function RiskBadge({ category }) {
  const risk = getRiskCategoryStyle(category);
  return <span className={`history-risk ${category?.toLowerCase() || ''}`}><i />{risk.label}</span>;
}

function initials(item) {
  const value = item.person_home_ownership || item.loan_intent || `A${item.id}`;
  return String(value).replace(/[^a-z0-9]/gi, '').slice(0, 2).toUpperCase();
}

export default function HistoryPage() {
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pages, setPages] = useState(1);
  const [search, setSearch] = useState('');
  const [risk, setRisk] = useState('');
  const [intent, setIntent] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchPredictions({ page, per_page: PAGE_SIZE })
      .then((data) => {
        if (!active) return;
        setItems(Array.isArray(data) ? data : data.items || []);
        setTotal(data?.total ?? 0);
        setPages(data?.pages ?? 1);
      })
      .catch((err) => { if (active) setError(err.message || 'Failed to load history'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [page, reloadKey]);

  const intents = useMemo(() => [...new Set(items.map((item) => item.loan_intent).filter(Boolean))].sort(), [items]);
  const filteredItems = useMemo(() => items.filter((item) => {
    const haystack = `${item.id} ${item.loan_intent || ''} ${item.loan_amnt || ''} ${item.risk_category || ''}`.toLowerCase();
    const created = item.created_at ? new Date(item.created_at) : null;
    const afterStart = !dateFrom || (created && created >= new Date(`${dateFrom}T00:00:00`));
    const beforeEnd = !dateTo || (created && created <= new Date(`${dateTo}T23:59:59`));
    return haystack.includes(search.toLowerCase()) && (!risk || item.risk_category === risk) && (!intent || item.loan_intent === intent) && afterStart && beforeEnd;
  }), [items, search, risk, intent, dateFrom, dateTo]);

  const averageProbability = items.length ? items.reduce((sum, item) => sum + Number(item.default_risk_probability || 0), 0) / items.length : 0;
  const highRiskCount = items.filter((item) => ['HIGH', 'VERY_HIGH'].includes(item.risk_category)).length;
  const clearFilters = () => { setSearch(''); setRisk(''); setIntent(''); setDateFrom(''); setDateTo(''); };
  const retry = () => { setError(null); setReloadKey((current) => current + 1); };

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

  return (
    <div className="history-page">
      <div className="history-page-header"><div><p className="history-eyebrow">Ledger &amp; governance <span>•</span> {items[0]?.model_version || 'Prediction ledger'}</p><h1>Prediction History</h1><p>Auditable records of underwriting decisions and AI risk assessments.</p></div><div className="history-actions"><button type="button" onClick={() => window.print()} className="history-button secondary">Print summary</button><Link to="/assessment" className="history-button primary">New assessment</Link></div></div>

      {error && <div className="history-error"><div><strong>Unable to load prediction history.</strong><span>Please try again.</span></div><button type="button" onClick={retry}>Retry</button></div>}

      <div className="history-metrics"><MetricCard label="Total assessments" value={total} note="Persisted prediction records" /><MetricCard label="Average default risk" value={formatPercent(averageProbability)} note="Current loaded ledger page" tone="amber" /><MetricCard label="High risk assessments" value={highRiskCount} note="High and very high risk" tone="red" /><MetricCard label="Average risk score" value={formatScore(items.length ? items.reduce((sum, item) => sum + Number(item.risk_score || 0), 0) / items.length : 0, 1)} note="Model-derived score / 100" tone="green" /></div>

      <section className="history-filters" aria-label="Prediction history filters"><label className="history-search"><span>⌕</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search by ID, intent, loan amount..." aria-label="Search prediction history" /></label><label><span>Risk level</span><select value={risk} onChange={(event) => setRisk(event.target.value)} aria-label="Filter by risk level">{RISK_FILTERS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><label><span>From</span><input type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} aria-label="Filter from date" /></label><label><span>To</span><input type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} aria-label="Filter to date" /></label><label><span>Loan intent</span><select value={intent} onChange={(event) => setIntent(event.target.value)} aria-label="Filter by loan intent"><option value="">All intents</option>{intents.map((value) => <option key={value} value={value}>{titleCase(value)}</option>)}</select></label></section>

      <section className="history-table-card"><div className="history-table-heading"><div><h2>Underwriting decision ledger</h2><p>Recent model inference records and risk outcomes.</p></div><span>{filteredItems.length} shown on this page</span></div>{loading ? <SkeletonRows /> : filteredItems.length ? <div className="history-table-scroll"><table><thead><tr><th>Assessment ID</th><th>Applicant profile</th><th className="numeric">Loan amount</th><th className="numeric">Risk score</th><th className="numeric">Default probability</th><th>Risk level</th><th>Date</th><th className="action-column">Action</th></tr></thead><tbody>{filteredItems.map((item) => <tr key={item.id}><td><Link className="history-id" to={`/history/${item.id}`}>#CW-{item.id}</Link></td><td><div className="applicant-cell"><span>{initials(item)}</span><div><strong>{titleCase(item.loan_intent)} applicant</strong><small>{titleCase(item.person_home_ownership)} profile</small></div></div></td><td className="numeric">{formatCurrency(item.loan_amnt)}</td><td className="numeric"><strong>{formatScore(item.risk_score, 0)}</strong><small> / 100</small></td><td className="numeric">{formatPercent(item.default_risk_probability)}</td><td><RiskBadge category={item.risk_category} /></td><td className="history-date">{formatDate(item.created_at)}</td><td className="action-column"><Link className="view-report" to={`/history/${item.id}`}>View report <span>→</span></Link></td></tr>)}</tbody></table></div> : <div className="history-empty"><strong>{items.length ? 'No matching assessments' : 'No prediction history yet'}</strong><p>{items.length ? 'Try adjusting your search or filters.' : 'Complete your first credit assessment to populate this ledger.'}</p><button type="button" onClick={items.length ? clearFilters : undefined}>{items.length ? 'Clear filters' : 'New assessment'}</button></div>}
      <div className="history-pagination"><span>Showing {filteredItems.length ? (page - 1) * PAGE_SIZE + 1 : 0} to {Math.min((page - 1) * PAGE_SIZE + filteredItems.length, total)} of {total} assessments</span><div><button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</button><strong>{page}</strong><button type="button" disabled={page >= pages} onClick={() => setPage(page + 1)}>Next</button></div></div></section>

      <div className="history-info-grid"><article><span>◈</span><div><strong>Prediction traceability</strong><p>Each decision links to its stored input snapshot and model output.</p></div></article><article><span>⌁</span><div><strong>Model transparency</strong><p>Risk scores and probabilities remain visible alongside every record.</p></div></article><article><span>✓</span><div><strong>Decision records</strong><p>Review the underlying assessment details before acting on an outcome.</p></div></article></div>
    </div>
  );
}
