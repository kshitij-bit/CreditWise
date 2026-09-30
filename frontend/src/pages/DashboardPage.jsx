import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { fetchAnalytics, fetchPredictions } from '../api/predictions';
import { formatCurrency, formatDate, formatPercent, formatScore, titleCase } from '../lib/format';
import { getDecisionStyle, getRiskCategoryStyle } from '../lib/riskStyles';
import './DashboardPage.css';

const PAGE_SIZE = 5;
const riskColors = { LOW: '#12b76a', MODERATE: '#f79009', HIGH: '#f04438', VERY_HIGH: '#b42318' };

function Skeleton({ className = '' }) {
  return <div className={`animate-pulse rounded bg-[#edf0f8] ${className}`} />;
}

function Kpi({ label, value, note, tone = 'blue' }) {
  return (
    <article className="dashboard-card p-4">
      <div className="flex items-start justify-between"><p className="dashboard-label">{label}</p><span className={`kpi-icon ${tone}`}>•</span></div>
      <p className="mt-3 text-2xl font-bold tracking-tight text-[#181c20]">{value}</p>
      <p className="mt-1 text-[11px] text-[#6d7486]">{note}</p>
    </article>
  );
}

function RiskDistribution({ analytics }) {
  const counts = analytics?.risk_category_counts || {};
  const total = analytics?.total_predictions || 0;
  const data = Object.entries(counts).filter(([, value]) => value > 0).map(([name, value]) => ({ name, value }));
  const chartData = data.length ? data : [{ name: 'No data', value: 1 }];

  return (
    <section className="dashboard-card p-5">
      <div><h2 className="dashboard-section-title">Risk Distribution</h2><p className="dashboard-subtitle">Portfolio exposure classification</p></div>
      <div className="mt-4 flex items-center gap-4">
        <div className="h-36 w-36 shrink-0"><ResponsiveContainer><PieChart><Pie data={chartData} innerRadius={43} outerRadius={64} paddingAngle={3} dataKey="value" stroke="none">{chartData.map((entry) => <Cell key={entry.name} fill={riskColors[entry.name] || '#dfe3f0'} />)}</Pie><Tooltip formatter={(value) => [value, 'Assessments']} /></PieChart></ResponsiveContainer></div>
        <div className="min-w-0 flex-1 space-y-3">{Object.keys(riskColors).map((key) => { const count = counts[key] || 0; return <div key={key} className="flex items-center justify-between gap-2 text-xs"><span className="flex items-center gap-2 text-[#434654]"><i className="h-2 w-2 rounded-full" style={{ backgroundColor: riskColors[key] }} />{titleCase(key)}</span><span className="font-semibold text-[#181c20]">{total ? `${((count / total) * 100).toFixed(1)}%` : '0%'}</span></div>; })}</div>
      </div>
      <div className="mt-5"><div className="mb-2 flex justify-between text-[10px] text-[#6d7486]"><span>Portfolio health ratio</span><span>{total ? `${(((counts.LOW || 0) / total) * 100).toFixed(1)}% low risk` : 'No assessments'}</span></div><div className="flex h-2 overflow-hidden rounded-full bg-[#edf0f8]"><div className="bg-[#12b76a]" style={{ width: `${total ? ((counts.LOW || 0) / total) * 100 : 0}%` }} /><div className="bg-[#f79009]" style={{ width: `${total ? ((counts.MODERATE || 0) / total) * 100 : 0}%` }} /><div className="bg-[#f04438]" style={{ width: `${total ? (((counts.HIGH || 0) + (counts.VERY_HIGH || 0)) / total) * 100 : 0}%` }} /></div></div>
    </section>
  );
}

function RecentAssessments({ items, total, page, pages, search, onSearch, onPageChange }) {
  const filtered = items.filter((item) => `${item.id} ${item.loan_intent} ${item.risk_category} ${item.decision}`.toLowerCase().includes(search.toLowerCase()));
  return (
    <section className="dashboard-card overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-[#e7e9f1] p-5 sm:flex-row sm:items-center sm:justify-between"><div><h2 className="dashboard-section-title">Recent Credit Assessments</h2><p className="dashboard-subtitle">Real-time inference queue and algorithmic outcomes</p></div><input value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Search assessments" className="dashboard-input w-full sm:w-52" /></div>
      {filtered.length ? <div className="overflow-x-auto"><table className="min-w-[720px] w-full text-left text-xs"><thead className="bg-[#f7f9ff] text-[10px] uppercase tracking-[0.08em] text-[#6d7486]"><tr><th className="px-5 py-3">Applicant / ID</th><th className="px-3 py-3">Loan amount</th><th className="px-3 py-3">Default probability</th><th className="px-3 py-3">Score</th><th className="px-3 py-3">Risk level</th><th className="px-3 py-3">Decision</th><th className="px-5 py-3 text-right">View</th></tr></thead><tbody className="divide-y divide-[#edf0f8]">{filtered.map((item) => { const risk = getRiskCategoryStyle(item.risk_category); const decision = getDecisionStyle(item.decision); return <tr key={item.id} className="text-[#434654] hover:bg-[#fbfcff]"><td className="px-5 py-3"><div className="flex items-center gap-2"><span className="flex h-7 w-7 items-center justify-center rounded-full bg-[#e7edff] text-[10px] font-bold text-[#2457d6]">{String(item.id).slice(-2)}</span><div><p className="font-semibold text-[#181c20]">Assessment #{item.id}</p><p className="text-[10px] text-[#8b91a3]">{formatDate(item.created_at)}</p></div></div></td><td className="px-3 py-3 font-medium text-[#181c20]">{formatCurrency(item.loan_amnt)}</td><td className="px-3 py-3">{formatPercent(item.default_risk_probability)}</td><td className="px-3 py-3 font-semibold text-[#181c20]">{formatScore(item.risk_score, 0)}</td><td className="px-3 py-3"><span className={`rounded px-2 py-1 text-[10px] font-semibold ${risk.bg} ${risk.text}`}>{titleCase(item.risk_category)}</span></td><td className="px-3 py-3"><span className={`rounded px-2 py-1 text-[10px] font-semibold ${decision.bg} ${decision.text}`}>{titleCase(item.decision)}</span></td><td className="px-5 py-3 text-right"><Link to={`/history/${item.id}`} className="text-base text-[#2457d6]">›</Link></td></tr>; })}</tbody></table></div> : <div className="p-8 text-center text-sm text-[#6d7486]">No assessments match your search.</div>}
      <div className="flex items-center justify-between border-t border-[#e7e9f1] px-5 py-3 text-[11px] text-[#6d7486]"><span>Showing {filtered.length ? (page - 1) * PAGE_SIZE + 1 : 0} to {Math.min(page * PAGE_SIZE, total)} of {total} assessments</span><div className="flex gap-1"><button disabled={page <= 1} onClick={() => onPageChange(page - 1)} className="page-button">Previous</button><span className="page-number">{page}</span><button disabled={page >= pages} onClick={() => onPageChange(page + 1)} className="page-button">Next</button></div></div>
    </section>
  );
}

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState(null);
  const [history, setHistory] = useState({ items: [], total: 0, pages: 1 });
  const [page, setPage] = useState(1);
  const [reloadKey, setReloadKey] = useState(0);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    Promise.all([fetchAnalytics(), fetchPredictions({ page, per_page: PAGE_SIZE })])
      .then(([data, recent]) => { if (active) { setAnalytics(data); setHistory({ items: recent?.items || [], total: recent?.total || 0, pages: recent?.pages || 1 }); } })
      .catch((err) => { if (active) setError(err.message || 'Failed to load analytics'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [page, reloadKey]);

  const decisionCounts = analytics?.decision_counts || {};
  const approvalRate = analytics?.total_predictions ? (decisionCounts.APPROVE || 0) / analytics.total_predictions : null;
  const retry = () => { setError(null); setReloadKey((current) => current + 1); };

  return <div className="dashboard-page"><div className="mb-6 flex flex-col justify-between gap-4 xl:flex-row xl:items-end"><div><p className="dashboard-eyebrow">Institutional analytics <span>•</span> {history.items[0]?.model_version || 'Credit risk portfolio'}</p><h1 className="mt-2 text-2xl font-semibold tracking-tight text-[#181c20]">Portfolio Risk Overview</h1><p className="mt-1 text-sm text-[#6d7486]">Aggregate exposure metrics and underwriting telemetry from your portfolio.</p></div><div className="flex flex-wrap gap-2"><span className="dashboard-filter">Last 30 days <span>⌄</span></span><Link to="/assessment" className="dashboard-action">+ New Assessment</Link></div></div>{error && <div className="dashboard-error"><div><p className="font-semibold">Unable to load portfolio data.</p><p className="text-xs">Please try again.</p></div><button onClick={retry}>Retry</button></div>}{loading ? <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">{[1, 2, 3, 4].map((item) => <Skeleton key={item} className="h-32" />)}</div> : <><div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4"><Kpi label="Total assessments" value={analytics?.total_predictions ?? 0} note="Persisted portfolio assessments" /><Kpi label="Average default probability" value={formatPercent(analytics?.average_default_risk_probability)} note="Across all assessments" tone="amber" /><Kpi label="High risk count" value={(analytics?.risk_category_counts?.HIGH || 0) + (analytics?.risk_category_counts?.VERY_HIGH || 0)} note="High and very high risk" tone="red" /><Kpi label="Approval rate" value={formatPercent(approvalRate)} note={`${decisionCounts.APPROVE || 0} approved decisions`} tone="green" /></div><div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2"><RiskDistribution analytics={analytics} /><section className="dashboard-card flex min-h-[280px] flex-col justify-between p-5"><div><h2 className="dashboard-section-title">Decision Mix</h2><p className="dashboard-subtitle">Current portfolio underwriting outcomes</p></div><div className="space-y-4 py-5">{['APPROVE', 'REVIEW', 'REJECT'].map((key) => <div key={key}><div className="mb-1 flex justify-between text-xs"><span className="text-[#434654]">{titleCase(key)}</span><span className="font-semibold text-[#181c20]">{decisionCounts[key] || 0}</span></div><div className="h-2 rounded-full bg-[#edf0f8]"><div className={`h-2 rounded-full ${key === 'APPROVE' ? 'bg-[#12b76a]' : key === 'REVIEW' ? 'bg-[#f79009]' : 'bg-[#f04438]'}`} style={{ width: `${analytics?.total_predictions ? ((decisionCounts[key] || 0) / analytics.total_predictions) * 100 : 0}%` }} /></div></div>)}</div><Link to="/history" className="text-xs font-semibold text-[#2457d6]">View full assessment history →</Link></section></div></>}{!loading && <div className="mt-4"><RecentAssessments items={history.items} total={history.total} page={page} pages={history.pages} search={search} onSearch={setSearch} onPageChange={(nextPage) => { setSearch(''); setPage(nextPage); }} /></div>}</div>;
}
