import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchAdminUsers, setUserActive } from '../api/model';
import { useAuth } from '../auth/AuthContext';
import { formatDate } from '../lib/format';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';
import StatCard from '../components/common/StatCard';

export default function AdminUsersPage() {
  const { user } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actingId, setActingId] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchAdminUsers()
      .then((data) => setUsers(Array.isArray(data) ? data : data.items || []))
      .catch((err) => setError(err.message || 'Failed to load users'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const activeCount = useMemo(() => users.filter((u) => u.is_active).length, [users]);
  const adminCount = useMemo(() => users.filter((u) => u.role === 'ADMIN').length, [users]);

  const toggleActive = async (u) => {
    setActingId(u.id);
    setError(null);
    try {
      await setUserActive(u.id, !u.is_active);
      setUsers((prev) =>
        prev.map((x) => (x.id === u.id ? { ...x, is_active: !u.is_active } : x))
      );
    } catch (err) {
      setError(err.message || 'Failed to update user');
    } finally {
      setActingId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">User Management</h1>
          <p className="text-sm text-slate-500">Manage analyst and admin accounts.</p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Active Users" value={activeCount} />
          <StatCard label="Admins" value={adminCount} />
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {loading ? (
        <LoadingSpinner label="Loading users…" />
      ) : users.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-10 text-center">
          <p className="text-3xl">👥</p>
          <p className="mt-2 text-sm font-medium text-slate-700">No users found</p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  ID
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Email
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Role
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Created
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((u) => (
                <tr key={u.id} className="hover:bg-slate-50">
                  <td className="whitespace-nowrap px-4 py-3 text-slate-500">#{u.id}</td>
                  <td className="whitespace-nowrap px-4 py-3 font-medium text-slate-800">
                    {u.email}
                    {u.email === user?.email && (
                      <span className="ml-2 rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-600">
                        you
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${
                        u.role === 'ADMIN'
                          ? 'bg-violet-100 text-violet-700'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {u.role}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${
                        u.is_active
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-red-100 text-red-600'
                      }`}
                    >
                      {u.is_active ? 'Active' : 'Disabled'}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-slate-500">
                    {formatDate(u.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs text-slate-400">
        Only administrators can view this page. Activation toggling is not exposed to analysts.
      </p>
    </div>
  );
}
