import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import ErrorMessage from '../components/common/ErrorMessage';

export default function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
    confirm: '',
  });
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (form.password !== form.confirm) {
      setError('Passwords do not match');
      return;
    }

    setSubmitting(true);
    try {
      await register({ name: form.name, email: form.email, password: form.password });
      navigate('/login', { replace: true });
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-100 px-4">
      <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="mb-6 text-center">
          <span className="text-3xl">🛡️</span>
          <h1 className="mt-2 text-2xl font-bold text-slate-900">Create Account</h1>
          <p className="text-sm text-slate-500">Join CreditWise AI</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && <ErrorMessage message={error} />}

          <div>
            <label htmlFor="name" className="mb-1 block text-sm font-medium text-slate-700">
              Full Name
            </label>
            <input
              id="name"
              name="name"
              type="text"
              required
              value={form.name}
              onChange={handleChange}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="Jane Doe"
            />
          </div>

          <div>
            <label htmlFor="email" className="mb-1 block text-sm font-medium text-slate-700">
              Email
            </label>
            <input
              id="email"
              name="email"
              type="email"
              required
              value={form.email}
              onChange={handleChange}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="mb-1 block text-sm font-medium text-slate-700">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              minLength={8}
              value={form.password}
              onChange={handleChange}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="At least 8 characters"
            />
          </div>

          <div>
            <label htmlFor="confirm" className="mb-1 block text-sm font-medium text-slate-700">
              Confirm Password
            </label>
            <input
              id="confirm"
              name="confirm"
              type="password"
              required
              value={form.confirm}
              onChange={handleChange}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="Repeat your password"
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-md bg-indigo-600 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-50"
          >
            {submitting ? 'Creating account…' : 'Create Account'}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-slate-500">
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-indigo-600 hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
