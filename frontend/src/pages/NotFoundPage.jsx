import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

export default function NotFoundPage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-slate-100 px-4 text-center">
      <span className="text-5xl">🔍</span>
      <h1 className="mt-4 text-3xl font-bold text-slate-900">404 · Page Not Found</h1>
      <p className="mt-2 max-w-md text-sm text-slate-600">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Link
        to="/dashboard"
        className="mt-6 rounded-md bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700"
      >
        Back to Dashboard
      </Link>
    </div>
  );
}
