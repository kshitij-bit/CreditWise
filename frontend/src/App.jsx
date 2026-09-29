import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './auth/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './guards/ProtectedRoute';
import AdminRoute from './guards/AdminRoute';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import AssessmentPage from './pages/AssessmentPage';
import HistoryPage from './pages/HistoryPage';
import HistoryDetailPage from './pages/HistoryDetailPage';
import ModelInfoPage from './pages/ModelInfoPage';
import ModelDetailPage from './pages/ModelDetailPage';
import AdminUsersPage from './pages/AdminUsersPage';
import UnauthorizedPage from './pages/UnauthorizedPage';
import NotFoundPage from './pages/NotFoundPage';

function AppRoutes() {
  const { loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center text-gray-500">
        Loading…
      </div>
    );
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/assessment" element={<AssessmentPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<HistoryDetailPage />} />
        <Route path="/model-info" element={<ModelInfoPage />} />
        <Route path="/model-detail" element={<ModelDetailPage />} />
        <Route
          path="/admin/users"
          element={
            <AdminRoute>
              <AdminUsersPage />
            </AdminRoute>
          }
        />
      </Route>

      <Route path="/unauthorized" element={<UnauthorizedPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}

export default function App() {
  return <AppRoutes />;
}
