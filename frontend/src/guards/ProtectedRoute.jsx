import { Navigate, useLocation } from 'react-router-dom';
import PropTypes from 'prop-types';
import { useAuth } from '../auth/AuthContext';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function ProtectedRoute({ children }) {
  const { user, initializing } = useAuth();
  const location = useLocation();
  if (initializing) return <LoadingSpinner label="Checking session..." />;
  if (!user) return <Navigate to="/login" replace state={{ from: location }} />;
  return children;
}

ProtectedRoute.propTypes = { children: PropTypes.node.isRequired };
