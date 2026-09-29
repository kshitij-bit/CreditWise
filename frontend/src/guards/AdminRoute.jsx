import { Navigate } from 'react-router-dom';
import PropTypes from 'prop-types';
import { useAuth } from '../auth/AuthContext';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function AdminRoute({ children }) {
  const { user, isAdmin, initializing } = useAuth();
  if (initializing) return <LoadingSpinner label="Checking session..." />;
  if (!user) return <Navigate to="/login" replace />;
  if (!isAdmin) return <Navigate to="/unauthorized" replace />;
  return children;
}

AdminRoute.propTypes = { children: PropTypes.node.isRequired };
