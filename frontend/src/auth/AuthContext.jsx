import { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';
import PropTypes from 'prop-types';
import { login as apiLogin, register as apiRegister, fetchMe } from '../api/auth';
import { setAuthToken, setUnauthorizedHandler } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const raw = localStorage.getItem('cw_user');
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });
  const [initializing, setInitializing] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('cw_token');
    if (token) setAuthToken(token);
    setUnauthorizedHandler(() => {
      localStorage.removeItem('cw_token');
      localStorage.removeItem('cw_user');
      setUser(null);
    });
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('cw_token');
    if (!token) {
      setInitializing(false);
      return;
    }
    fetchMe()
      .then((data) => {
        // /auth/me returns the user directly; login returns it under `user`.
        const currentUser = data.user || data;
        setUser(currentUser);
        localStorage.setItem('cw_user', JSON.stringify(currentUser));
      })
      .catch(() => {
        localStorage.removeItem('cw_token');
        localStorage.removeItem('cw_user');
        setUser(null);
      })
      .finally(() => setInitializing(false));
  }, []);

  const login = useCallback(async (credentials) => {
    const data = await apiLogin(credentials);
    localStorage.setItem('cw_token', data.access_token);
    localStorage.setItem('cw_user', JSON.stringify(data.user));
    setAuthToken(data.access_token);
    setUser(data.user);
    return data.user;
  }, []);

  const register = useCallback(async (userData) => {
    const data = await apiRegister(userData);
    return data;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('cw_token');
    localStorage.removeItem('cw_user');
    setAuthToken(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      initializing,
      isAuthenticated: !!user,
      isAdmin: !!user && user.role === 'ADMIN',
      login,
      register,
      logout,
    }),
    [user, initializing, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

AuthProvider.propTypes = { children: PropTypes.node.isRequired };
