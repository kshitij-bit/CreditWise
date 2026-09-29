import axios from 'axios';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000/api').replace(/\/$/, '');

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

let authToken = null;
let unauthorizedHandler = null;

export function setAuthToken(token) {
  authToken = token || null;
}

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler;
}

client.interceptors.request.use((config) => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  }
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    if (status === 401 && unauthorizedHandler) {
      unauthorizedHandler();
    }
    return Promise.reject(error);
  },
);

export function getErrorMessage(error) {
  if (!error) return 'Something went wrong. Please try again.';
  if (typeof error === 'string') return error;
  const data = error.response?.data;
  if (data) {
    if (typeof data.error === 'string') return data.error;
    if (typeof data.message === 'string') return data.message;
    if (Array.isArray(data.errors)) {
      return data.errors.map((e) => (typeof e === 'string' ? e : e.msg || JSON.stringify(e))).join(' ');
    }
  }
  if (error.code === 'ECONNABORTED') return 'The request timed out. Please try again.';
  if (error.message === 'Network Error') return 'Cannot reach the server. Is the backend running?';
  if (error.message) return error.message;
  return 'Something went wrong. Please try again.';
}

export default client;
