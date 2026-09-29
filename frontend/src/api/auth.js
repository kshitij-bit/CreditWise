import client from './client';
import { getErrorMessage } from './client';

export async function login(credentials) {
  const response = await client.post('/auth/login', credentials);
  return response.data;
}

export async function register(userData) {
  const response = await client.post('/auth/register', userData);
  return response.data;
}

export async function fetchMe() {
  const response = await client.get('/auth/me');
  return response.data;
}

export { getErrorMessage };
