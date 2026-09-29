import client from './client';

export async function fetchModelInfo() {
  const response = await client.get('/model/info');
  return response.data;
}

export async function fetchModelExplain() {
  const response = await client.get('/model/explain');
  return response.data;
}

export async function fetchShapExplanation(predictionId) {
  const response = await client.get(`/explain/${predictionId}`);
  return response.data;
}

export async function fetchAdminUsers() {
  const response = await client.get('/admin/users');
  return response.data;
}

export async function setUserActive(userId, isActive) {
  const response = await client.put(`/admin/users/${userId}`, { is_active: isActive });
  return response.data;
}
