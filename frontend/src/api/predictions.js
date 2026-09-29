import client from './client';

export async function createPrediction(payload) {
  const response = await client.post('/predict', payload);
  return response.data;
}

export async function fetchPredictions(params = {}) {
  const response = await client.get('/history', { params });
  return response.data;
}

export async function fetchPrediction(id) {
  const response = await client.get(`/history/${id}`);
  return response.data;
}

export async function fetchHistoryItem(id) {
  const response = await client.get(`/history/${id}`);
  return response.data;
}

export async function fetchAnalytics() {
  const response = await client.get('/analytics');
  return response.data;
}
