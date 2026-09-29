export function formatCurrency(value, digits = 0) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  try {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: digits,
    }).format(Number(value));
  } catch {
    return String(value);
  }
}

export function formatPercent(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function formatScore(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

export function formatDate(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function titleCase(value) {
  if (value == null || value === '') return '-';
  return String(value)
    .replace(/_/g, ' ')
    .toLowerCase()
    .split(' ')
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : ''))
    .join(' ');
}

export function formatOptional(value) {
  if (value === null || value === undefined || value === '') return 'Not provided';
  return String(value);
}
