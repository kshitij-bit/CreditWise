export const HOME_OWNERSHIP_OPTIONS = ['RENT', 'MORTGAGE', 'OWN', 'OTHER'];

export const LOAN_INTENT_OPTIONS = [
  'PERSONAL',
  'EDUCATION',
  'MEDICAL',
  'VENTURE',
  'HOMEIMPROVEMENT',
  'DEBTCONSOLIDATION',
];

export const DEFAULT_ON_FILE_OPTIONS = ['N', 'Y'];

export const DEFAULT_FORM = {
  person_age: '',
  person_income: '',
  person_home_ownership: 'RENT',
  person_emp_length: '',
  loan_intent: 'PERSONAL',
  loan_amnt: '',
  cb_person_default_on_file: 'N',
  cb_person_cred_hist_length: '',
};

export const MODEL_BASED_NOTE =
  'The score and decision are model-based estimates derived from the applicant attributes supplied. Forecasted probabilities are generated for decision support only and should be interpreted alongside full underwriting review.';

export const RISK_SCORE_DISCLAIMER =
  'Score is a model-derived risk indicator in the 0-100 range and is not a credit bureau score.';

export const RISK_CATEGORY_LOW = 'LOW';
export const RISK_CATEGORY_MODERATE = 'MODERATE';
export const RISK_CATEGORY_HIGH = 'HIGH';
export const RISK_CATEGORY_VERY_HIGH = 'VERY_HIGH';

export const DECISION_APPROVE = 'APPROVE';
export const DECISION_REVIEW = 'REVIEW';
export const DECISION_REJECT = 'REJECT';

const RISK_CATEGORY_STYLES = {
  LOW: { label: 'Low Risk', badge: 'bg-emerald-100 text-emerald-800', text: 'text-emerald-600' },
  MODERATE: { label: 'Moderate Risk', badge: 'bg-yellow-100 text-yellow-800', text: 'text-yellow-600' },
  HIGH: { label: 'High Risk', badge: 'bg-orange-100 text-orange-800', text: 'text-orange-600' },
  VERY_HIGH: { label: 'Very High Risk', badge: 'bg-red-100 text-red-800', text: 'text-red-600' },
};

const DECISION_STYLES = {
  APPROVE: { label: 'Approve', badge: 'bg-emerald-100 text-emerald-800', text: 'text-emerald-600' },
  REVIEW: { label: 'Review', badge: 'bg-yellow-100 text-yellow-800', text: 'text-yellow-600' },
  REJECT: { label: 'Reject', badge: 'bg-red-100 text-red-800', text: 'text-red-600' },
};

export function getRiskCategoryStyle(category) {
  return RISK_CATEGORY_STYLES[category] || { label: category || 'Unknown', badge: 'bg-gray-100 text-gray-800', text: 'text-gray-600' };
}

export function getDecisionStyle(decision) {
  return DECISION_STYLES[decision] || { label: decision || 'Unknown', badge: 'bg-gray-100 text-gray-800', text: 'text-gray-600' };
}
