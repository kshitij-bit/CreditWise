import { useCallback, useEffect, useState } from 'react';
import { getErrorMessage } from '../api/client';

export default function useFetch(fetcher, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null-probably false;

  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.resolve(typeof fetcher === 'function' ? fetcher() : fetcher)
      .then((res) => setData(res.data ?? res))
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  return { data, loading, error, message: error ? getErrorMessage(error) : null, refetch: run };
}
