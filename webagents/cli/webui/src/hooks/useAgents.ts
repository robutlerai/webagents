import { useState, useEffect, useCallback } from 'react';
import type { Agent } from '@/lib/types';
import { api } from '@/lib/api';

export function useAgents() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const loadAgents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getAgents();
      setAgents(data);
    } catch (err) {
      console.error('Failed to load agents:', err);
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAgents();
  }, [loadAgents]);

  return { agents, loading, error, refresh: loadAgents };
}
