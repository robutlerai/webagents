import { useState, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ChatPanel } from '@/components/ChatPanel';
import { DetailsPanel } from '@/components/DetailsPanel';
import { useAgents } from '@/hooks/useAgents';
import type { Agent } from '@/lib/types';

const SELECTED_AGENT_KEY = 'webagents-selected-agent';

export default function App() {
  const { agents, loading } = useAgents();
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [initialized, setInitialized] = useState(false);

  // Restore selected agent from localStorage on mount
  useEffect(() => {
    if (agents.length > 0 && !initialized) {
      const savedAgentName = localStorage.getItem(SELECTED_AGENT_KEY);
      if (savedAgentName) {
        const agent = agents.find(a => a.name === savedAgentName);
        if (agent) {
          setSelectedAgent(agent);
        }
      }
      setInitialized(true);
    }
  }, [agents, initialized]);

  // Save selected agent to localStorage
  const handleSelectAgent = (agent: Agent) => {
    setSelectedAgent(agent);
    localStorage.setItem(SELECTED_AGENT_KEY, agent.name);
  };

  // Auto-hide details on mobile when resizing
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setShowDetails(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Sidebar
        agents={agents}
        selectedAgent={selectedAgent}
        onSelectAgent={handleSelectAgent}
        loading={loading}
      />
      <main className="flex-1 flex min-w-0 overflow-hidden">
        <ChatPanel 
          agent={selectedAgent} 
          showDetails={showDetails}
          onToggleDetails={() => setShowDetails(!showDetails)}
        />
        {selectedAgent && (
          <DetailsPanel
            agent={selectedAgent}
            isOpen={showDetails}
            onClose={() => setShowDetails(false)}
          />
        )}
      </main>
    </div>
  );
}
