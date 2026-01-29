import type { Agent } from '@/lib/types';
import { Sparkles, Search, Menu, X, User, Settings, Sun, Moon } from 'lucide-react';
import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface Props {
  agents: Agent[];
  selectedAgent: Agent | null;
  onSelectAgent: (agent: Agent) => void;
  loading?: boolean;
}

const THEME_KEY = 'webagents-theme';

export function Sidebar({ agents, selectedAgent, onSelectAgent, loading }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [isDark, setIsDark] = useState(() => {
    // Initialize from localStorage or default to dark
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(THEME_KEY);
      if (saved) {
        return saved === 'dark';
      }
      // Check system preference
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return true;
  });

  // Apply theme on mount and changes
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem(THEME_KEY, isDark ? 'dark' : 'light');
  }, [isDark]);

  const filtered = agents.filter(a => 
    a.name.toLowerCase().includes(search.toLowerCase()) ||
    a.description?.toLowerCase().includes(search.toLowerCase())
  );

  const toggleTheme = () => {
    setIsDark(!isDark);
  };

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="md:hidden fixed top-3 left-3 z-50 p-2 bg-background border rounded-lg"
      >
        {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Overlay */}
      {isOpen && <div className="md:hidden fixed inset-0 bg-black/50 z-40" onClick={() => setIsOpen(false)} />}

      {/* Sidebar */}
      <aside className={cn(
        'fixed md:relative inset-y-0 left-0 z-40 w-64 bg-sidebar-background border-r border-sidebar-border flex flex-col transition-transform md:translate-x-0',
        isOpen ? 'translate-x-0' : '-translate-x-full'
      )}>
        {/* Header */}
        <div className="h-14 px-4 border-b border-sidebar-border flex items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg text-sidebar-foreground">WebAgents</span>
          </div>
        </div>

        {/* Search */}
        <div className="p-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search agents..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm bg-sidebar-accent rounded-lg border-0 focus:outline-none focus:ring-2 focus:ring-primary text-sidebar-foreground placeholder:text-muted-foreground"
            />
          </div>
        </div>

        {/* Agent list */}
        <div className="flex-1 overflow-y-auto p-2">
          <div className="text-xs font-medium text-muted-foreground px-2 py-1">Local Agents</div>
          {loading ? (
            <div className="px-2 py-4 text-sm text-muted-foreground">Loading...</div>
          ) : filtered.length === 0 ? (
            <div className="px-2 py-4 text-sm text-muted-foreground">No agents found</div>
          ) : (
            <div className="space-y-1">
              {filtered.map(agent => (
                <button
                  key={agent.name}
                  onClick={() => { onSelectAgent(agent); setIsOpen(false); }}
                  className={cn(
                    'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors',
                    selectedAgent?.name === agent.name
                      ? 'bg-sidebar-accent text-sidebar-foreground'
                      : 'hover:bg-sidebar-accent/50 text-sidebar-foreground'
                  )}
                >
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{agent.name}</div>
                    <div className="text-xs text-muted-foreground truncate">{agent.description || 'AI Assistant'}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer with user menu */}
        <div className="p-3 border-t border-sidebar-border space-y-2">
          <button
            onClick={toggleTheme}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-sidebar-accent text-sidebar-foreground text-sm"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            <span>{isDark ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-sidebar-accent text-sidebar-foreground text-sm">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-sidebar-accent text-sidebar-foreground text-sm">
            <User className="w-4 h-4" />
            <span>Login</span>
          </button>
        </div>
      </aside>
    </>
  );
}
