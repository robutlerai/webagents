import type { Agent } from '@/lib/types';
import { X, Trash2, Settings, Info, Cpu, MapPin, Terminal } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  agent: Agent;
  isOpen: boolean;
  onClose: () => void;
  onDelete?: () => void;
}

export function DetailsPanel({ agent, isOpen, onClose, onDelete }: Props) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="md:hidden fixed inset-0 bg-black/50 z-40"
          onClick={onClose}
        />
      )}

      {/* Panel - fixed on mobile, relative on desktop */}
      <aside className={cn(
        'fixed md:relative inset-y-0 right-0 z-40 w-80 bg-card border-l flex flex-col transition-transform md:transition-none',
        isOpen ? 'translate-x-0' : 'translate-x-full md:hidden'
      )}>
        {/* Header */}
        <div className="h-14 px-4 border-b flex items-center justify-between shrink-0">
          <h2 className="font-semibold">Agent Details</h2>
          <button onClick={onClose} className="p-2 hover:bg-muted rounded-lg">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Agent Info */}
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Cpu className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold">{agent.name}</h3>
                <span className="text-xs px-2 py-0.5 bg-green-500/10 text-green-500 rounded-full">Online</span>
              </div>
            </div>
            
            <p className="text-sm text-muted-foreground">{agent.description || 'An AI assistant'}</p>
          </div>

          {/* Details */}
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Settings className="w-4 h-4" />
              <span>Model:</span>
              <span className="text-foreground">{agent.model || 'Auto'}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Info className="w-4 h-4" />
              <span>Version:</span>
              <span className="text-foreground">{agent.version || '1.0.0'}</span>
            </div>
            {agent.source && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <MapPin className="w-4 h-4" />
                <span>Source:</span>
                <span className="text-foreground truncate">{agent.source}</span>
              </div>
            )}
          </div>

          {/* Commands section */}
          <div className="pt-4 border-t">
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Terminal className="w-4 h-4" />
              Quick Commands
            </h4>
            <div className="space-y-1 text-sm">
              <div className="p-2 bg-muted rounded font-mono text-xs">/help - Show help</div>
              <div className="p-2 bg-muted rounded font-mono text-xs">/clear - Clear chat</div>
              <div className="p-2 bg-muted rounded font-mono text-xs">/session - Manage sessions</div>
            </div>
          </div>
        </div>

        {/* Footer with delete */}
        <div className="p-4 border-t shrink-0">
          <button
            onClick={onDelete}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            <span>Delete Agent</span>
          </button>
        </div>
      </aside>
    </>
  );
}
