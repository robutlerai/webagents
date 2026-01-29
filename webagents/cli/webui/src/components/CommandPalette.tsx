import type { Command } from '@/lib/types';
import { cn } from '@/lib/utils';
import { useRef, useEffect } from 'react';

interface Props {
  commands: Command[];
  isOpen: boolean;
  selectedIndex: number;
  onClose: () => void;
  onSelect: (command: Command) => void;
}

// Convert /session/history to /session history for display
function formatCommandDisplay(path: string): string {
  if (path.startsWith('/')) {
    return '/' + path.slice(1).replace(/\//g, ' ');
  }
  return path;
}

// Check if command likely needs arguments based on description or path
function getArgHint(cmd: Command): string | null {
  const path = cmd.path.toLowerCase();
  if (path.includes('load')) return '<session_id>';
  if (path.includes('save') && !path.includes('history')) return '<name>';
  if (path.includes('restore')) return '<checkpoint_id>';
  return null;
}

export function CommandPalette({ commands, isOpen, selectedIndex, onSelect }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

  // Scroll selected item into view
  useEffect(() => {
    if (selectedRef.current && listRef.current) {
      selectedRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [selectedIndex]);

  if (!isOpen || commands.length === 0) return null;

  return (
    <div 
      ref={listRef}
      className="absolute bottom-full left-0 right-0 mb-2 bg-popover border rounded-lg shadow-lg overflow-hidden max-h-48 overflow-y-auto"
    >
      {commands.map((cmd, i) => {
        const argHint = getArgHint(cmd);
        return (
          <button
            key={cmd.path}
            ref={i === selectedIndex ? selectedRef : null}
            onClick={() => onSelect(cmd)}
            className={cn(
              "w-full px-3 py-2 text-left text-sm flex items-center gap-2 hover:bg-muted transition-colors",
              i === selectedIndex && "bg-muted"
            )}
          >
            <span className="font-mono text-primary">{formatCommandDisplay(cmd.path)}</span>
            {argHint && <span className="font-mono text-muted-foreground/70">{argHint}</span>}
            {cmd.description && <span className="text-muted-foreground truncate">{cmd.description}</span>}
          </button>
        );
      })}
    </div>
  );
}
