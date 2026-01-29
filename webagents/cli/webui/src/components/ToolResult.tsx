import { useState, ReactNode } from 'react';
import { ChevronRight, Loader2, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  icon: ReactNode;
  status: 'success' | 'error' | 'loading' | 'warning';
  statusText: string;
  children: ReactNode;
  defaultExpanded?: boolean;
}

export function ToolResult({ icon, status, statusText, children, defaultExpanded = false }: Props) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const statusIcon = status === 'loading' ? <Loader2 className="w-3 h-3 animate-spin" /> :
                     status === 'success' ? <CheckCircle className="w-3 h-3 text-green-500" /> :
                     status === 'error' ? <XCircle className="w-3 h-3 text-red-500" /> :
                     <AlertTriangle className="w-3 h-3 text-yellow-500" />;

  return (
    <div className="my-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
      >
        <div className="w-4 h-4">{icon}</div>
        <span className="text-sm">{statusText}</span>
        {statusIcon}
        <ChevronRight className={cn("w-3 h-3 transition-transform", isExpanded && "rotate-90")} />
      </button>
      {isExpanded && (
        <div className="ml-2 pl-4 mt-1 border-l-2 border-muted text-sm">
          {children}
        </div>
      )}
    </div>
  );
}
