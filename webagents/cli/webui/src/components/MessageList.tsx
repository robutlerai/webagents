import type { Message } from '@/lib/types';
import { Markdown } from './Markdown';
import { User, Sparkles, Paperclip, Terminal } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  messages: Message[];
  isStreaming?: boolean;
}

// Animated blob indicator - single dot with organic wave animation
function StreamingCursor() {
  return (
    <span className="inline-block w-3 h-3 rounded-full bg-primary animate-blob-breathe" />
  );
}

export function MessageList({ messages, isStreaming }: Props) {
  return (
    <div className="flex flex-col gap-4">
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1;
        const showCursor = isStreaming && isLast && msg.role === 'assistant';
        const showStreamingCursor = showCursor && !msg.content;
        
        // System messages (command output)
        if (msg.role === 'system') {
          return (
            <div key={msg.id || i} className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-yellow-500/10 flex items-center justify-center shrink-0 ring-1 ring-yellow-500/30">
                <Terminal className="w-4 h-4 text-yellow-500" />
              </div>
              <div className="max-w-[85%] rounded-2xl px-4 py-3 bg-yellow-500/10 border border-yellow-500/20">
                <Markdown content={msg.content} />
              </div>
            </div>
          );
        }
        
        return (
          <div
            key={msg.id || i}
            className={cn(
              'flex gap-3',
              msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
            )}
          >
            <div className={cn(
              'w-8 h-8 rounded-full flex items-center justify-center shrink-0',
              msg.role === 'user' 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-muted ring-1 ring-border'
            )}>
              {msg.role === 'user' ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
            </div>
            
            <div className={cn(
              'max-w-[80%] rounded-2xl px-4 py-3',
              msg.role === 'user'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted'
            )}>
              {msg.role === 'user' ? (
                <div className="space-y-2">
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {msg.attachments.map(att => (
                        <div key={att.id} className="flex items-center gap-1 px-2 py-0.5 bg-primary-foreground/20 rounded text-xs">
                          <Paperclip className="w-3 h-3" />
                          <span className="truncate max-w-24">{att.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : showStreamingCursor ? (
                <StreamingCursor />
              ) : (
                <Markdown content={msg.content} toolCalls={msg.toolCalls} isStreaming={showCursor} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
