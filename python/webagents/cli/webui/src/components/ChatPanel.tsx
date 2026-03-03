import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { Agent, Attachment, Command } from '@/lib/types';
import { useChat } from '@/hooks/useChat';
import { MessageList } from './MessageList';
import { CommandPalette } from './CommandPalette';
import { Send, Square, Sparkles, Paperclip, Mic, PanelRight, X, Radio } from 'lucide-react';
import { cn } from '@/lib/utils';
import { api } from '@/lib/api';

interface Props {
  agent: Agent | null;
  showDetails: boolean;
  onToggleDetails: () => void;
}

// Only keep truly local commands that don't hit the API
const LOCAL_COMMANDS: Command[] = [
  { path: '/help', description: 'Show available commands' },
  { path: '/clear', description: 'Clear chat history' },
];

export function ChatPanel({ agent, showDetails, onToggleDetails }: Props) {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [showCommands, setShowCommands] = useState(false);
  const [realtimeMode, setRealtimeMode] = useState(false);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [agentCommands, setAgentCommands] = useState<Command[]>([]);
  const { messages, sendMessage, executeCommand, isStreaming, clearMessages, stopStreaming, addSystemMessage } = useChat(agent);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportsRealtime = agent?.skills?.includes('realtime') || false;

  // Load agent-specific commands
  useEffect(() => {
    if (agent) {
      api.getCommands(agent.name).then(cmds => {
        // Commands from API come as 'session/history', normalize to '/session/history'
        setAgentCommands(cmds.map(c => ({ 
          path: c.path.startsWith('/') ? c.path : '/' + c.path, 
          description: c.description 
        })));
      });
    } else {
      setAgentCommands([]);
    }
  }, [agent?.name]);

  const allCommands = useMemo(() => [...LOCAL_COMMANDS, ...agentCommands], [agentCommands]);

  const filteredCommands = useMemo(() => {
    if (!input.startsWith('/')) return [];
    const filter = input.slice(1).toLowerCase();
    if (!filter) return allCommands.slice(0, 10); // Show first 10 when just typing /
    return allCommands.filter(cmd => 
      cmd.path.slice(1).toLowerCase().includes(filter) ||
      cmd.description?.toLowerCase().includes(filter)
    ).slice(0, 10);
  }, [input, allCommands]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  useEffect(() => {
    const shouldShow = input.startsWith('/') && filteredCommands.length > 0;
    setShowCommands(shouldShow);
    if (shouldShow) setSelectedCommandIndex(0);
  }, [input, filteredCommands.length]);

  // Refocus after streaming ends
  useEffect(() => {
    if (!isStreaming) {
      textareaRef.current?.focus();
    }
  }, [isStreaming]);

  // Format help like CLI REPL with proper markdown
  const formatHelp = useCallback(() => {
    let help = '## Commands\n\n';
    help += '- `/clear` - Clear chat\n';
    help += '- `/help` - Show this help\n';
    
    const sessionCmds = agentCommands.filter(c => c.path.startsWith('/session'));
    const mcpCmds = agentCommands.filter(c => c.path.startsWith('/mcp'));
    const otherCmds = agentCommands.filter(c => !c.path.startsWith('/session') && !c.path.startsWith('/mcp'));
    
    if (sessionCmds.length > 0) {
      help += '\n## Session\n\n';
      sessionCmds.forEach(c => {
        const displayPath = '/' + c.path.slice(1).replace(/\//g, ' ');
        help += `- \`${displayPath}\` - ${c.description || ''}\n`;
      });
    }
    
    if (mcpCmds.length > 0) {
      help += '\n## MCP\n\n';
      mcpCmds.forEach(c => {
        const displayPath = '/' + c.path.slice(1).replace(/\//g, ' ');
        help += `- \`${displayPath}\` - ${c.description || ''}\n`;
      });
    }
    
    if (otherCmds.length > 0) {
      help += '\n## Other\n\n';
      otherCmds.forEach(c => {
        const displayPath = '/' + c.path.slice(1).replace(/\//g, ' ');
        help += `- \`${displayPath}\` - ${c.description || ''}\n`;
      });
    }
    
    return help;
  }, [agentCommands]);

  const handleLocalCommand = useCallback(async (command: string): Promise<boolean> => {
    const cmd = command.trim().toLowerCase();
    
    if (cmd === '/clear') {
      clearMessages();
      return true;
    }
    if (cmd === '/help') {
      addSystemMessage(formatHelp());
      return true;
    }
    
    return false;
  }, [clearMessages, addSystemMessage, formatHelp]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    const trimmedInput = input.trim();
    
    if (!trimmedInput && attachments.length === 0) return;
    if (isStreaming) return;

    const currentInput = trimmedInput;
    setInput('');
    setAttachments([]);

    // Check if it's a slash command
    if (currentInput.startsWith('/')) {
      // First try local commands
      const isLocal = await handleLocalCommand(currentInput);
      if (isLocal) {
        textareaRef.current?.focus();
        return;
      }
      
      // Then try API commands
      await executeCommand(currentInput);
      textareaRef.current?.focus();
      return;
    }

    // Send as regular message
    sendMessage(currentInput, attachments);
  }, [input, attachments, isStreaming, handleLocalCommand, executeCommand, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showCommands) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCommandIndex(i => Math.min(i + 1, filteredCommands.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCommandIndex(i => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selectedCommandIndex]) {
          // Insert command + space for user to type args, don't execute
          setInput(filteredCommands[selectedCommandIndex].path + ' ');
          setShowCommands(false);
        }
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setShowCommands(false);
        return;
      }
      if (e.key === 'Tab' && filteredCommands[selectedCommandIndex]) {
        e.preventDefault();
        setInput(filteredCommands[selectedCommandIndex].path + ' ');
        return;
      }
    }
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    
    Array.from(files).forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        setAttachments(prev => [...prev, {
          id: crypto.randomUUID(),
          name: file.name,
          type: file.type,
          data: reader.result as string,
        }]);
      };
      reader.readAsDataURL(file);
    });
    e.target.value = '';
  };

  const handleCommandSelect = (cmd: Command) => {
    setShowCommands(false);
    
    // For simple action commands, execute immediately
    if (cmd.path === '/clear') {
      clearMessages();
      setInput('');
      textareaRef.current?.focus();
      return;
    }
    if (cmd.path === '/help') {
      setInput('');
      addSystemMessage(formatHelp());
      textareaRef.current?.focus();
      return;
    }
    
    // For other commands, insert with space so user can add args
    // Commands that don't need args will be executed on Enter
    setInput(cmd.path + ' ');
    textareaRef.current?.focus();
  };

  const toggleRealtimeMode = () => {
    setRealtimeMode(!realtimeMode);
  };

  if (!agent) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-4 max-w-md px-4">
          <div className="w-16 h-16 mx-auto rounded-full bg-primary/10 flex items-center justify-center">
            <Sparkles className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-2xl font-semibold">Welcome to WebAgents</h2>
          <p className="text-muted-foreground">Select an agent from the sidebar to start</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-w-0 h-full relative">
      {realtimeMode && (
        <div className="absolute inset-0 bg-background/95 flex flex-col items-center justify-center z-50">
          <div className="text-center space-y-6">
            <div className="w-32 h-32 rounded-full bg-red-500/20 flex items-center justify-center">
              <div className="w-24 h-24 rounded-full bg-red-500/30 flex items-center justify-center animate-pulse">
                <Radio className="w-12 h-12 text-red-500" />
              </div>
            </div>
            <div>
              <p className="text-xl font-medium">Listening...</p>
              <p className="text-sm text-muted-foreground mt-1">Speak to {agent.name}</p>
              {!supportsRealtime && (
                <p className="text-xs text-yellow-500 mt-2">Note: This agent doesn't support realtime audio</p>
              )}
            </div>
            <button 
              onClick={() => setRealtimeMode(false)}
              className="px-6 py-2 bg-destructive text-destructive-foreground rounded-full hover:bg-destructive/90"
            >
              End Conversation
            </button>
          </div>
        </div>
      )}

      <header className="h-14 border-b px-4 flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
          <Sparkles className="w-4 h-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <h1 className="font-semibold truncate">{agent.name}</h1>
          <p className="text-xs text-muted-foreground truncate">{agent.description || 'AI Assistant'}</p>
        </div>
        <button 
          onClick={onToggleDetails} 
          className={cn(
            "p-2 rounded-lg transition-colors",
            showDetails ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground hover:bg-muted"
          )}
          title="Toggle details"
        >
          <PanelRight className="w-4 h-4" />
        </button>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Sparkles className="w-12 h-12 text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">Send a message to start chatting with {agent.name}</p>
              <p className="text-sm text-muted-foreground mt-2">Type / for commands</p>
            </div>
          ) : (
            <MessageList messages={messages} isStreaming={isStreaming} />
          )}
        </div>
      </div>

      {attachments.length > 0 && (
        <div className="px-4 py-2 border-t flex gap-2 flex-wrap">
          {attachments.map(att => (
            <div key={att.id} className="relative group">
              {att.type.startsWith('image/') ? (
                <div className="relative w-16 h-16 rounded-lg overflow-hidden border">
                  <img src={att.data} alt={att.name} className="w-full h-full object-cover" />
                  <button 
                    onClick={() => setAttachments(prev => prev.filter(a => a.id !== att.id))}
                    className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-4 h-4 text-white" />
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg text-sm">
                  <Paperclip className="w-3 h-3" />
                  <span className="truncate max-w-24">{att.name}</span>
                  <button onClick={() => setAttachments(prev => prev.filter(a => a.id !== att.id))} className="hover:text-destructive">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="border-t p-4 shrink-0">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
          <CommandPalette
            commands={filteredCommands}
            isOpen={showCommands}
            selectedIndex={selectedCommandIndex}
            onClose={() => setShowCommands(false)}
            onSelect={handleCommandSelect}
          />
          
          <div className="flex items-end gap-2 bg-muted/50 rounded-2xl border p-2">
            <input type="file" ref={fileInputRef} onChange={handleFileSelect} className="hidden" multiple accept="image/*,.pdf,.txt,.md,.json" />
            <button type="button" onClick={() => fileInputRef.current?.click()} className="p-2 text-muted-foreground hover:text-foreground rounded-lg hover:bg-muted">
              <Paperclip className="w-4 h-4" />
            </button>

            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Message ${agent.name}... (/ for commands)`}
              rows={1}
              disabled={isStreaming}
              autoFocus
              className="flex-1 resize-none bg-transparent border-0 focus:outline-none px-2 py-1.5 max-h-[200px] text-sm"
            />

            <button 
              type="button" 
              onClick={toggleRealtimeMode} 
              className={cn(
                "p-2 rounded-lg transition-colors",
                realtimeMode ? "text-red-500 bg-red-500/10" : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
              title={supportsRealtime ? "Realtime audio mode" : "Voice input"}
            >
              <Mic className="w-4 h-4" />
            </button>

            <button
              type={isStreaming ? 'button' : 'submit'}
              onClick={isStreaming ? stopStreaming : undefined}
              disabled={!isStreaming && !input.trim() && attachments.length === 0}
              className={cn(
                'shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-colors',
                isStreaming 
                  ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
                  : 'bg-primary text-primary-foreground disabled:opacity-50'
              )}
            >
              {isStreaming ? <Square className="w-3 h-3" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
