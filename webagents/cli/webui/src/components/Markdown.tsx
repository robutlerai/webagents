import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.css';
import { useState, useMemo, useEffect, useRef } from 'react';
import { ChevronRight, Brain, Loader2, Wrench, CheckCircle, Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

import type { ToolCall } from '@/lib/types';

interface MarkdownProps {
  content: string;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

function ThinkingBlock({ content, isStreaming }: { content: string; isStreaming?: boolean }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Extract subtitle from **bold** markers in content (like CLI REPL does)
  const subtitle = useMemo(() => {
    const matches = content.match(/\*\*([^*]+)\*\*/g);
    if (matches && matches.length > 0) {
      // Get last complete subtitle
      const last = matches[matches.length - 1];
      return last.replace(/\*\*/g, '').trim();
    }
    // Try incomplete subtitle at end
    const openMatch = content.match(/\*\*([^*]+)$/);
    if (openMatch) {
      return openMatch[1].trim() + '...';
    }
    return null;
  }, [content]);

  const headerText = subtitle || (isStreaming ? 'Thinking...' : 'Thinking');
  
  return (
    <div className="my-2 border border-border/50 rounded-lg bg-card overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:bg-muted/50 transition-colors"
      >
        {isStreaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
        <span className="font-medium truncate">{headerText}</span>
        <ChevronRight className={cn("w-4 h-4 ml-auto shrink-0 transition-transform", isExpanded && "rotate-90")} />
      </button>
      {isExpanded && (
        <div className="px-3 py-2 border-t border-border/50 bg-muted/30 max-h-64 overflow-y-auto prose prose-sm dark:prose-invert prose-p:my-1 prose-headings:my-2 text-muted-foreground">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

function ToolCallBlock({ toolCall }: { toolCall: ToolCall }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const { name, args, result, status } = toolCall;
  
  const getToolSummary = () => {
    const argsObj = args || {};
    
    if (name === 'read_file' || name === 'read') {
      const path = argsObj.path || argsObj.file_path || 'file';
      return `Read ${String(path).split('/').pop()}`;
    } else if (name === 'write_file' || name === 'write') {
      const path = argsObj.file_path || argsObj.path || 'file';
      return `Wrote ${String(path).split('/').pop()}`;
    } else if (name === 'list_directory') {
      const path = argsObj.path || '.';
      return `Listed ${String(path).split('/').pop() || 'directory'}`;
    } else if (['shell', 'bash', 'run_command'].includes(name)) {
      const cmd = argsObj.command || argsObj.cmd || '';
      return `Ran: ${String(cmd).slice(0, 30)}${String(cmd).length > 30 ? '...' : ''}`;
    }
    return name.replace(/_/g, ' ');
  };

  const statusIcon = status === 'running' || status === 'pending'
    ? <Loader2 className="w-4 h-4 animate-spin" /> 
    : status === 'error'
    ? <Wrench className="w-4 h-4 text-red-500" />
    : <Wrench className="w-4 h-4 text-green-500" />;

  return (
    <div className="my-2 border border-border/50 rounded-lg bg-card overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:bg-muted/50 transition-colors"
      >
        {statusIcon}
        <span className="font-medium truncate">{getToolSummary()}</span>
        <CheckCircle className="w-3 h-3 text-green-500 ml-1" />
        <ChevronRight className={cn("w-4 h-4 ml-auto shrink-0 transition-transform", isExpanded && "rotate-90")} />
      </button>
      {isExpanded && (
        <div className="px-3 py-2 border-t border-border/50 bg-muted/30 space-y-3 max-h-64 overflow-y-auto">
          {args && Object.keys(args).length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground mb-1">Arguments</div>
              <pre className="text-xs bg-background/50 p-2 rounded overflow-x-auto font-mono border border-border/30">
                {JSON.stringify(args, null, 2)}
              </pre>
            </div>
          )}
          {result && (
            <div>
              <div className="text-xs font-medium text-muted-foreground mb-1">Result</div>
              <pre className="text-xs bg-background/50 p-2 rounded overflow-x-auto max-h-32 overflow-y-auto font-mono border border-border/30">
                {typeof result === 'string' 
                  ? result.slice(0, 1000) + (result.length > 1000 ? '...' : '')
                  : JSON.stringify(result, null, 2).slice(0, 1000)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CodeBlock({ className, children, ...props }: any) {
  const [copied, setCopied] = useState(false);
  const codeRef = useRef<HTMLElement>(null);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const isInline = !className;
  const codeString = String(children).replace(/\n$/, '');
  
  useEffect(() => {
    if (codeRef.current && language && !isInline) {
      // Remove previous highlighting
      codeRef.current.removeAttribute('data-highlighted');
      try {
        hljs.highlightElement(codeRef.current);
      } catch {}
    }
  }, [codeString, language, isInline]);
  
  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  if (isInline) {
    return <code className="px-1.5 py-0.5 rounded bg-muted text-sm font-mono" {...props}>{children}</code>;
  }
  
  return (
    <div className="my-3 rounded-lg overflow-hidden group relative" style={{ backgroundColor: 'hsl(var(--code-bg))' }}>
      <div 
        className="px-3 py-1.5 text-xs flex items-center justify-between"
        style={{ backgroundColor: 'hsl(var(--code-header-bg))', color: 'hsl(var(--muted-foreground))' }}
      >
        <span>{language || 'text'}</span>
        <button 
          onClick={handleCopy}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-muted rounded"
        >
          {copied ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
        </button>
      </div>
      <pre className="p-3 overflow-x-auto">
        <code 
          ref={codeRef} 
          className={cn("text-sm font-mono", language && `language-${language}`)} 
          style={{ color: 'hsl(var(--code-text))' }}
          {...props}
        >
          {children}
        </code>
      </pre>
    </div>
  );
}

export function Markdown({ content, toolCalls, isStreaming }: MarkdownProps) {
  // Parse content into segments (thinking blocks and text)
  // Tool calls are rendered based on their position in the content stream
  const segments = useMemo(() => {
    const result: Array<{ type: string; content: string; position: number; data?: any }> = [];
    let remaining = content;
    let currentPosition = 0;
    
    // Sort tool calls by position
    const sortedToolCalls = toolCalls 
      ? [...toolCalls].sort((a, b) => (a.position || 0) - (b.position || 0))
      : [];
    let toolCallIndex = 0;
    
    // Helper to insert any tool calls that should appear before a given position
    const insertToolCallsBefore = (pos: number) => {
      while (toolCallIndex < sortedToolCalls.length) {
        const tc = sortedToolCalls[toolCallIndex];
        const tcPos = tc.position || 0;
        if (tcPos <= pos) {
          result.push({
            type: 'tool_call',
            content: '',
            position: tcPos,
            data: tc
          });
          toolCallIndex++;
        } else {
          break;
        }
      }
    };
    
    // Process thinking blocks from content
    while (remaining.length > 0) {
      const thinkMatch = remaining.match(/<think>([\s\S]*?)(<\/think>|$)/);
      
      if (!thinkMatch) {
        // Insert any tool calls before final text
        insertToolCallsBefore(currentPosition + remaining.length);
        // No more thinking blocks - add remaining text
        if (remaining.trim()) {
          result.push({ type: 'text', content: remaining.trim(), position: currentPosition });
        }
        break;
      }
      
      const thinkIndex = remaining.indexOf(thinkMatch[0]);
      
      // Add text before thinking block (and any tool calls in that range)
      if (thinkIndex > 0) {
        insertToolCallsBefore(currentPosition + thinkIndex);
        const textBefore = remaining.slice(0, thinkIndex).trim();
        if (textBefore) {
          result.push({ type: 'text', content: textBefore, position: currentPosition });
        }
      }
      
      currentPosition += thinkIndex;
      
      // Insert tool calls before thinking block
      insertToolCallsBefore(currentPosition);
      
      // Add thinking block
      const thinkContent = thinkMatch[1].trim();
      const isComplete = thinkMatch[2] === '</think>';
      result.push({ 
        type: isComplete ? 'thinking' : 'streaming-thinking', 
        content: thinkContent,
        position: currentPosition
      });
      
      currentPosition += thinkMatch[0].length;
      remaining = remaining.slice(thinkIndex + thinkMatch[0].length);
    }
    
    // Insert any remaining tool calls at the end
    while (toolCallIndex < sortedToolCalls.length) {
      const tc = sortedToolCalls[toolCallIndex];
      result.push({
        type: 'tool_call',
        content: '',
        position: tc.position || content.length,
        data: tc
      });
      toolCallIndex++;
    }
    
    return result;
  }, [content, toolCalls]);

  return (
    <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-2 prose-headings:my-3">
      {segments.map((segment, i) => {
        if (segment.type === 'thinking') return <ThinkingBlock key={i} content={segment.content} />;
        if (segment.type === 'streaming-thinking') return <ThinkingBlock key={i} content={segment.content} isStreaming />;
        if (segment.type === 'tool_call' && segment.data) {
          return <ToolCallBlock key={i} toolCall={segment.data} />;
        }
        
        return (
          <ReactMarkdown
            key={i}
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              code: CodeBlock,
              pre: ({ children }) => <>{children}</>,
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">{children}</a>
              ),
              img: ({ src, alt }) => (
                <img 
                  src={src} 
                  alt={alt || ''} 
                  className="max-w-full h-auto rounded-lg my-2"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
              ),
            }}
          >
            {segment.content}
          </ReactMarkdown>
        );
      })}
      {isStreaming && segments.length > 0 && segments[segments.length - 1].type === 'text' && (
        <span className="inline-block w-2.5 h-2.5 ml-1 rounded-full bg-primary animate-blob-breathe align-middle" />
      )}
    </div>
  );
}
