import { useState, useCallback, useEffect, useRef } from 'react';
import type { Agent, Message, ToolCall } from '@/lib/types';
import { api } from '@/lib/api';

// Helper to format common result objects
function formatResultObject(obj: any): string {
  if (!obj || typeof obj !== 'object') {
    return typeof obj === 'string' ? obj : JSON.stringify(obj);
  }
  
  // Handle tools list (from /mcp tools)
  if (Array.isArray(obj.tools)) {
    let md = `**${obj.tools.length} tools available:**\n\n`;
    for (const tool of obj.tools) {
      md += `- **${tool.name}** (${tool.server || 'local'})\n`;
      if (tool.description) {
        md += `  ${tool.description}\n`;
      }
    }
    return md;
  }
  
  // Handle sessions list (from /session history)
  if (Array.isArray(obj.sessions)) {
    if (obj.sessions.length === 0) {
      return '*No sessions found*';
    }
    let md = `**${obj.sessions.length} sessions:**\n\n`;
    for (const s of obj.sessions) {
      const date = s.created_at ? new Date(s.created_at).toLocaleString() : '';
      md += `- \`${s.session_id?.slice(0, 8) || 'unknown'}\` - ${s.message_count || 0} messages`;
      if (date) md += ` (${date})`;
      md += '\n';
    }
    return md;
  }
  
  // Handle servers list (from /mcp servers)
  if (Array.isArray(obj.servers)) {
    if (obj.servers.length === 0) {
      return '*No MCP servers connected*';
    }
    let md = `**${obj.servers.length} servers:**\n\n`;
    for (const s of obj.servers) {
      const name = typeof s === 'string' ? s : s.name || s.id || 'unknown';
      md += `- **${name}**\n`;
    }
    return md;
  }
  
  // Handle prompts list (from /mcp prompts)
  if (Array.isArray(obj.prompts)) {
    if (obj.prompts.length === 0) {
      return '*No prompts available*';
    }
    let md = `**${obj.prompts.length} prompts:**\n\n`;
    for (const p of obj.prompts) {
      md += `- **${p.name}**`;
      if (p.description) md += ` - ${p.description}`;
      md += '\n';
    }
    return md;
  }
  
  // Handle resources list (from /mcp resources)
  if (Array.isArray(obj.resources)) {
    if (obj.resources.length === 0) {
      return '*No resources available*';
    }
    let md = `**${obj.resources.length} resources:**\n\n`;
    for (const r of obj.resources) {
      md += `- **${r.name || r.uri || 'unknown'}**\n`;
    }
    return md;
  }
  
  // Handle display field
  if (obj.display) {
    return obj.display;
  }
  
  // Handle success message
  if (obj.success === true || obj.status === 'ok') {
    const msg = obj.message || obj.msg || 'Done';
    return `✓ ${msg}`;
  }
  
  // Fallback to JSON
  return '```json\n' + JSON.stringify(obj, null, 2) + '\n```';
}

export function useChat(agent: Agent | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef(false);

  useEffect(() => {
    console.log('[useChat] Agent changed to:', agent?.name);
    setMessages([]);
    abortRef.current = false;
    
    // Try to auto-load most recent session
    if (agent) {
      console.log('[useChat] Loading session for:', agent.name);
      api.getRecentSession(agent.name).then(session => {
        console.log('[useChat] Got session:', session);
        if (session && session.messages.length > 0) {
          // Merge consecutive assistant/tool exchanges into single UI messages
          // This mirrors how streaming displays them as one turn
          const loadedMessages: Message[] = [];
          let i = 0;
          
          while (i < session.messages.length) {
            const m = session.messages[i];
            
            if (m.role === 'user' || m.role === 'system') {
              loadedMessages.push({
                id: `loaded-${i}`,
                role: m.role as 'user' | 'system',
                content: m.content ?? '',
              });
              i++;
            } else if (m.role === 'assistant') {
              // Collect all consecutive assistant/tool messages into one UI message
              const allToolCalls: ToolCall[] = [];
              let combinedContent = '';
              
              while (i < session.messages.length) {
                const current = session.messages[i];
                
                if (current.role === 'assistant') {
                  // Append content and track position for tool calls
                  if (current.content) {
                    combinedContent += current.content;
                  }
                  
                  // Collect tool calls with position = current content length
                  // This places them after the content that preceded them
                  if (current.tool_calls && Array.isArray(current.tool_calls)) {
                    for (const tc of current.tool_calls) {
                      allToolCalls.push({
                        id: `tc-${allToolCalls.length}`,
                        name: tc.function?.name || tc.name || 'unknown',
                        args: typeof tc.function?.arguments === 'string' 
                          ? JSON.parse(tc.function.arguments || '{}')
                          : (tc.function?.arguments || tc.args || {}),
                        result: undefined, // Will be filled from following tool message
                        status: 'success' as const,
                        position: combinedContent.length, // Position in content stream
                      });
                    }
                  }
                  i++;
                } else if (current.role === 'tool') {
                  // Match tool result to the most recent unmatched tool call
                  const pendingToolCall = allToolCalls.find(tc => tc.result === undefined);
                  if (pendingToolCall) {
                    pendingToolCall.result = current.content || '';
                  }
                  i++;
                } else {
                  // Hit a user message, stop collecting
                  break;
                }
              }
              
              loadedMessages.push({
                id: `loaded-assistant-${loadedMessages.length}`,
                role: 'assistant',
                content: combinedContent,
                toolCalls: allToolCalls.length > 0 ? allToolCalls : undefined,
              });
            } else if (m.role === 'tool') {
              // Orphan tool message (shouldn't happen), skip
              i++;
            } else {
              i++;
            }
          }
          
          console.log('[useChat] Setting messages:', loadedMessages.length);
          setMessages(loadedMessages);
        }
      }).catch((e) => {
        console.error('[useChat] Error loading session:', e);
      });
    }
  }, [agent?.name]);

  const addSystemMessage = useCallback((content: string) => {
    const msg: Message = {
      id: crypto.randomUUID(),
      role: 'system',
      content,
    };
    setMessages(prev => [...prev, msg]);
  }, []);

  const sendMessage = useCallback(async (content: string, attachments?: any[]) => {
    if (!agent || isStreaming) return;

    const userMsg: Message = { 
      id: crypto.randomUUID(), 
      role: 'user', 
      content,
      attachments 
    };
    const assistantMsg: Message = { 
      id: crypto.randomUUID(), 
      role: 'assistant', 
      content: '',
      toolCalls: []
    };
    
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);
    abortRef.current = false;

    try {
      const history = [...messages, userMsg].map(m => ({ role: m.role, content: m.content }));
      let fullContent = '';
      const toolCalls: ToolCall[] = [];

      for await (const event of api.chatStream(agent.name, history)) {
        if (abortRef.current) break;
        
        if (event.type === 'content' && event.content) {
          fullContent += event.content;
          
          // Mark any 'running' tool calls as complete when we get content after them
          // (This means the agentic loop has moved past them)
          // Real results come from tool_result events - this just updates status
          for (const tc of toolCalls) {
            if (tc.status === 'running') {
              tc.status = 'success';
            }
          }
          
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...assistantMsg, content: fullContent, toolCalls: [...toolCalls] };
            return updated;
          });
        }
        
        if (event.type === 'tool_call' && event.toolCall) {
          const tc: ToolCall = {
            id: event.toolCall.id,
            name: event.toolCall.name || 'unknown',
            args: {},
            status: 'running',  // Start as running
            position: fullContent.length  // Track position in content stream
          };
          
          try {
            tc.args = JSON.parse(event.toolCall.arguments || '{}');
          } catch {
            tc.args = { raw: event.toolCall.arguments };
          }
          
          toolCalls.push(tc);
          
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...assistantMsg, content: fullContent, toolCalls: [...toolCalls] };
            return updated;
          });
        }
        
        // Handle tool result - update existing tool call with result
        // Note: Match by ID AND no result yet (handles duplicate IDs like "call_0")
        if (event.type === 'tool_result' && event.toolResult) {
          const tr = event.toolResult;
          // Find first tool call with matching ID that doesn't have a result yet
          const existingTc = toolCalls.find(tc => tc.id === tr.id && tc.result === undefined);
          if (existingTc) {
            existingTc.status = tr.status === 'error' ? 'error' : 'success';
            existingTc.result = tr.result;
            
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { ...assistantMsg, content: fullContent, toolCalls: [...toolCalls] };
              return updated;
            });
          }
        }
      }
      // Session persistence handled server-side by SessionManagerSkill hooks
    } catch (error) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...assistantMsg,
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  }, [agent, messages, isStreaming]);

  const executeCommand = useCallback(async (commandInput: string): Promise<boolean> => {
    if (!agent) return false;

    const result = await api.executeCommand(agent.name, commandInput);
    // Format display: /session load abc -> /session load abc
    const displayPath = commandInput.startsWith('/') ? commandInput.trim() : '/' + commandInput.trim();
    
    if (result.success) {
      const output = result.output;
      let formattedOutput: string;
      
      if (typeof output === 'object' && output !== null) {
        if (output.display) {
          formattedOutput = output.display;
        } else if (output.result) {
          formattedOutput = formatResultObject(output.result);
        } else {
          formattedOutput = formatResultObject(output);
        }
      } else if (typeof output === 'string') {
        formattedOutput = output;
      } else {
        formattedOutput = '```json\n' + JSON.stringify(output, null, 2) + '\n```';
      }
      
      addSystemMessage(`**Command:** \`${displayPath}\`\n\n${formattedOutput}`);
    } else {
      addSystemMessage(`**Command:** \`${displayPath}\`\n\n**Error:** ${result.error}`);
    }
    
    return result.success;
  }, [agent, addSystemMessage]);

  const clearMessages = useCallback(() => setMessages([]), []);
  const stopStreaming = useCallback(() => { abortRef.current = true; }, []);

  return { messages, sendMessage, executeCommand, isStreaming, clearMessages, stopStreaming, addSystemMessage };
}
