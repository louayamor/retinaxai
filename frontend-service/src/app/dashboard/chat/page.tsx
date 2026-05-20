'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { AIChartRenderer } from '@/components/charts/ai-chart-renderer';
import {
  createChatSession,
  deleteChatSession,
  getChatSession,
  listChatSessions,
  sendChatMessage,
} from '@/lib/api';
import type {
  ChatMessage,
  ChatSessionDetail,
  ChatSessionItem,
} from '@/lib/api';
import { useChatWebSocket } from '@/hooks/use-chat-websocket';
import {
  MessageSquare,
  Plus,
  Trash2,
  Send,
  Loader2,
  Sparkles,
  BarChart3,
  User,
  Stethoscope,
  Brain,
  Search,
} from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

export default function ChatPage() {
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessionDetail, setSessionDetail] = useState<ChatSessionDetail | null>(null);
  const [input, setInput] = useState('');
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);

  const chatWs = useChatWebSocket();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const data = await listChatSessions();
      setSessions(data.sessions);
    } catch {
      toast.error('Failed to load conversations');
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  const loadSession = useCallback(async (id: string) => {
    setLoadingMessages(true);
    try {
      const detail = await getChatSession(id);
      setSessionDetail(detail);
      setActiveSessionId(id);
    } catch {
      toast.error('Failed to load conversation');
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  const handleNewChat = useCallback(async () => {
    try {
      const { session_id: sessionId } = await createChatSession();
      setSessions((prev) => [
        {
          id: sessionId,
          title: 'New Chat',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          message_count: 0,
        },
        ...prev,
      ]);
      setActiveSessionId(sessionId);
      setSessionDetail({
        id: sessionId,
        title: 'New Chat',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        message_count: 0,
        messages: [],
      });
    } catch {
      toast.error('Failed to create new chat');
    }
  }, []);

  const handleDeleteSession = useCallback(
    async (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      try {
        await deleteChatSession(id);
        setSessions((prev) => prev.filter((s) => s.id !== id));
        if (activeSessionId === id) {
          setActiveSessionId(null);
          setSessionDetail(null);
        }
        toast.success('Conversation deleted');
      } catch {
        toast.error('Failed to delete conversation');
      }
    },
    [activeSessionId],
  );

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || chatWs.status === 'connecting' || chatWs.status === 'sending') return;

    let sessionId = activeSessionId;
    if (!sessionId) {
      try {
        const { session_id: newId } = await createChatSession();
        sessionId = newId;
        setActiveSessionId(newId);
        setSessions((prev) => [
          {
            id: newId,
            title: text.length > 50 ? text.slice(0, 50) + '...' : text,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            message_count: 0,
          },
          ...prev,
        ]);
      } catch {
        toast.error('Failed to create chat session');
        return;
      }
    }

    setInput('');

    const userMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
    };

    setSessionDetail((prev) => {
      if (!prev) return prev;
      return { ...prev, messages: [...prev.messages, userMsg] };
    });

    const history = sessionDetail?.messages ?? [];

    try {
      const final = await chatWs.send(text, history);
      if (final.error) {
        toast.error(final.error);
        return;
      }

      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: final.summary,
        chart: final.chart ?? undefined,
        sources: final.sources.length > 0 ? final.sources : undefined,
        created_at: new Date().toISOString(),
      };

      setSessionDetail((prev) => {
        if (!prev) return prev;
        const msgs = [...prev.messages, assistantMsg];
        return { ...prev, messages: msgs };
      });

      try {
        await sendChatMessage(sessionId, text);
      } catch {
        // REST fallback for persistence — non-critical
      }
    } catch {
      toast.error('Failed to get response. The AI service may be unavailable.');
      setSessionDetail((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          messages: prev.messages.filter((m) => m.id !== userMsg.id),
        };
      });
    }
  }, [input, chatWs, activeSessionId, sessionDetail]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        void handleSend();
      }
    },
    [handleSend],
  );

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    scrollToBottom();
  }, [sessionDetail?.messages, scrollToBottom]);

  return (
    <PageContainer className="h-[calc(100vh-8rem)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Stethoscope className="h-5 w-5 text-[var(--brand-teal)]" />
          <div>
            <h1 className="text-lg font-bold tracking-tight">AI Assistant</h1>
            <p className="text-xs text-muted-foreground">
              Ask about DR, model performance, patient data, or clinical findings
            </p>
          </div>
        </div>
        <Button onClick={handleNewChat} size="sm">
          <Plus className="mr-1.5 h-4 w-4" />
          New Chat
        </Button>
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Session Sidebar */}
        <div className="w-56 shrink-0 flex flex-col border rounded-lg bg-card">
          <div className="p-3 border-b">
            <p className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
              <MessageSquare className="h-3.5 w-3.5" />
              Conversations
            </p>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {loadingSessions ? (
              Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))
            ) : sessions.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-8">
                No conversations yet
              </p>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => { void loadSession(s.id); }}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); void loadSession(s.id); } }}
                  role="button"
                  tabIndex={0}
                  className={cn(
                    'w-full text-left p-2 rounded-md text-sm transition-colors group cursor-pointer',
                    activeSessionId === s.id
                      ? 'bg-accent text-accent-foreground'
                      : 'hover:bg-muted',
                  )}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate text-xs leading-tight">{s.title}</span>
                    <button
                      onClick={(e) => { void handleDeleteSession(s.id, e); }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive shrink-0"
                      title="Delete"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {s.message_count} msgs
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col border rounded-lg bg-card min-w-0">
          {!activeSessionId && !loadingMessages ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 p-8">
              <Sparkles className="h-12 w-12 text-muted-foreground opacity-30" />
              <div className="text-center max-w-sm">
                <h3 className="font-semibold mb-1">AI Assistant</h3>
                <p className="text-sm text-muted-foreground">
                  Ask anything about diabetic retinopathy, patient statistics,
                  model performance, or clinical data.
                </p>
              </div>
              <div className="grid gap-2 w-full max-w-sm">
                {[
                  'What is the DR severity distribution?',
                  'How accurate is the imaging model?',
                  'What are the top clinical features?',
                  'Show me model performance metrics.',
                ].map((q) => (
                  <Button
                    key={q}
                    variant="outline"
                    size="sm"
                    className="text-xs justify-start h-auto py-2 px-3"
                    onClick={() => {
                      setInput(q);
                      setTimeout(() => {
                        inputRef.current?.focus();
                      }, 50);
                    }}
                  >
                    {q}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {loadingMessages ? (
                  <div className="space-y-4">
                    {Array.from({ length: 3 }).map((_, i) => (
                      <div key={i} className={cn('flex', i % 2 === 0 ? 'justify-end' : 'justify-start')}>
                        <Skeleton className={cn(
                          'rounded-lg',
                          i % 2 === 0 ? 'h-10 w-48' : 'h-20 w-64',
                        )} />
                      </div>
                    ))}
                  </div>
                ) : (
                  sessionDetail?.messages.map((msg) => (
                    <ChatBubble key={msg.id} message={msg} />
                  ))
                )}
                {(chatWs.status === 'connecting' || chatWs.status === 'sending') && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-muted">
                      {chatWs.thinking && chatWs.thinking.stage === 'retrieving' ? (
                        <Search className="h-3.5 w-3.5 animate-pulse text-muted-foreground" />
                      ) : (
                        <Brain className="h-3.5 w-3.5 animate-pulse text-muted-foreground" />
                      )}
                      <div className="flex flex-col gap-0.5">
                        <span className="text-xs text-muted-foreground">
                          {chatWs.thinking?.message ?? 'Thinking...'}
                        </span>
                        <span className="text-[10px] text-muted-foreground/60 capitalize">
                          {chatWs.thinking?.stage ?? ''}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                {chatWs.status === 'error' && chatWs.error && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-destructive/10 text-destructive text-xs">
                      Connection failed. Please try again.
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-3 border-t">
                <div className="flex gap-2">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about DR, model performance, or clinical data..."
                    rows={2}
                    disabled={chatWs.status === 'connecting' || chatWs.status === 'sending'}
                    className="flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
                  />
                  <Button
                    onClick={() => { void handleSend(); }}
                    disabled={!input.trim() || chatWs.status === 'connecting' || chatWs.status === 'sending'}
                    size="icon"
                    className="shrink-0"
                  >
                    {chatWs.status === 'connecting' || chatWs.status === 'sending' ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </PageContainer>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const hasChart = message.chart && message.chart.data?.length > 0;
  const hasSources = message.sources && message.sources.length > 0;

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className="flex gap-2 max-w-[85%]">
        {!isUser && (
          <div className="shrink-0 mt-1">
            <Sparkles className="h-5 w-5 text-[var(--brand-teal)]" />
          </div>
        )}
        <div
          className={cn(
            'rounded-lg px-4 py-3',
            isUser
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted',
          )}
        >
          <div className="text-sm whitespace-pre-wrap leading-relaxed">
            {message.content}
          </div>

          {hasChart && (
            <div className="mt-3 rounded border bg-card/50 p-3">
              <div className="flex items-center gap-1.5 mb-2">
                <BarChart3 className="h-3 w-3 text-muted-foreground" />
                <span className="text-xs font-medium">
                  {message.chart!.title}
                </span>
              </div>
              <AIChartRenderer spec={message.chart!} height={180} />
            </div>
          )}

          {hasSources && (
            <details className="mt-2 text-xs">
              <summary className={cn(
                'cursor-pointer transition-colors',
                isUser ? 'text-primary-foreground/70 hover:text-primary-foreground' : 'text-muted-foreground hover:text-foreground',
              )}>
                {message.sources!.length} source{message.sources!.length !== 1 ? 's' : ''}
              </summary>
              <div className="mt-1.5 space-y-1">
                {message.sources!.map((src, i) => (
                  <div key={i} className="rounded border bg-background/50 px-2 py-1">
                    <span className="font-mono text-xs text-[var(--brand-teal)]">
                      {src.artifact_id}
                    </span>
                    <p className="text-muted-foreground leading-relaxed mt-0.5">
                      {src.snippet}
                    </p>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
        {isUser && (
          <div className="shrink-0 mt-1">
            <User className="h-5 w-5 text-muted-foreground" />
          </div>
        )}
      </div>
    </div>
  );
}
