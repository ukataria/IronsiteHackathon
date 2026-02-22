import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2 } from 'lucide-react';
import * as api from '@/api';

type Condition = 'anchor_calibrated' | 'baseline' | 'depth' | 'all';
type Vlm       = 'claude' | 'gpt4o' | 'ollama';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text?: string;
  answers?: api.ConditionAnswer[];
  frameId: string;
}

interface Props {
  frameId: string | null;
  condition: Condition;
  vlm: Vlm;
}

const SUGGESTED = [
  'What is the stud spacing here?',
  'Are there any code violations?',
  'How high is the electrical box from the floor?',
];

const CONDITION_LABELS: Record<string, string> = {
  baseline:          'Baseline',
  depth:             'Depth-Augmented',
  anchor_calibrated: 'DeepAnchored',
};

const VERDICT_CLS: Record<string, string> = {
  PASS:    'text-success',
  FAIL:    'text-destructive',
  UNKNOWN: 'text-muted-foreground',
};

export function LiveChat({ frameId, condition, vlm }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input,    setInput]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const send = useCallback(
    async (question: string) => {
      if (!frameId || !question.trim() || loading) return;
      const q = question.trim();
      setInput('');

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        text: q,
        frameId,
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const res = await api.askQuestion(frameId, q, condition, vlm);
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: 'assistant', answers: res.answers, frameId },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            text: `Error: ${err instanceof Error ? err.message : String(err)}`,
            frameId,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [frameId, condition, vlm, loading],
  );

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(input); }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* ── Messages ─────────────────────────────────────────────────── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">

        {/* Empty state */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 pb-8">
            <p className="text-muted-foreground text-xs text-center">
              {frameId ? 'Ask about this frame…' : 'Select a frame to start asking questions.'}
            </p>
            {frameId && (
              <div className="flex flex-col gap-2 w-full max-w-xs mt-2">
                {SUGGESTED.map((q) => (
                  <button
                    key={q}
                    onClick={() => void send(q)}
                    className="text-left text-xs px-3 py-2 rounded border border-border bg-card hover:border-primary hover:text-primary transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Message bubbles */}
        {messages.map((msg) =>
          msg.role === 'user' ? (
            <div key={msg.id} className="flex justify-end">
              <div className="max-w-[80%] px-3 py-2 rounded-xl bg-primary text-primary-foreground text-sm leading-relaxed">
                {msg.text}
              </div>
            </div>
          ) : (
            <div key={msg.id} className="flex justify-start w-full">
              <div className="w-full space-y-2">
                {/* Plain error / fallback text */}
                {msg.text && (
                  <div className="px-3 py-2 rounded-xl bg-card border border-border text-sm text-muted-foreground">
                    {msg.text}
                  </div>
                )}

                {/* Condition cards */}
                {msg.answers?.map((a) => (
                  <div
                    key={a.condition}
                    className={`rounded-xl border overflow-hidden ${
                      a.condition === 'anchor_calibrated'
                        ? 'border-primary/60'
                        : 'border-border'
                    }`}
                  >
                    {/* Card header */}
                    <div
                      className={`flex items-center justify-between px-3 py-2 border-b ${
                        a.condition === 'anchor_calibrated'
                          ? 'border-primary/20 bg-primary/5'
                          : 'border-border bg-card'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold">
                          {CONDITION_LABELS[a.condition] ?? a.condition}
                        </span>
                        {a.condition === 'anchor_calibrated' && (
                          <span className="text-[9px] bg-primary text-primary-foreground px-1.5 py-0.5 rounded font-bold tracking-wide">
                            ★ DEEPANCHORED
                          </span>
                        )}
                      </div>
                      <span className={`text-[10px] font-mono font-bold ${VERDICT_CLS[a.verdict] ?? ''}`}>
                        {a.verdict}
                      </span>
                    </div>

                    {/* Response body */}
                    <div className="px-3 py-2.5 text-[11px] leading-relaxed whitespace-pre-wrap font-mono text-muted-foreground bg-card">
                      {a.response}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ),
        )}

        {/* Typing indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="px-3 py-2 rounded-xl bg-card border border-border">
              <Loader2 size={14} className="animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
      </div>

      {/* ── Input bar ────────────────────────────────────────────────── */}
      <div className="p-3 border-t border-border shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={!frameId || loading}
            placeholder={
              frameId
                ? "Ask about this frame — e.g. 'What is the stud spacing?'"
                : 'Select a frame first…'
            }
            className="flex-1 bg-secondary text-foreground text-xs px-3 py-2 rounded border border-border focus:outline-none focus:border-primary placeholder:text-muted-foreground font-mono disabled:opacity-50"
          />
          <button
            onClick={() => void send(input)}
            disabled={!frameId || !input.trim() || loading}
            className="flex items-center justify-center w-8 h-8 rounded bg-primary text-primary-foreground disabled:opacity-40 hover:opacity-90 transition-opacity shrink-0"
          >
            <Send size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}
