import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import type { CalibrationData, QAExchange } from './types';
import { sendChatMessage } from '@/api';

interface Props {
  imageId: string | null;
  calibration: CalibrationData | null;
  vlmResponse: string | null;
}

export function SpatialQA({ imageId, calibration, vlmResponse }: Props) {
  const [exchanges, setExchanges] = useState<QAExchange[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Reset chat when frame changes
  useEffect(() => {
    setExchanges([]);
  }, [imageId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [exchanges.length]);

  const handleSubmit = useCallback(async () => {
    const q = input.trim();
    if (!q || !calibration || !imageId || loading) return;

    setInput('');
    setLoading(true);

    // Optimistically add the question
    setExchanges(prev => [...prev, { question: q, answer: '…' }]);

    const answer = await sendChatMessage(imageId, q, exchanges);

    setExchanges(prev => [
      ...prev.slice(0, -1),
      { question: q, answer },
    ]);
    setLoading(false);
  }, [input, calibration, imageId, loading, exchanges]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 pt-3 pb-2">
        <span className="panel-header">Ask PreCheck</span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 space-y-3">
        {exchanges.length === 0 && (
          <div className="flex flex-col items-center gap-2 mt-4">
            <p className="text-muted-foreground text-[10px] text-center">
              Ask about any measurement in this scene…
            </p>
            {vlmResponse && (
              <button
                onClick={() => setExchanges([{ question: '', answer: vlmResponse }])}
                className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80"
              >
                Show full inspection report
              </button>
            )}
          </div>
        )}
        {exchanges.map((ex, i) => (
          <div key={i} className="space-y-1.5">
            {ex.question && (
              <div>
                <span className="text-[10px] font-semibold tracking-wider text-muted-foreground">YOU</span>
                <p className="text-xs text-foreground mt-0.5">{ex.question}</p>
              </div>
            )}
            <div>
              <span className="text-[10px] font-semibold tracking-wider text-primary">PRECHECK</span>
              <div className="text-[11px] text-foreground mt-0.5 leading-snug
                [&_h1]:text-xs [&_h1]:font-bold [&_h1]:mt-1.5 [&_h1]:mb-0.5
                [&_h2]:text-[11px] [&_h2]:font-semibold [&_h2]:mt-1.5 [&_h2]:mb-0.5 [&_h2]:text-primary
                [&_h3]:text-[11px] [&_h3]:font-semibold [&_h3]:mt-1 [&_h3]:mb-0
                [&_p]:my-0.5
                [&_strong]:font-semibold
                [&_table]:w-full [&_table]:text-[10px] [&_table]:border-collapse [&_table]:my-1
                [&_th]:text-left [&_th]:px-1 [&_th]:py-px [&_th]:border [&_th]:border-border [&_th]:bg-secondary [&_th]:font-semibold
                [&_td]:px-1 [&_td]:py-px [&_td]:border [&_td]:border-border
                [&_ul]:my-0.5 [&_ul]:pl-3 [&_ol]:my-0.5 [&_ol]:pl-3 [&_li]:my-0
                [&_hr]:border-border [&_hr]:my-1
                [&_code]:bg-secondary [&_code]:px-0.5 [&_code]:rounded [&_code]:text-[10px]">
                <ReactMarkdown>{ex.answer}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-border">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          placeholder={loading ? 'Thinking…' : 'Ask about any measurement in this scene…'}
          disabled={loading}
          className="w-full bg-secondary text-foreground text-xs px-3 py-2 rounded border border-border focus:outline-none focus:border-primary placeholder:text-muted-foreground font-mono disabled:opacity-50"
        />
      </div>
    </div>
  );
}
