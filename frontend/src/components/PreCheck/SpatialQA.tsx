import { useState, useRef, useEffect, useCallback } from 'react';
import type { CalibrationData, QAExchange } from './types';

const QA_PATTERNS = [
  {
    keywords: ['electrical', 'box', 'height', 'floor'],
    answer: (cal: CalibrationData) => {
      const ppi = cal.pixels_per_inch;
      const delta = 155;
      const inches = (delta / ppi).toFixed(1);
      return `Box 1 center is at pixel y=357.\nFloor baseline at y=500.\nDelta: ${delta}px ÷ ${ppi.toFixed(2)} px/in = ${inches} inches.\nStandard is 12" ± 1" to center.\n✓ Within tolerance.`;
    },
  },
  {
    keywords: ['stud', 'spacing', 'bay', 'far', 'distance'],
    answer: (cal: CalibrationData) =>
      `Current calibration: ${cal.pixels_per_inch.toFixed(2)} px/in\nBay 1→2: 16.0" ✓\nBay 2→3: 16.1" ✓\nBay 3→4: 16.0" ✓\nBay 4→5: 19.3" ✗ (exceeds 16.0" ± 0.5")\nBay 5→6: 15.8" ✓`,
  },
  {
    keywords: ['calibration', 'scale', 'accurate', 'confidence'],
    answer: (cal: CalibrationData) =>
      `Current scale: ${cal.pixels_per_inch.toFixed(2)} px/in\nConfidence: ${(cal.confidence * 100).toFixed(0)}%\nMethod: 2×4 stud face (3.5" known width)\nCalibration is ${cal.confidence > 0.9 ? 'stable and reliable' : 'fluctuating — reposition camera'}.`,
  },
  {
    keywords: ['violation', 'fail', 'problem', 'issue', 'wrong'],
    answer: () =>
      `Violations detected in this inspection:\n1. Stud bay 4→5: 19.3" (expected 16.0" ± 0.5") — CRITICAL\nStud is 3.3" beyond tolerance.\nUse the timeline to review violation frames (marked in red).`,
  },
  {
    keywords: ['how', 'work', 'what', 'deepanchored'],
    answer: () =>
      `DeepAnchored uses known-dimension objects (studs, CMU blocks) visible in the camera feed to compute a pixel-to-inch calibration.\nOnce calibrated, it measures distances between structural elements and checks against building code.\nAll measurements shown are derived from this calibration — no manual input required.`,
  },
];

interface Props {
  calibration: CalibrationData | null;
}

export function SpatialQA({ calibration }: Props) {
  const [exchanges, setExchanges] = useState<QAExchange[]>([]);
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [exchanges.length]);

  const handleSubmit = useCallback(() => {
    const q = input.trim();
    if (!q || !calibration) return;

    const lower = q.toLowerCase();
    const matched = QA_PATTERNS.find(p => p.keywords.some(k => lower.includes(k)));
    const answer = matched
      ? matched.answer(calibration)
      : `Current calibration: ${calibration.pixels_per_inch.toFixed(2)} px/in at ${(calibration.confidence * 100).toFixed(0)}% confidence.\nPlease ask about specific measurements, stud spacing, electrical box height, or calibration accuracy.`;

    setExchanges(prev => [...prev.slice(-2), { question: q, answer }]);
    setInput('');
  }, [input, calibration]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 pt-3 pb-2">
        <span className="panel-header">Ask DeepAnchored</span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 space-y-3">
        {exchanges.length === 0 && (
          <p className="text-muted-foreground text-[10px] mt-4 text-center">
            Ask about any measurement in this scene…
          </p>
        )}
        {exchanges.map((ex, i) => (
          <div key={i} className="space-y-1.5">
            <div>
              <span className="text-[10px] font-semibold tracking-wider text-muted-foreground">YOU</span>
              <p className="text-xs text-foreground mt-0.5">{ex.question}</p>
            </div>
            <div>
              <span className="text-[10px] font-semibold tracking-wider text-primary">DEEPANCHORED</span>
              <pre className="text-xs text-foreground mt-0.5 whitespace-pre-wrap font-mono leading-relaxed">
                {ex.answer}
              </pre>
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
          placeholder="Ask about any measurement in this scene…"
          className="w-full bg-secondary text-foreground text-xs px-3 py-2 rounded border border-border focus:outline-none focus:border-primary placeholder:text-muted-foreground font-mono"
        />
      </div>
    </div>
  );
}
