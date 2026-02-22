import { useEffect, useRef, useState } from 'react';
import type { CalibrationData, Finding } from './types';

function useCountUp(target: number, duration = 400) {
  const [display, setDisplay] = useState(target);
  const prevRef = useRef(target);
  const rafRef = useRef(0);

  useEffect(() => {
    const start = prevRef.current;
    const diff = target - start;
    if (Math.abs(diff) < 0.001) { prevRef.current = target; return; }

    const t0 = performance.now();
    const animate = (time: number) => {
      const p = Math.min(1, (time - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(start + diff * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(animate);
      else prevRef.current = target;
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration]);

  return display;
}

interface Props {
  calibration: CalibrationData | null;
  currentFrame: number;
  totalFrames: number;
  findings: Finding[];
  confidenceHistory: number[];
}

export function CalibrationPanel({ calibration, currentFrame, totalFrames, findings, confidenceHistory }: Props) {
  const ppi = useCountUp(calibration?.pixels_per_inch ?? 0);
  const conf = useCountUp((calibration?.confidence ?? 0) * 100);

  const violations = findings.filter(f => f.type === 'violation');
  const compliant = findings.filter(f => f.type === 'compliant');
  const studViolations = violations.filter(f => f.label.includes('Stud'));
  const boxViolations = violations.filter(f => f.label.includes('box') || f.label.includes('Box'));

  // Sparkline
  const sparkPoints = confidenceHistory.length > 1
    ? confidenceHistory.map((c, i) => {
        const x = (i / (confidenceHistory.length - 1)) * 100;
        const y = 28 - (c - 0.8) * (28 / 0.2); // map 0.8-1.0 → 28-0
        return `${x},${Math.max(0, Math.min(28, y))}`;
      }).join(' ')
    : '';

  return (
    <div className="flex flex-col h-full p-3 text-xs overflow-y-auto">
      {/* Spatial Calibration */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <span className="panel-header">Spatial Calibration</span>
          <span className="status-dot-live" />
          <span className="text-[10px] text-primary font-medium">LIVE</span>
        </div>

        <div className="panel-divider" />

        <div className="space-y-2 mb-3">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Scale</span>
            <span className="font-mono font-bold text-foreground">{ppi.toFixed(2)} <span className="text-muted-foreground font-normal">px / in</span></span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-mono font-bold text-foreground">{conf.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-start">
            <span className="text-muted-foreground">Method</span>
            <span className="font-mono text-foreground text-right whitespace-pre-line text-[10px]">
              {calibration?.method_label ?? '—'}
            </span>
          </div>
        </div>

        {/* Sparkline */}
        {sparkPoints && (
          <div className="mb-3">
            <svg viewBox="0 0 100 30" className="w-full h-6">
              <polyline
                points={sparkPoints}
                fill="none"
                stroke="hsl(25, 95%, 53%)"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        )}
      </div>

      {/* Inspection Summary */}
      <div className="mt-auto">
        <div className="panel-header mb-3">Inspection Summary</div>
        <div className="panel-divider" />

        <div className="space-y-1.5">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Frames analyzed</span>
            <span className="font-mono text-foreground">
              {String(currentFrame + 1).padStart(3, '0')} / {totalFrames}
            </span>
          </div>

          <div className="flex justify-between">
            <span className="text-muted-foreground">Violations found</span>
            <span className="font-mono font-bold text-destructive">{violations.length}</span>
          </div>

          {studViolations.length > 0 && (
            <div className="flex justify-between pl-4">
              <span className="text-muted-foreground">Stud spacing</span>
              <span className="font-mono text-muted-foreground">{studViolations.length}</span>
            </div>
          )}
          {boxViolations.length > 0 && (
            <div className="flex justify-between pl-4">
              <span className="text-muted-foreground">Box height</span>
              <span className="font-mono text-muted-foreground">{boxViolations.length}</span>
            </div>
          )}

          <div className="flex justify-between">
            <span className="text-muted-foreground">Compliant frames</span>
            <span className="font-mono text-success">{compliant.length}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
