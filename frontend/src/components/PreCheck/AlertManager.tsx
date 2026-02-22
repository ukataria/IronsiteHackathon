import { useEffect, useState } from 'react';
import type { AlertData } from './types';
import { X } from 'lucide-react';

interface Props {
  alerts: AlertData[];
  onDismiss: (id: string) => void;
  onJumpToFrame: (frameIndex: number) => void;
}

export function AlertManager({ alerts, onDismiss, onJumpToFrame }: Props) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 w-[340px] pointer-events-none">
      {alerts.map(alert => (
        <AlertToast
          key={alert.id}
          alert={alert}
          onDismiss={() => onDismiss(alert.id)}
          onJump={() => {
            onJumpToFrame(alert.frame_id);
            onDismiss(alert.id);
          }}
        />
      ))}
    </div>
  );
}

function AlertToast({ alert, onDismiss, onJump }: { alert: AlertData; onDismiss: () => void; onJump: () => void }) {
  const [fading, setFading] = useState(false);

  useEffect(() => {
    const fadeTimer = setTimeout(() => setFading(true), 2700);
    const removeTimer = setTimeout(onDismiss, 3000);
    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(removeTimer);
    };
  }, [onDismiss]);

  return (
    <div className={`alert-toast pointer-events-auto ${fading ? 'animate-fade-out' : ''}`}>
      <div className="flex items-start justify-between mb-2">
        <span className="font-semibold text-[11px] tracking-wide flex items-center gap-1.5">
          ⚠ MEASUREMENT ALERT
        </span>
        <button onClick={onDismiss} className="text-muted-foreground hover:text-foreground -mt-0.5">
          <X size={12} />
        </button>
      </div>

      <div className="border-t border-border/50 pt-2 space-y-1 font-mono text-[11px]">
        <div className="flex justify-between">
          <span className="text-muted-foreground">{alert.label}:</span>
          <span className="text-foreground">{alert.measured}"</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Expected:</span>
          <span className="text-foreground">{alert.expected}" ± {alert.tolerance}"</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Over by:</span>
          <span className="text-destructive font-bold">
            +{alert.delta.toFixed(1)}" [{alert.severity.toUpperCase()}]
          </span>
        </div>
      </div>

      <button
        onClick={onJump}
        className="mt-2 text-[10px] font-medium text-primary hover:underline"
      >
        Jump to frame
      </button>
    </div>
  );
}
