import { useEffect, useRef } from 'react';
import type { Finding } from './types';

interface Props {
  findings: Finding[];
  isPlaying: boolean;
  onSeekToFrame: (frameIndex: number) => void;
}

export function FindingsFeed({ findings, isPlaying, onSeekToFrame }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [findings.length]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-3 pt-3 pb-2">
        <span className="panel-header">Findings</span>
        <span className={isPlaying ? 'status-dot-live' : 'status-dot-idle'} />
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 pb-3 space-y-2">
        {findings.length === 0 && (
          <p className="text-muted-foreground text-[10px] mt-4 text-center">
            Findings will appear as the video plays…
          </p>
        )}
        {findings.map((f) => {
          const cardClass =
            f.type === 'violation' ? 'finding-card-fail' :
            f.type === 'info' ? 'finding-card-info' : 'finding-card-pass';

          const icon = f.type === 'violation' ? '✗' : f.type === 'info' ? 'ℹ' : '✓';

          return (
            <div
              key={f.id + f.frame_id}
              className={cardClass}
              onClick={() => onSeekToFrame(f.frame_id - 1)}
            >
              <div className="flex items-start gap-1.5">
                <span className={`font-bold ${
                  f.type === 'violation' ? 'text-destructive' :
                  f.type === 'info' ? 'text-info' : 'text-success'
                }`}>{icon}</span>
                <div className="min-w-0">
                  <div className="font-medium text-foreground">{f.label}</div>
                  <div className="font-mono text-[10px] mt-0.5">
                    <span className="text-foreground">{f.value}</span>
                    <span className="text-muted-foreground"> · {f.detail}</span>
                  </div>
                  {f.type === 'violation' && f.delta != null && (
                    <div className="font-mono text-[10px] mt-0.5">
                      <span className="text-destructive">
                        Delta +{f.delta.toFixed(1)}" · {(f.severity || 'CRITICAL').toUpperCase()}
                      </span>
                    </div>
                  )}
                  <div className="text-muted-foreground text-[10px] mt-0.5">
                    Frame {String(f.frame_id).padStart(3, '0')}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
