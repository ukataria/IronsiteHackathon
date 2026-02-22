import { imageUrl, type FrameSummary } from '@/api';

interface Props {
  frames: FrameSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function FrameStrip({ frames, selectedId, onSelect }: Props) {
  if (!frames.length) return null;

  return (
    <div className="flex gap-2 overflow-x-auto pb-1 shrink-0">
      {frames.map((f) => {
        const sel = f.id === selectedId;
        const border =
          sel                       ? 'border-primary ring-1 ring-primary'
          : f.compliance === 'pass' ? 'border-success/40 hover:border-success/70'
          : f.compliance === 'fail' ? 'border-destructive/40 hover:border-destructive/70'
          :                           'border-border hover:border-primary/40';

        const shortLabel = f.id.includes('_f')
          ? `f${parseInt(f.id.split('_f').pop() ?? '0', 10).toString().padStart(4, '0')}`
          : f.id;

        return (
          <button
            key={f.id}
            onClick={() => onSelect(f.id)}
            className={`shrink-0 w-[72px] rounded-lg overflow-hidden border-2 transition-all ${border} focus:outline-none`}
          >
            <img
              src={imageUrl(f.id, 'raw')}
              alt={f.id}
              className="w-full h-11 object-cover bg-secondary"
              loading="lazy"
            />
            <div className="flex items-center justify-between px-1.5 py-1 bg-card">
              <span className="font-mono text-[9px] text-muted-foreground truncate">{shortLabel}</span>
              <span
                className={`text-[9px] font-bold ${
                  f.compliance === 'pass' ? 'text-success'
                  : f.compliance === 'fail' ? 'text-destructive'
                  : 'text-muted-foreground'
                }`}
              >
                {f.compliance === 'pass' ? '✓' : f.compliance === 'fail' ? '✗' : '·'}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
