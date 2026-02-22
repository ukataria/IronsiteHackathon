import { useState, useRef, useCallback } from 'react';
import { Upload, Film, ImageIcon, Check, Loader2, AlertCircle } from 'lucide-react';
import * as api from '@/api';

const STAGES = [
  'Extracting frames',
  'Detecting calibration anchors',
  'Computing px → inch scale',
  'Extracting measurements',
  'Running AI inspection',
];

interface Props {
  onComplete: (frames: api.FrameSummary[]) => void;
  vlm: string;
}

export function UploadZone({ onComplete, vlm }: Props) {
  const [dragOver, setDragOver]   = useState(false);
  const [processing, setProcessing] = useState(false);
  const [stage, setStage]         = useState('');
  const [progress, setProgress]   = useState(0);
  const [error, setError]         = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const activeStageIndex = STAGES.findIndex((s) =>
    stage.toLowerCase().includes(s.split(' ')[0].toLowerCase()),
  );

  const handleFile = useCallback(
    async (file: File) => {
      setProcessing(true);
      setError('');
      setProgress(2);
      setStage('Uploading…');

      try {
        const { job_id } = await api.uploadFile(file, vlm);

        await new Promise<void>((resolve, reject) => {
          const ws = api.createJobWebSocket(job_id);

          ws.onmessage = (e) => {
            const msg = JSON.parse(e.data as string) as {
              type: string;
              stage?: string;
              progress?: number;
              frame_ids?: string[];
              message?: string;
            };
            if (msg.type === 'stage') {
              setStage(msg.stage ?? '');
              setProgress(msg.progress ?? 0);
            } else if (msg.type === 'complete') {
              ws.close();
              resolve();
            } else if (msg.type === 'error') {
              ws.close();
              reject(new Error(msg.message ?? 'Pipeline error'));
            }
          };

          // Don't reject on WS error — let the polling fallback take over
          ws.onerror = () => { ws.close(); };

          // Fallback: poll if WS doesn't connect within 3 s
          const pollTimer = setTimeout(async () => {
            try {
              let done = false;
              while (!done) {
                await new Promise((r) => setTimeout(r, 2000));
                const status = await api.getJobStatus(job_id);
                setStage(status.stage);
                setProgress(status.progress);
                if (status.status === 'complete') { done = true; resolve(); }
                if (status.status === 'error') { done = true; reject(new Error(status.error)); }
              }
            } catch (err) {
              reject(err);
            }
          }, 3000);

          ws.onopen = () => clearTimeout(pollTimer);
        });

        const frames = await api.getJobFrames(job_id);
        onComplete(frames);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setProcessing(false);
      }
    },
    [vlm, onComplete],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  // ── Processing overlay ──────────────────────────────────────────────────────
  if (processing) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-sm w-full">
          <div className="text-center mb-8">
            <span className="panel-header block mb-2">Processing</span>
            <h2 className="text-base font-bold text-foreground">{stage || 'Starting…'}</h2>
            <div className="mt-4 h-1 rounded-full bg-secondary overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="mt-1.5 font-mono text-[10px] text-muted-foreground block">
              {progress}%
            </span>
          </div>

          <div className="space-y-2">
            {STAGES.map((s, i) => {
              const done   = i < activeStageIndex;
              const active = i === activeStageIndex;
              return (
                <div
                  key={i}
                  className={`flex items-center gap-3 p-2.5 rounded border text-xs transition-colors ${
                    active ? 'border-primary bg-card text-foreground'
                    : done  ? 'border-success/30 bg-card text-muted-foreground'
                    :         'border-border bg-card/30 text-muted-foreground/40'
                  }`}
                >
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
                      done   ? 'bg-success text-success-foreground'
                      : active ? 'bg-primary text-primary-foreground'
                      :         'bg-secondary text-muted-foreground'
                    }`}
                  >
                    {done   ? <Check size={10} />
                    : active ? <Loader2 size={10} className="animate-spin" />
                    : <span className="font-mono text-[9px]">{i + 1}</span>}
                  </div>
                  <span className="font-medium">{s}</span>
                </div>
              );
            })}
          </div>

          {error && (
            <div className="mt-4 flex gap-2 p-3 rounded border border-destructive/50 bg-destructive/10 text-destructive text-xs">
              <AlertCircle size={14} className="shrink-0 mt-0.5" />
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Drop zone ───────────────────────────────────────────────────────────────
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-lg w-full text-center">
        <div className="text-[3rem] mb-3 select-none">🎥</div>
        <h2 className="text-2xl font-bold mb-2">Upload a construction video</h2>
        <p className="text-muted-foreground text-sm leading-relaxed mb-8 max-w-sm mx-auto">
          DeepAnchored extracts frames, detects calibration anchors, converts pixels&nbsp;to&nbsp;inches,
          and runs a 3-condition AI inspection on every frame.
        </p>

        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi,.jpg,.jpeg,.png"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />

        <div
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-10 cursor-pointer transition-colors ${
            dragOver
              ? 'border-primary bg-primary/5'
              : 'border-border hover:border-primary/50 hover:bg-card'
          }`}
        >
          <div className="flex items-center justify-center gap-6 mb-4 text-muted-foreground">
            <Film size={28} />
            <Upload size={22} />
            <ImageIcon size={28} />
          </div>
          <p className="text-sm font-medium">Drop a video or image here</p>
          <p className="text-xs text-muted-foreground mt-1">MP4 · MOV · AVI · JPG · PNG</p>
        </div>

        {error && (
          <div className="mt-4 flex gap-2 p-3 rounded border border-destructive/50 bg-destructive/10 text-destructive text-xs">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            {error}
          </div>
        )}

        <p className="mt-6 text-[10px] text-muted-foreground font-mono">
          or run: uv run python scripts/run_batch.py
        </p>
      </div>
    </div>
  );
}
