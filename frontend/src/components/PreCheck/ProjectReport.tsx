import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface AggStats {
  count: number;
  compliant: number;
  compliance_pct: number;
  mean_inches: number;
  target_inches: number;
  worst_val?: number;
}

interface ReportData {
  response: string;
  aggregated?: {
    total_frames: number;
    frames_with_detections: number;
    brick_h_spacings?: AggStats;
    brick_v_spacings?: AggStats;
    stud_spacings?: AggStats;
    rebar_spacings?: AggStats;
    cmu_spacings?: AggStats;
    electrical_boxes?: AggStats;
  };
}

interface Props {
  onClose: () => void;
}

export function ProjectReport({ onClose }: Props) {
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function load() {
    setLoading(true);
    setError('');
    try {
      // Try cached first
      const res = await fetch('/api/report');
      const data = await res.json();
      if (data.response) {
        setReport(data);
      } else {
        // Generate it
        const gen = await fetch('/api/report/generate', { method: 'POST' });
        if (!gen.ok) throw new Error('Generation failed');
        setReport(await gen.json());
      }
    } catch (e) {
      setError('Failed to generate report. Make sure the Flask server is running.');
    } finally {
      setLoading(false);
    }
  }

  // Auto-load on mount
  useState(() => { load(); });

  const agg = report?.aggregated;

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border bg-card shrink-0">
        <div>
          <span className="text-sm font-bold tracking-wide text-primary">PROJECT REPORT</span>
          {agg && (
            <span className="ml-3 text-[10px] font-mono text-muted-foreground">
              {agg.total_frames} frames · {agg.frames_with_detections} with detections
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={load}
            disabled={loading}
            className="text-[10px] font-mono text-muted-foreground hover:text-foreground disabled:opacity-40"
          >
            {loading ? 'Generating…' : 'Regenerate'}
          </button>
          <button
            onClick={onClose}
            className="text-[10px] font-mono text-muted-foreground hover:text-foreground"
          >
            ✕ Close
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Stats sidebar */}
        {agg && (
          <aside className="w-[200px] shrink-0 border-r border-border p-4 overflow-y-auto space-y-5">
            <p className="text-[10px] font-semibold tracking-wider text-muted-foreground uppercase">At a Glance</p>
            {[
              { label: 'Brick Spacing', sublabel: `avg ${agg.brick_h_spacings?.mean_inches}" · target 8.0"`, data: agg.brick_h_spacings },
              { label: 'Course Height', sublabel: `avg ${agg.brick_v_spacings?.mean_inches}" · target 2.625"`, data: agg.brick_v_spacings },
              { label: 'Stud Spacing', sublabel: `avg ${agg.stud_spacings?.mean_inches}" · target 16.0"`, data: agg.stud_spacings },
              { label: 'Rebar Spacing', sublabel: `avg ${agg.rebar_spacings?.mean_inches}" · target 12.0"`, data: agg.rebar_spacings },
              { label: 'CMU Spacing', sublabel: `avg ${agg.cmu_spacings?.mean_inches}" · target 16.0"`, data: agg.cmu_spacings },
              { label: 'Elec Box Height', sublabel: `avg ${agg.electrical_boxes?.mean_inches}" · target 12.0"`, data: agg.electrical_boxes },
            ].map(({ label, sublabel, data }) => {
              if (!data) return null;
              const pct = data.compliance_pct;
              const pass = pct >= 90;
              const warn = pct >= 70;
              const statusLabel = pass ? 'PASS' : warn ? 'REVIEW' : 'FAIL';
              const statusColor = pass ? 'text-green-500' : warn ? 'text-yellow-500' : 'text-red-500';
              const barColor = pass ? 'bg-green-500' : warn ? 'bg-yellow-500' : 'bg-red-500';
              return (
                <div key={label}>
                  <div className="flex items-center justify-between mb-0.5">
                    <p className="text-[11px] font-medium text-foreground">{label}</p>
                    <span className={`text-[10px] font-bold ${statusColor}`}>{statusLabel}</span>
                  </div>
                  <p className="text-[9px] text-muted-foreground mb-1">{sublabel}</p>
                  <div className="h-1.5 bg-secondary rounded overflow-hidden">
                    <div className={`h-full rounded ${barColor}`} style={{ width: `${pct}%` }} />
                  </div>
                  <p className="text-[9px] text-muted-foreground mt-0.5">{pct}% of checks passed</p>
                </div>
              );
            })}
          </aside>
        )}

        {/* Report body */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <p className="text-xs">Analyzing {agg?.total_frames ?? 'all'} frames…</p>
            </div>
          )}
          {error && !loading && (
            <p className="text-xs text-red-500">{error}</p>
          )}
          {report?.response && !loading && (
            <div className="max-w-2xl text-[12px] text-foreground leading-relaxed
              [&_h1]:text-sm [&_h1]:font-bold [&_h1]:mt-4 [&_h1]:mb-1
              [&_h2]:text-[12px] [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:mb-0.5 [&_h2]:text-primary
              [&_h3]:text-[11px] [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-0
              [&_p]:my-1
              [&_strong]:font-semibold
              [&_table]:w-full [&_table]:text-[11px] [&_table]:border-collapse [&_table]:my-2
              [&_th]:text-left [&_th]:px-2 [&_th]:py-1 [&_th]:border [&_th]:border-border [&_th]:bg-secondary [&_th]:font-semibold
              [&_td]:px-2 [&_td]:py-1 [&_td]:border [&_td]:border-border
              [&_ul]:my-1 [&_ul]:pl-4 [&_ol]:my-1 [&_ol]:pl-4 [&_li]:my-0.5
              [&_hr]:border-border [&_hr]:my-2">
              <ReactMarkdown>{report.response}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
