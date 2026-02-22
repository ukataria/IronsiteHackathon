import { useState, useCallback } from 'react';
import { UploadZone } from './UploadZone';
import { FrameStrip } from './FrameStrip';
import { LiveChat } from './LiveChat';
import * as api from '@/api';

type Tab       = 'inspector' | 'comparison';
type ImageType = 'raw' | 'annotated' | 'measured';
type Condition = 'anchor_calibrated' | 'baseline' | 'depth' | 'all';
type Vlm       = 'claude' | 'gpt4o' | 'ollama';

const CONDITION_LABELS: Record<string, string> = {
  baseline:          'Baseline',
  depth:             'Depth-Augmented',
  anchor_calibrated: 'DeepAnchored',
};

// ---------------------------------------------------------------------------
// Comparison tab (3-column side-by-side)
// ---------------------------------------------------------------------------

function ComparisonTab({ frameId, vlm }: { frameId: string | null; vlm: Vlm }) {
  const [question, setQuestion] = useState(
    'What deficiencies exist in this construction work? Provide a full inspection report.',
  );
  const [results,  setResults]  = useState<api.ConditionAnswer[] | null>(null);
  const [loading,  setLoading]  = useState(false);

  const run = async () => {
    if (!frameId || loading) return;
    setLoading(true);
    try {
      const res = await api.askQuestion(frameId, question, 'all', vlm);
      setResults(res.answers);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 p-4 gap-3">
      {/* Question input */}
      <div className="flex gap-2 shrink-0">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void run()}
          className="flex-1 bg-secondary text-foreground text-xs px-3 py-2 rounded border border-border focus:outline-none focus:border-primary font-mono"
          placeholder="Inspection question…"
        />
        <button
          onClick={() => void run()}
          disabled={!frameId || loading}
          className="px-4 py-2 rounded bg-primary text-primary-foreground text-xs font-bold disabled:opacity-40 shrink-0"
        >
          {loading ? 'Running…' : 'Run'}
        </button>
      </div>

      {/* Results grid */}
      {results ? (
        <div className="grid grid-cols-3 gap-3 flex-1 overflow-y-auto min-h-0">
          {results.map((r) => (
            <div
              key={r.condition}
              className={`rounded-xl border overflow-hidden flex flex-col ${
                r.condition === 'anchor_calibrated' ? 'border-primary' : 'border-border'
              }`}
            >
              <div
                className={`flex items-center justify-between px-3 py-2.5 border-b shrink-0 ${
                  r.condition === 'anchor_calibrated'
                    ? 'border-primary/20 bg-primary/5'
                    : 'border-border bg-card'
                }`}
              >
                <div>
                  <div className="text-xs font-bold">
                    {CONDITION_LABELS[r.condition] ?? r.condition}
                  </div>
                  {r.condition === 'anchor_calibrated' && (
                    <div className="text-[9px] text-primary font-mono mt-0.5">
                      Calibrated px → inch scale
                    </div>
                  )}
                </div>
                <span
                  className={`text-[10px] font-mono font-bold ${
                    r.verdict === 'PASS'    ? 'text-success'
                    : r.verdict === 'FAIL' ? 'text-destructive'
                    :                        'text-muted-foreground'
                  }`}
                >
                  {r.verdict}
                </span>
              </div>
              <div className="p-3 text-[11px] leading-relaxed whitespace-pre-wrap font-mono text-muted-foreground overflow-y-auto flex-1">
                {r.response}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-xs">
          {frameId ? 'Click Run to compare all 3 conditions' : 'Select a frame first'}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main LiveInspector
// ---------------------------------------------------------------------------

export function LiveInspector() {
  const [view,         setView]         = useState<'upload' | 'inspector'>('upload');
  const [tab,          setTab]          = useState<Tab>('inspector');
  const [frames,       setFrames]       = useState<api.FrameSummary[]>([]);
  const [selectedId,   setSelectedId]   = useState<string | null>(null);
  const [imageType,    setImageType]    = useState<ImageType>('raw');
  const [measurements, setMeasurements] = useState<api.Measurements | null>(null);
  const [condition,    setCondition]    = useState<Condition>('anchor_calibrated');
  const [vlm,          setVlm]          = useState<Vlm>('claude');

  const handleUploadComplete = useCallback(
    async (newFrames: api.FrameSummary[]) => {
      setFrames(newFrames);
      if (newFrames.length > 0) {
        setSelectedId(newFrames[0].id);
        setMeasurements(await api.getMeasurements(newFrames[0].id));
      }
      setView('inspector');
    },
    [],
  );

  const handleSelectFrame = useCallback(async (id: string) => {
    setSelectedId(id);
    setMeasurements(await api.getMeasurements(id).catch(() => null));
  }, []);

  // ── Upload view ─────────────────────────────────────────────────────────────
  if (view === 'upload') {
    return <UploadZone onComplete={handleUploadComplete} vlm={vlm} />;
  }

  // ── Inspector view ──────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-1.5 border-b border-border bg-card shrink-0 flex-wrap">
        {/* Tabs */}
        <div className="flex gap-1">
          {(['inspector', 'comparison'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1 text-xs font-semibold rounded transition-colors ${
                tab === t
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t === 'inspector' ? 'Inspector' : 'Comparison'}
            </button>
          ))}
        </div>

        <div className="flex-1" />

        {/* Model */}
        <select
          value={vlm}
          onChange={(e) => setVlm(e.target.value as Vlm)}
          className="bg-secondary border border-border text-foreground text-xs px-2 py-1 rounded focus:outline-none focus:border-primary"
        >
          <option value="claude">Claude Sonnet</option>
          <option value="gpt4o">GPT-4o</option>
          <option value="ollama">Ollama (local)</option>
        </select>

        {/* Condition (only relevant in inspector tab) */}
        {tab === 'inspector' && (
          <select
            value={condition}
            onChange={(e) => setCondition(e.target.value as Condition)}
            className="bg-secondary border border-border text-foreground text-xs px-2 py-1 rounded focus:outline-none focus:border-primary"
          >
            <option value="anchor_calibrated">DeepAnchored (calibrated)</option>
            <option value="baseline">Baseline</option>
            <option value="depth">Depth</option>
            <option value="all">Compare all 3</option>
          </select>
        )}

        {/* Upload new */}
        <button
          onClick={() => { setView('upload'); setFrames([]); setSelectedId(null); setMeasurements(null); }}
          className="text-[11px] text-muted-foreground hover:text-foreground border border-border px-2 py-1 rounded transition-colors"
        >
          ↑ New upload
        </button>
      </div>

      {/* Frame strip */}
      {frames.length > 0 && (
        <div className="px-4 py-2 border-b border-border bg-card shrink-0 overflow-x-auto">
          <FrameStrip frames={frames} selectedId={selectedId} onSelect={handleSelectFrame} />
        </div>
      )}

      {/* Main area */}
      {tab === 'comparison' ? (
        <ComparisonTab frameId={selectedId} vlm={vlm} />
      ) : (
        <div className="flex flex-1 min-h-0">
          {/* ── Left panel: image viewer + measurements (40%) ── */}
          <div className="w-[40%] flex flex-col border-r border-border min-h-0">
            {/* Image type toggle */}
            <div className="flex gap-1 px-3 py-2 border-b border-border bg-card shrink-0">
              {(['raw', 'annotated', 'measured'] as ImageType[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setImageType(t)}
                  className={`px-2 py-1 text-[10px] font-mono font-bold rounded uppercase tracking-wider transition-colors ${
                    imageType === t
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {t === 'raw' ? '① Raw' : t === 'annotated' ? '② Anchors' : '③ Measured'}
                </button>
              ))}
            </div>

            {/* Image */}
            <div className="flex-1 flex items-center justify-center bg-muted/10 overflow-hidden">
              {selectedId ? (
                <img
                  key={`${selectedId}-${imageType}`}
                  src={api.imageUrl(selectedId, imageType)}
                  alt={`${imageType} frame`}
                  className="max-w-full max-h-full object-contain"
                  onError={(e) => {
                    if (imageType !== 'raw') {
                      (e.target as HTMLImageElement).src = api.imageUrl(selectedId, 'raw');
                    }
                  }}
                />
              ) : (
                <p className="text-muted-foreground text-xs">Select a frame above</p>
              )}
            </div>

            {/* Measurements panel */}
            {measurements && (
              <div className="border-t border-border p-3 overflow-y-auto max-h-56 shrink-0 bg-card">
                <div className="flex items-center justify-between mb-2">
                  <span className="panel-header">Calibrated Measurements</span>
                  <span className="font-mono text-[10px] text-primary">
                    {measurements.scale_pixels_per_inch.toFixed(1)} px/in
                    &nbsp;·&nbsp;
                    {(measurements.calibration_confidence * 100).toFixed(0)}% conf
                  </span>
                </div>

                {measurements.stud_spacings.map((s, i) => (
                  <div key={`stud-${i}`} className="flex justify-between items-center py-1.5 border-b border-border last:border-0 text-xs">
                    <span className="text-muted-foreground">Stud bay {i + 1}</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono">{s.inches.toFixed(1)}"</span>
                      <span className={s.compliant ? 'text-success text-[10px] font-bold' : 'text-destructive text-[10px] font-bold'}>
                        {s.compliant ? '✓' : '✗'}
                      </span>
                    </div>
                  </div>
                ))}

                {measurements.rebar_spacings.map((s, i) => (
                  <div key={`rebar-${i}`} className="flex justify-between items-center py-1.5 border-b border-border last:border-0 text-xs">
                    <span className="text-muted-foreground">Rebar bay {i + 1}</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono">{s.inches.toFixed(1)}"</span>
                      <span className={s.compliant ? 'text-success text-[10px] font-bold' : 'text-destructive text-[10px] font-bold'}>
                        {s.compliant ? '✓' : '✗'}
                      </span>
                    </div>
                  </div>
                ))}

                {measurements.electrical_box_heights.map((h, i) => (
                  <div key={`box-${i}`} className="flex justify-between items-center py-1.5 border-b border-border last:border-0 text-xs">
                    <span className="text-muted-foreground">Box {i + 1} height</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono">{h.height_inches.toFixed(1)}"</span>
                      <span className={h.compliant ? 'text-success text-[10px] font-bold' : 'text-destructive text-[10px] font-bold'}>
                        {h.compliant ? '✓' : '✗'}
                      </span>
                    </div>
                  </div>
                ))}

                {measurements.summary && (
                  <p className="mt-2 text-[10px] text-muted-foreground leading-relaxed">
                    {measurements.summary}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* ── Right panel: chat (60%) ── */}
          <div className="flex-1 flex flex-col min-h-0">
            <LiveChat frameId={selectedId} condition={condition} vlm={vlm} />
          </div>
        </div>
      )}
    </div>
  );
}
