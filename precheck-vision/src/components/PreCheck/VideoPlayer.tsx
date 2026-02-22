import { useRef, useEffect, useState, useCallback } from 'react';
import type { FrameData, AnchorData } from './types';
import { CANVAS_WIDTH, CANVAS_HEIGHT } from '@/data/demoData';
import { Play, Pause, Ruler } from 'lucide-react';

const ANCHOR_COLORS: Record<string, string> = {
  stud: '#F97316',
  cmu: '#06B6D4',
  rebar: '#A855F7',
  elec_box: '#EAB308',
};

const ANCHOR_KNOWN: Record<string, string> = {
  stud: '2×4 stud — 3.5″ face width',
  cmu: 'CMU block — 8″ nominal',
  rebar: '#4 rebar — 0.5″ dia',
  elec_box: 'Single-gang box — 2″ × 3″',
};

interface Props {
  frameData: FrameData | null;
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeek: (frame: number) => void;
  frames: FrameData[];
  measureMode: boolean;
  onToggleMeasure: () => void;
}

export function VideoPlayer({
  frameData, currentFrame, totalFrames, isPlaying,
  onTogglePlay, onSeek, frames, measureMode, onToggleMeasure,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredAnchor, setHoveredAnchor] = useState<AnchorData | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [measurePoints, setMeasurePoints] = useState<{ x: number; y: number }[]>([]);
  const lineProgressRef = useRef(1);
  const prevFrameRef = useRef(-1);
  const animFrameRef = useRef(0);

  const toCanvasCoords = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * CANVAS_WIDTH,
      y: ((clientY - rect.top) / rect.height) * CANVAS_HEIGHT,
    };
  }, []);

  const drawScene = useCallback((ctx: CanvasRenderingContext2D) => {
    const W = CANVAS_WIDTH, H = CANVAS_HEIGHT;

    // Wall background
    ctx.fillStyle = '#3A3530';
    ctx.fillRect(0, 0, W, H);

    // Subtle grain texture
    ctx.globalAlpha = 0.04;
    for (let i = 0; i < 800; i++) {
      const gx = Math.random() * W;
      const gy = Math.random() * H;
      ctx.fillStyle = Math.random() > 0.5 ? '#fff' : '#000';
      ctx.fillRect(gx, gy, 1, 1);
    }
    ctx.globalAlpha = 1;

    // Floor
    ctx.fillStyle = '#252018';
    ctx.fillRect(0, 510, W, 30);

    // Bottom plate
    ctx.fillStyle = '#5C4A3A';
    ctx.fillRect(0, 498, W, 14);

    // Top plate
    ctx.fillStyle = '#5C4A3A';
    ctx.fillRect(0, 12, W, 14);

    // Studs from frame data
    if (frameData) {
      for (const anchor of frameData.anchors) {
        const [x1, y1, x2, y2] = anchor.box;
        if (anchor.type === 'stud') {
          ctx.fillStyle = '#6B5344';
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          // Grain lines
          ctx.strokeStyle = '#5A4434';
          ctx.lineWidth = 0.5;
          for (let y = y1 + 8; y < y2; y += 12 + Math.random() * 6) {
            ctx.beginPath();
            ctx.moveTo(x1 + 2, y);
            ctx.lineTo(x2 - 2, y + 4 + Math.random() * 4);
            ctx.stroke();
          }
        } else if (anchor.type === 'elec_box') {
          ctx.fillStyle = '#4A5568';
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          ctx.strokeStyle = '#2D3748';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          // Knockout holes
          ctx.fillStyle = '#3A4558';
          const cx = (x1 + x2) / 2;
          ctx.beginPath();
          ctx.arc(cx, y1 + 8, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(cx, y2 - 8, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Camera watermark
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '10px "Space Mono", monospace';
    ctx.fillText('CAM-01  SITE INSPECTION', 12, H - 8);
    if (frameData) {
      const ts = `FRM ${String(frameData.frame_id).padStart(3, '0')}`;
      ctx.fillText(ts, W - 90, H - 8);
    }
  }, [frameData]);

  const drawOverlay = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!frameData) return;
    const progress = lineProgressRef.current;
    const now = Date.now();

    // Bounding boxes
    for (const anchor of frameData.anchors) {
      const [x1, y1, x2, y2] = anchor.box;
      const color = ANCHOR_COLORS[anchor.type] || '#F97316';

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label chip
      const labelText = anchor.label;
      ctx.font = 'bold 9px "Space Mono", monospace';
      const tw = ctx.measureText(labelText).width;
      const chipW = tw + 10;
      const chipH = 16;
      const chipX = x1;
      const chipY = y1 - chipH - 3;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(chipX, chipY, chipW, chipH, 2);
      ctx.fill();

      ctx.fillStyle = '#000';
      ctx.fillText(labelText, chipX + 5, chipY + 11);
    }

    // Measurement lines
    for (const m of frameData.measurements) {
      const midX = (m.cx_a + m.cx_b) / 2;
      const midY = (m.cy_a + m.cy_b) / 2;
      const dx = m.cx_b - m.cx_a;
      const dy = m.cy_b - m.cy_a;

      const sx = midX - (dx / 2) * progress;
      const sy = midY - (dy / 2) * progress;
      const ex = midX + (dx / 2) * progress;
      const ey = midY + (dy / 2) * progress;

      const lineColor = m.compliant ? '#4ADE80' : '#FB7185';

      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.setLineDash([]);

      // Endpoint markers
      if (progress > 0.5) {
        ctx.fillStyle = lineColor;
        ctx.beginPath();
        ctx.arc(m.cx_a, m.cy_a, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(m.cx_b, m.cy_b, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Distance label
      if (progress > 0.7) {
        const label = m.compliant ? `${m.inches}\"` : `${m.inches}" ✗`;
        ctx.font = 'bold 11px "Space Mono", monospace';
        const tw = ctx.measureText(label).width;
        const lx = midX - tw / 2 - 5;
        const ly = midY - 12;

        ctx.fillStyle = 'rgba(15,13,11,0.85)';
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + 10, 18, 3);
        ctx.fill();

        ctx.fillStyle = lineColor;
        ctx.fillText(label, lx + 5, ly + 13);
      }
    }

    // Violation pulse rings
    const violations = frameData.measurements.filter(m => !m.compliant);
    for (const v of violations) {
      const pulseAlpha = 0.2 + 0.2 * Math.sin(now / 300);
      const fromAnchor = frameData.anchors.find(a => a.id === v.from_anchor);
      const toAnchor = frameData.anchors.find(a => a.id === v.to_anchor);
      if (fromAnchor && toAnchor) {
        const rx1 = Math.min(fromAnchor.box[0], toAnchor.box[0]) - 6;
        const ry1 = Math.min(fromAnchor.box[1], toAnchor.box[1]) - 6;
        const rx2 = Math.max(fromAnchor.box[2], toAnchor.box[2]) + 6;
        const ry2 = Math.max(fromAnchor.box[3], toAnchor.box[3]) + 6;

        ctx.strokeStyle = `rgba(251, 113, 133, ${pulseAlpha})`;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
      }
    }

    // Measure mode points and line
    if (measurePoints.length > 0) {
      ctx.fillStyle = '#F97316';
      for (const p of measurePoints) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      if (measurePoints.length === 2 && frameData.calibration) {
        const [p1, p2] = measurePoints;
        ctx.strokeStyle = '#F97316';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.setLineDash([]);

        const dist = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
        const inches = dist / frameData.calibration.pixels_per_inch;
        const label = `${dist.toFixed(0)}px ÷ ${frameData.calibration.pixels_per_inch.toFixed(2)} px/in = ${inches.toFixed(1)}"`;
        const mx = (p1.x + p2.x) / 2;
        const my = (p1.y + p2.y) / 2 - 16;

        ctx.font = '10px "Space Mono", monospace';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(15,13,11,0.9)';
        ctx.beginPath();
        ctx.roundRect(mx - tw / 2 - 6, my - 2, tw + 12, 18, 3);
        ctx.fill();
        ctx.fillStyle = '#F97316';
        ctx.fillText(label, mx - tw / 2, my + 12);
      }
    }
  }, [frameData, measurePoints]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Reset line progress on frame change
    if (prevFrameRef.current !== currentFrame) {
      lineProgressRef.current = 0;
      prevFrameRef.current = currentFrame;
    }

    const startTime = performance.now();
    const duration = 300;

    const render = (time: number) => {
      const elapsed = time - startTime;
      if (lineProgressRef.current < 1) {
        lineProgressRef.current = Math.min(1, elapsed / duration);
      }

      ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      drawScene(ctx);
      drawOverlay(ctx);

      // Keep animating if violations exist (for pulse) or line still drawing
      const hasViolations = frameData?.measurements.some(m => !m.compliant);
      if (lineProgressRef.current < 1 || hasViolations) {
        animFrameRef.current = requestAnimationFrame(render);
      }
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [frameData, currentFrame, drawScene, drawOverlay]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
    if (!frameData) return;
    const { x, y } = toCanvasCoords(e.clientX, e.clientY);
    setTooltipPos({ x: e.clientX, y: e.clientY });

    const found = frameData.anchors.find(a => {
      const [x1, y1, x2, y2] = a.box;
      return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    });
    setHoveredAnchor(found || null);
  }, [frameData, toCanvasCoords]);

  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    if (!measureMode) return;
    const { x, y } = toCanvasCoords(e.clientX, e.clientY);
    setMeasurePoints(prev => {
      if (prev.length >= 2) return [{ x, y }];
      return [...prev, { x, y }];
    });
  }, [measureMode, toCanvasCoords]);

  useEffect(() => {
    if (!measureMode) setMeasurePoints([]);
  }, [measureMode]);

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const frame = Math.round(pct * (totalFrames - 1));
    onSeek(Math.max(0, Math.min(totalFrames - 1, frame)));
  };

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Canvas area */}
      <div
        ref={containerRef}
        className="relative flex-1 flex items-center justify-center bg-background overflow-hidden"
      >
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          className="max-w-full max-h-full border border-border"
          style={{ cursor: measureMode ? 'crosshair' : 'default', imageRendering: 'auto' }}
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={() => setHoveredAnchor(null)}
          onClick={handleCanvasClick}
        />

        {/* Hover tooltip */}
        {hoveredAnchor && !measureMode && (
          <div
            className="fixed z-50 px-2 py-1 rounded text-[10px] font-mono bg-card border border-border text-foreground pointer-events-none"
            style={{ left: tooltipPos.x + 12, top: tooltipPos.y - 10 }}
          >
            {ANCHOR_KNOWN[hoveredAnchor.type] || hoveredAnchor.type}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 px-4 py-2 border-t border-border bg-card">
        <button
          onClick={onTogglePlay}
          className="flex items-center justify-center w-8 h-8 rounded bg-secondary text-foreground hover:bg-primary hover:text-primary-foreground transition-colors"
        >
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        </button>

        <button
          onClick={onToggleMeasure}
          className={`flex items-center justify-center w-8 h-8 rounded transition-colors ${
            measureMode ? 'bg-primary text-primary-foreground' : 'bg-secondary text-foreground hover:bg-muted'
          }`}
          title="Measure Mode"
        >
          <Ruler size={14} />
        </button>

        {/* Scrubber */}
        <div className="flex-1 relative h-3 cursor-pointer group" onClick={handleScrubberClick}>
          <div className="absolute inset-y-1 inset-x-0 bg-secondary rounded-full" />
          {/* Colored ticks */}
          {frames.map((f, i) => {
            const hasViolation = f.measurements.some(m => !m.compliant);
            return (
              <div
                key={i}
                className="absolute top-0.5 w-[2px] h-2 rounded-full"
                style={{
                  left: `${(i / (totalFrames - 1)) * 100}%`,
                  backgroundColor: hasViolation ? 'hsl(353, 94%, 71%)' : 'hsl(142, 71%, 45%)',
                  opacity: 0.5,
                }}
              />
            );
          })}
          {/* Playhead */}
          <div
            className="absolute top-0 w-3 h-3 rounded-full bg-primary border-2 border-background -translate-x-1/2"
            style={{ left: `${(currentFrame / Math.max(1, totalFrames - 1)) * 100}%` }}
          />
        </div>

        {/* Frame counter */}
        <span className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          Frame {String(currentFrame + 1).padStart(3, '0')} / {totalFrames}
        </span>
      </div>
    </div>
  );
}
