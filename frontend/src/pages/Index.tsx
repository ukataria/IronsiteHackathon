import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { generateDemoData } from '@/data/demoData';
import { VideoPlayer } from '@/components/PreCheck/VideoPlayer';
import { CalibrationPanel } from '@/components/PreCheck/CalibrationPanel';
import { FindingsFeed } from '@/components/PreCheck/FindingsFeed';
import { SpatialQA } from '@/components/PreCheck/SpatialQA';
import { AlertManager } from '@/components/PreCheck/AlertManager';
import { LiveInspector } from '@/components/PreCheck/LiveInspector';
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable';
import type { Finding, AlertData } from '@/components/PreCheck/types';

const demoData = generateDemoData();

const Index = () => {
  const [mode, setMode] = useState<'demo' | 'live'>('demo');
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [measureMode, setMeasureMode] = useState(false);
  const [confidenceHistory, setConfidenceHistory] = useState<number[]>([]);
  const seenRef = useRef(new Set<string>());
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  const frameData = demoData.frames[currentFrameIndex] ?? null;
  const totalFrames = demoData.total_frames;

  // Playback
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentFrameIndex(prev => {
          if (prev >= totalFrames - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, totalFrames]);

  // Process findings on frame change
  useEffect(() => {
    if (!frameData) return;

    const newFindings: Finding[] = [];
    const newAlerts: AlertData[] = [];

    for (const m of frameData.measurements) {
      if (seenRef.current.has(m.id)) continue;
      seenRef.current.add(m.id);

      if (!m.compliant) {
        newFindings.push({
          id: m.id,
          type: 'violation',
          label: m.label,
          value: `${m.inches}"`,
          detail: `expected ${m.expected}" ± ${m.tolerance}"`,
          frame_id: frameData.frame_id,
          severity: m.severity,
          expected: m.expected,
          tolerance: m.tolerance,
          delta: m.inches - m.expected,
        });
        newAlerts.push({
          id: `alert_${m.id}_${Date.now()}`,
          label: m.label,
          measured: m.inches,
          expected: m.expected,
          tolerance: m.tolerance,
          delta: m.inches - m.expected,
          severity: m.severity,
          frame_id: frameData.frame_id,
          timestamp: Date.now(),
        });
      } else {
        newFindings.push({
          id: m.id,
          type: m.type === 'box_height' ? 'info' : 'compliant',
          label: m.label,
          value: `${m.inches}"`,
          detail: m.type === 'box_height' ? `${m.inches}" from floor · within tolerance` : 'compliant',
          frame_id: frameData.frame_id,
        });
      }
    }

    if (newFindings.length) setFindings(prev => [...newFindings, ...prev]);
    if (newAlerts.length) setAlerts(prev => [...prev, ...newAlerts]);

    // Confidence history
    setConfidenceHistory(prev => {
      const next = [...prev, frameData.calibration.confidence];
      return next.length > 10 ? next.slice(-10) : next;
    });
  }, [currentFrameIndex, frameData]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.code) {
        case 'Space':
          e.preventDefault();
          setIsPlaying(p => !p);
          break;
        case 'ArrowLeft':
          if (!isPlaying) setCurrentFrameIndex(p => Math.max(0, p - 1));
          break;
        case 'ArrowRight':
          if (!isPlaying) setCurrentFrameIndex(p => Math.min(totalFrames - 1, p + 1));
          break;
        case 'Escape':
          setMeasureMode(false);
          break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isPlaying, totalFrames]);

  const handleSeek = useCallback((frame: number) => {
    setCurrentFrameIndex(frame);
  }, []);

  const dismissAlert = useCallback((id: string) => {
    setAlerts(prev => prev.filter(a => a.id !== id));
  }, []);

  if (mode === 'live') {
    return (
      <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
        <header className="flex items-center justify-between px-4 py-2 border-b border-border bg-card shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-sm font-bold tracking-wide text-primary">DEEPANCHORED</h1>
            <span className="text-[10px] text-muted-foreground font-mono">v1.0 LIVE</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex rounded overflow-hidden border border-border text-[10px] font-mono">
              <button
                onClick={() => setMode('demo')}
                className="px-3 py-1 text-muted-foreground hover:text-foreground transition-colors"
              >
                DEMO
              </button>
              <button
                className="px-3 py-1 bg-primary text-primary-foreground"
              >
                LIVE
              </button>
            </div>
          </div>
        </header>
        <LiveInspector />
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-border bg-card shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold tracking-wide text-primary">DEEPANCHORED</h1>
          <span className="text-[10px] text-muted-foreground font-mono">v1.0 DEMO</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded overflow-hidden border border-border text-[10px] font-mono mr-3">
            <button
              className="px-3 py-1 bg-primary text-primary-foreground"
            >
              DEMO
            </button>
            <button
              onClick={() => setMode('live')}
              className="px-3 py-1 text-muted-foreground hover:text-foreground transition-colors"
            >
              LIVE
            </button>
          </div>
          <span className="status-dot-live" />
          <span className="text-[10px] font-mono text-muted-foreground">INSPECTION ACTIVE</span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 min-h-0 max-w-[1400px] w-full mx-auto">
        {/* Left sidebar */}
        <aside className="w-[260px] shrink-0 border-r border-border bg-card">
          <CalibrationPanel
            calibration={frameData?.calibration ?? null}
            currentFrame={currentFrameIndex}
            totalFrames={totalFrames}
            findings={findings}
            confidenceHistory={confidenceHistory}
          />
        </aside>

        {/* Center - Video */}
        <main className="flex-1 flex flex-col min-w-0">
          <VideoPlayer
            frameData={frameData}
            currentFrame={currentFrameIndex}
            totalFrames={totalFrames}
            isPlaying={isPlaying}
            onTogglePlay={() => setIsPlaying(p => !p)}
            onSeek={handleSeek}
            frames={demoData.frames}
            measureMode={measureMode}
            onToggleMeasure={() => setMeasureMode(p => !p)}
          />
        </main>

        {/* Right panel */}
        <aside className="w-[320px] shrink-0 border-l border-border bg-card">
          <ResizablePanelGroup direction="vertical">
            <ResizablePanel defaultSize={60} minSize={30}>
              <FindingsFeed
                findings={findings}
                isPlaying={isPlaying}
                onSeekToFrame={handleSeek}
              />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={40} minSize={20}>
              <SpatialQA calibration={frameData?.calibration ?? null} />
            </ResizablePanel>
          </ResizablePanelGroup>
        </aside>
      </div>

      {/* Alerts */}
      <AlertManager
        alerts={alerts}
        onDismiss={dismissAlert}
        onJumpToFrame={(fi) => {
          setCurrentFrameIndex(fi);
          setIsPlaying(false);
        }}
      />
    </div>
  );
};

export default Index;
