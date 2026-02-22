import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { generateDemoData } from '@/data/demoData';
import { fetchFrameList, fetchRawFrameData, fetchVlmResponse } from '@/api';
import { VideoPlayer } from '@/components/PreCheck/VideoPlayer';
import { CalibrationPanel } from '@/components/PreCheck/CalibrationPanel';
import { FindingsFeed } from '@/components/PreCheck/FindingsFeed';
import { SpatialQA } from '@/components/PreCheck/SpatialQA';
import { AlertManager } from '@/components/PreCheck/AlertManager';
import { ProjectReport } from '@/components/PreCheck/ProjectReport';
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable';
import type { FrameData, Finding, AlertData } from '@/components/PreCheck/types';

// Fallback when API has no processed frames yet
const demoData = generateDemoData();

const Index = () => {
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [measureMode, setMeasureMode] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [confidenceHistory, setConfidenceHistory] = useState<number[]>([]);
  const seenRef = useRef(new Set<string>());
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  // ---------------------------------------------------------------------------
  // API data fetching
  // ---------------------------------------------------------------------------

  const { data: frameIds = [] } = useQuery({
    queryKey: ['frames'],
    queryFn: fetchFrameList,
    retry: 1,
    staleTime: 30_000,
  });

  const useApi = frameIds.length > 0;

  const { data: apiFrames = [] } = useQuery<FrameData[]>({
    queryKey: ['allFrames', frameIds],
    queryFn: async () => {
      const results = await Promise.all(
        frameIds.map(async (id, idx) => {
          const raw = await fetchRawFrameData(id);
          return { ...raw, frame_id: idx } as FrameData;
        })
      );
      return results;
    },
    enabled: useApi,
    staleTime: 30_000,
  });

  // Use API frames when available, fall back to demo data
  const frames: FrameData[] = useApi && apiFrames.length > 0 ? apiFrames : demoData.frames;
  const totalFrames = frames.length;
  const frameData = frames[currentFrameIndex] ?? null;

  // Fetch VLM anchor-calibrated report for current frame
  const { data: vlmResponse = '' } = useQuery({
    queryKey: ['vlm', frameData?.image_id],
    queryFn: () => fetchVlmResponse(frameData!.image_id!),
    enabled: !!frameData?.image_id,
    staleTime: Infinity,
  });

  // ---------------------------------------------------------------------------
  // Playback
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentFrameIndex(prev => {
          if (prev >= totalFrames - 1) { setIsPlaying(false); return prev; }
          return prev + 1;
        });
      }, 1000);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, totalFrames]);

  // Reset state when switching data source
  useEffect(() => {
    seenRef.current.clear();
    setFindings([]);
    setAlerts([]);
    setCurrentFrameIndex(0);
  }, [useApi]);

  // ---------------------------------------------------------------------------
  // Process findings on frame change
  // ---------------------------------------------------------------------------

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
          detail: m.type === 'box_height'
            ? `${m.inches}" from floor · within tolerance`
            : 'compliant',
          frame_id: frameData.frame_id,
        });
      }
    }

    if (newFindings.length) setFindings(prev => [...newFindings, ...prev]);
    if (newAlerts.length) setAlerts(prev => [...prev, ...newAlerts]);

    setConfidenceHistory(prev => {
      const next = [...prev, frameData.calibration.confidence];
      return next.length > 10 ? next.slice(-10) : next;
    });
  }, [currentFrameIndex, frameData]);

  // ---------------------------------------------------------------------------
  // Keyboard shortcuts
  // ---------------------------------------------------------------------------

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

  const handleSeek = useCallback((frame: number) => setCurrentFrameIndex(frame), []);
  const dismissAlert = useCallback((id: string) => setAlerts(prev => prev.filter(a => a.id !== id)), []);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      <header className="flex items-center justify-between px-4 py-2 border-b border-border bg-card shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold tracking-wide text-primary">PRECHECK</h1>
          <span className="text-[10px] text-muted-foreground font-mono">
            {useApi ? `v1.0 LIVE · ${frameIds.length} frames` : 'v1.0 DEMO'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {useApi && (
            <button
              onClick={() => setShowReport(true)}
              className="text-[10px] font-mono text-muted-foreground hover:text-primary border border-border hover:border-primary px-2 py-1 rounded"
            >
              Project Report
            </button>
          )}
          <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
            <span className={useApi ? 'status-dot-live' : 'status-dot-idle'} />
            <span>{useApi ? 'PIPELINE DATA' : 'DEMO MODE'}</span>
          </div>
        </div>
      </header>

      <div className="flex flex-1 min-h-0 max-w-[1400px] w-full mx-auto">
        <aside className="w-[260px] shrink-0 border-r border-border bg-card">
          <CalibrationPanel
            calibration={frameData?.calibration ?? null}
            currentFrame={currentFrameIndex}
            totalFrames={totalFrames}
            findings={findings}
            confidenceHistory={confidenceHistory}
          />
        </aside>

        <main className="flex-1 flex flex-col min-w-0">
          <VideoPlayer
            frameData={frameData}
            currentFrame={currentFrameIndex}
            totalFrames={totalFrames}
            isPlaying={isPlaying}
            onTogglePlay={() => setIsPlaying(p => !p)}
            onSeek={handleSeek}
            frames={frames}
            measureMode={measureMode}
            onToggleMeasure={() => setMeasureMode(p => !p)}
          />
        </main>

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
              <SpatialQA
                imageId={frameData?.image_id ?? null}
                calibration={frameData?.calibration ?? null}
                vlmResponse={vlmResponse || null}
              />
            </ResizablePanel>
          </ResizablePanelGroup>
        </aside>
      </div>

      {showReport && <ProjectReport onClose={() => setShowReport(false)} />}

      <AlertManager
        alerts={alerts}
        onDismiss={dismissAlert}
        onJumpToFrame={(fi) => { setCurrentFrameIndex(fi); setIsPlaying(false); }}
      />
    </div>
  );
};

export default Index;
