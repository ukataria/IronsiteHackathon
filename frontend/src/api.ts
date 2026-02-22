/** PreCheck backend API client. Set VITE_API_URL in .env.local to override. */

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UploadResponse {
  job_id: string;
  file_type: 'video' | 'image';
  filename: string;
}

export interface JobStatus {
  status: 'pending' | 'running' | 'complete' | 'error';
  stage: string;
  progress: number;
  frames_extracted: number;
  error?: string;
}

export interface FrameSummary {
  id: string;
  thumbnail_url: string;
  compliance: 'pass' | 'fail' | 'unknown';
}

export interface StudSpacing {
  inches: number;
  compliant: boolean;
  cx_a?: number;
  cy_a?: number;
  cx_b?: number;
  cy_b?: number;
}

export interface ElecBoxHeight {
  box_id: number;
  height_inches: number;
  compliant: boolean;
}

export interface Measurements {
  scale_pixels_per_inch: number;
  calibration_confidence: number;
  element_counts: Record<string, number>;
  stud_spacings: StudSpacing[];
  rebar_spacings: StudSpacing[];
  electrical_box_heights: ElecBoxHeight[];
  summary: string;
}

export interface ConditionAnswer {
  condition: string;
  response: string;
  verdict: 'PASS' | 'FAIL' | 'UNKNOWN';
}

export interface AskResponse {
  question: string;
  frame_id: string;
  answers: ConditionAnswer[];
}

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

export const imageUrl = (
  frameId: string,
  type: 'raw' | 'annotated' | 'measured' | 'depth',
): string => `${BASE}/api/frames/${encodeURIComponent(frameId)}/image?type=${type}`;

// ---------------------------------------------------------------------------
// REST calls
// ---------------------------------------------------------------------------

async function _json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadFile(file: File, vlm = 'claude'): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  return _json<UploadResponse>(
    await fetch(`${BASE}/api/upload?vlm=${encodeURIComponent(vlm)}`, {
      method: 'POST',
      body: form,
    }),
  );
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return _json<JobStatus>(await fetch(`${BASE}/api/job/${jobId}/status`));
}

export async function getJobFrames(jobId: string): Promise<FrameSummary[]> {
  const data = await _json<{ frames: FrameSummary[] }>(
    await fetch(`${BASE}/api/job/${jobId}/frames`),
  );
  return data.frames;
}

export async function listFrames(): Promise<FrameSummary[]> {
  const data = await _json<{ frames: FrameSummary[] }>(await fetch(`${BASE}/api/frames`));
  return data.frames;
}

export async function getMeasurements(frameId: string): Promise<Measurements | null> {
  const res = await fetch(`${BASE}/api/frames/${encodeURIComponent(frameId)}/measurements`);
  if (res.status === 404) return null;
  return _json<Measurements>(res);
}

export async function askQuestion(
  frameId: string,
  question: string,
  condition: string,
  vlm: string,
): Promise<AskResponse> {
  return _json<AskResponse>(
    await fetch(`${BASE}/api/frames/${encodeURIComponent(frameId)}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, condition, vlm }),
    }),
  );
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export function createJobWebSocket(jobId: string): WebSocket {
  const wsBase = BASE.replace(/^https?/, (s) => (s === 'https' ? 'wss' : 'ws'));
  return new WebSocket(`${wsBase}/ws/job/${jobId}`);
}
