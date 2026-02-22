/**
 * API client — fetches pipeline outputs from the Flask backend.
 * All endpoints proxy through Vite at /api → http://localhost:5000
 */

import type { CalibrationData, AnchorData, MeasurementData } from './components/PreCheck/types';

/** Shape returned by GET /api/frame/<image_id> — no frame_id yet (assigned client-side). */
export interface RawFrameData {
  image_id: string;
  image_width: number;
  image_height: number;
  calibration: CalibrationData;
  anchors: AnchorData[];
  measurements: MeasurementData[];
}

/** GET /api/frames — sorted list of processed image IDs. */
export async function fetchFrameList(): Promise<string[]> {
  const res = await fetch('/api/frames');
  if (!res.ok) return [];
  return res.json();
}

/** GET /api/frame/<image_id> — full frame data for one image. */
export async function fetchRawFrameData(imageId: string): Promise<RawFrameData> {
  const res = await fetch(`/api/frame/${imageId}`);
  if (!res.ok) throw new Error(`Frame not found: ${imageId}`);
  return res.json();
}

/** URL for the raw frame JPEG from data/frames/. */
export function getFrameImageUrl(imageId: string): string {
  return `/api/image/${imageId}`;
}

/** GET /api/vlm/<image_id>/<condition> — VLM inspection response text. */
export async function fetchVlmResponse(
  imageId: string,
  condition = 'anchor_calibrated',
): Promise<string> {
  const res = await fetch(`/api/vlm/${imageId}/${condition}`);
  if (!res.ok) return '';
  const data = await res.json();
  return data.response ?? '';
}

/** POST /api/chat/<image_id> — follow-up question about this frame. */
export async function sendChatMessage(
  imageId: string,
  question: string,
  history: { question: string; answer: string }[],
): Promise<string> {
  const res = await fetch(`/api/chat/${imageId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok) return 'Error reaching server.';
  const data = await res.json();
  return data.response ?? '';
}
