import type { DemoData, FrameData, AnchorData, MeasurementData } from '@/components/PreCheck/types';

export const CANVAS_WIDTH = 960;
export const CANVAS_HEIGHT = 540;

const STUD_CENTERS = [60, 235, 410, 585, 797, 930];
const STUD_W = 40;
const STUD_TOP = 25;
const STUD_BOT = 500;

const ELEC_BOX = { x: 395, y: 330, w: 48, h: 55 };

const STUD_SPACINGS: { inches: number; compliant: boolean; severity: 'ok' | 'warning' | 'critical' }[] = [
  { inches: 16.0, compliant: true, severity: 'ok' },
  { inches: 16.1, compliant: true, severity: 'ok' },
  { inches: 16.0, compliant: true, severity: 'ok' },
  { inches: 19.3, compliant: false, severity: 'critical' },
  { inches: 15.8, compliant: true, severity: 'ok' },
];

function makeStudAnchors(count: number): AnchorData[] {
  return STUD_CENTERS.slice(0, count).map((cx, i) => ({
    id: i,
    type: 'stud' as const,
    box: [cx - STUD_W / 2, STUD_TOP, cx + STUD_W / 2, STUD_BOT] as [number, number, number, number],
    confidence: 0.88 + Math.sin(i * 1.3) * 0.05 + 0.04,
    label: 'STUD',
  }));
}

function makeElecBoxAnchor(): AnchorData {
  return {
    id: 100,
    type: 'elec_box',
    box: [ELEC_BOX.x, ELEC_BOX.y, ELEC_BOX.x + ELEC_BOX.w, ELEC_BOX.y + ELEC_BOX.h],
    confidence: 0.93,
    label: 'ELEC BOX',
  };
}

function makeStudMeasurements(count: number): MeasurementData[] {
  const out: MeasurementData[] = [];
  for (let i = 0; i < Math.min(count - 1, STUD_SPACINGS.length); i++) {
    const sp = STUD_SPACINGS[i];
    out.push({
      id: `stud_${i}_${i + 1}`,
      type: 'stud_spacing',
      label: `Stud bay ${i + 1} → ${i + 2}`,
      from_anchor: i,
      to_anchor: i + 1,
      cx_a: STUD_CENTERS[i],
      cy_a: 265,
      cx_b: STUD_CENTERS[i + 1],
      cy_b: 265,
      inches: sp.inches,
      compliant: sp.compliant,
      expected: 16.0,
      tolerance: 0.5,
      severity: sp.severity,
    });
  }
  return out;
}

function makeBoxHeightMeasurement(): MeasurementData {
  const bcy = ELEC_BOX.y + ELEC_BOX.h / 2;
  return {
    id: 'box_height_1',
    type: 'box_height',
    label: 'Elec. box 1 height',
    from_anchor: 100,
    to_anchor: -1,
    cx_a: ELEC_BOX.x + ELEC_BOX.w / 2,
    cy_a: bcy,
    cx_b: ELEC_BOX.x + ELEC_BOX.w / 2,
    cy_b: STUD_BOT,
    inches: 11.8,
    compliant: true,
    expected: 12.0,
    tolerance: 1.0,
    severity: 'ok',
  };
}

export function generateDemoData(): DemoData {
  const frames: FrameData[] = [];
  const total = 120;

  for (let i = 1; i <= total; i++) {
    const confNoise = Math.sin(i * 0.3) * 0.03 + (Math.sin(i * 1.7) * 0.005);
    const ppiNoise = Math.sin(i * 0.2) * 0.12;

    let studs: number;
    let showBox = false;
    let showBoxMeas = false;

    if (i <= 15) {
      studs = 3;
    } else if (i <= 35) {
      studs = 5;
    } else if (i <= 60) {
      studs = 5;
      showBox = true;
      showBoxMeas = i >= 45;
    } else {
      studs = 6;
      showBox = true;
      showBoxMeas = true;
    }

    const anchors = makeStudAnchors(studs);
    if (showBox) anchors.push(makeElecBoxAnchor());

    const measurements = makeStudMeasurements(studs);
    if (showBoxMeas) measurements.push(makeBoxHeightMeasurement());

    frames.push({
      frame_id: i,
      calibration: {
        pixels_per_inch: +(18.69 + ppiNoise).toFixed(2),
        confidence: +Math.max(0.85, Math.min(0.98, 0.94 + confNoise)).toFixed(3),
        method: 'stud_face_3.5in',
        method_label: '2×4 stud face\n(3.5″ known width)',
      },
      anchors,
      measurements,
    });
  }

  return { total_frames: total, fps: 1, frames };
}
