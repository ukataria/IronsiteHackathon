export interface AnchorData {
  id: number;
  type: 'stud' | 'cmu' | 'rebar' | 'elec_box';
  box: [number, number, number, number];
  confidence: number;
  label: string;
}

export interface MeasurementData {
  id: string;
  type: 'stud_spacing' | 'box_height';
  label: string;
  from_anchor: number;
  to_anchor: number;
  cx_a: number;
  cy_a: number;
  cx_b: number;
  cy_b: number;
  inches: number;
  compliant: boolean;
  expected: number;
  tolerance: number;
  severity: 'ok' | 'warning' | 'critical';
}

export interface CalibrationData {
  pixels_per_inch: number;
  confidence: number;
  method: string;
  method_label: string;
}

export interface FrameData {
  frame_id: number;
  calibration: CalibrationData;
  anchors: AnchorData[];
  measurements: MeasurementData[];
}

export interface DemoData {
  total_frames: number;
  fps: number;
  frames: FrameData[];
}

export interface Finding {
  id: string;
  type: 'compliant' | 'violation' | 'info';
  label: string;
  value: string;
  detail: string;
  frame_id: number;
  measurement_id?: string;
  severity?: string;
  expected?: number;
  tolerance?: number;
  delta?: number;
}

export interface AlertData {
  id: string;
  label: string;
  measured: number;
  expected: number;
  tolerance: number;
  delta: number;
  severity: string;
  frame_id: number;
  timestamp: number;
}

export interface QAExchange {
  question: string;
  answer: string;
}
