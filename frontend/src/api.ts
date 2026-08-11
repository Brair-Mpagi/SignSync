/**
 * SignSync API client.
 *
 * The response types mirror `signsync/api/schemas.py`. Every response carries
 * `warnings`, and `Warning` is non-optional on purpose: plan §16.3 requires the
 * product to be transparent about its limits, and a required field is much harder
 * for a component author to forget than an optional one.
 */

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export interface Warning {
  code: string;
  message: string;
}

export interface Capabilities {
  recognition: boolean;
  speech_input: boolean;
  speech_output: boolean;
  avatar: boolean;
  validated_lexicon: boolean;
}

export interface Health {
  status: string;
  capabilities: Capabilities;
  warnings: Warning[];
  disclaimer: string;
}

export interface Joint {
  name: string;
  parent: string | null;
  offset: [number, number, number];
}

export interface Rig {
  version: number;
  joints: Joint[];
  faceChannels: string[];
}

export interface Frame {
  rotations: number[][];
  root: [number, number, number];
  face: Record<string, number>;
}

export interface AnimationData {
  version: number;
  fps: number;
  duration: number;
  glosses: string[];
  segments: { start: number; end: number; gloss: string }[];
  frames: Frame[];
  rig?: Rig;
}

export interface SignToEnglishResponse {
  text: string;
  glosses: string[];
  confidence: number;
  /** True when the signer should be asked to repeat rather than trusting the output. */
  needsRepeat: boolean;
  frame: string;
  unresolved: string[];
  predictions: { gloss: string; confidence: number; start: number; end: number }[];
  speech: {
    engine: string;
    /** Whether sound was actually produced. Never infer this from the request. */
    audible: boolean;
    duration: number;
    detail: string;
    pcm16?: string;
    sampleRate?: number;
  };
  warnings: Warning[];
}

export interface EnglishToSignResponse {
  transcript: string;
  transcriptConfidence: number;
  glosses: string[];
  notation: string;
  frame: string;
  markers: {
    marker: string;
    start: number;
    end: number;
    intensity: number;
    scope: string[];
  }[];
  animation: AnimationData;
  /** Glosses with no motion available. Show these as text; do not approximate. */
  missing: string[];
  /** Glosses rendered from generated motion rather than a recording of a signer. */
  generated: string[];
  complete: boolean;
  warnings: Warning[];
}

export interface Metrics {
  stages: { stage: string; count: number; mean_ms: number; p50_ms: number; p95_ms: number }[];
  total_p95_ms: number;
  /** Whether the round trip meets objective O11 (under two seconds). */
  meets_o11: boolean;
  bottleneck: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error ?? `${path} failed with ${response.status}`);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>('/health'),
  rig: () => request<Rig>('/api/rig'),
  metrics: () => request<Metrics>('/api/metrics'),

  signToEnglish: (glosses: string[], markers: string[] = [], speak = true) =>
    request<SignToEnglishResponse>('/api/sign-to-english', {
      method: 'POST',
      body: JSON.stringify({ glosses, markers, speak }),
    }),

  englishToSign: (text: string) =>
    request<EnglishToSignResponse>('/api/english-to-sign', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),
};

/**
 * Stream landmark frames to the server (Mode A).
 *
 * Landmarks rather than video: a few KB/s instead of a stream, and the camera
 * feed never leaves the participant's device (plan §16, §17).
 */
export function openSignSocket(handlers: {
  onSign?: (message: { gloss: string; confidence: number; start: number; end: number }) => void;
  onTranslation?: (message: SignToEnglishResponse) => void;
  onError?: (error: string) => void;
}): WebSocket {
  const socket = new WebSocket(`${BASE.replace(/^http/, 'ws')}/ws/sign`);
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'sign') handlers.onSign?.(message);
    else if (message.type === 'translation') handlers.onTranslation?.(message);
    else if (message.type === 'error') handlers.onError?.(message.error);
  };
  return socket;
}
