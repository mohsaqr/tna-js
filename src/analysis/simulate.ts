/**
 * Simulate sequence data from a TNA model via Markov chain.
 */
import type { TNA, SequenceData } from '../core/types.js';
import { SeededRNG } from '../core/rng.js';

export interface SimulateOptions {
  /** Number of sequences to generate (default 50). */
  n?: number;
  /** Sequence length: fixed number or [min, max] range (default 20). */
  seqLength?: number | [number, number];
  /** Random seed (default 42). */
  seed?: number;
}

/**
 * Simulate sequences by walking the Markov chain defined by the model.
 * Initial states are sampled from model.inits; transitions use model.weights rows.
 */
export function simulate(
  model: TNA,
  options: SimulateOptions = {},
): SequenceData {
  const { n = 50, seqLength = 20, seed = 42 } = options;
  const rng = new SeededRNG(seed);
  const labels = model.labels;
  const a = labels.length;
  if (a === 0) return [];

  const [minLen, maxLen] = typeof seqLength === 'number'
    ? [seqLength, seqLength]
    : seqLength;

  // Build cumulative probability tables for efficient sampling
  const initsCum = buildCumulative(model.inits, a);
  const weightsCum: Float64Array[] = [];
  for (let i = 0; i < a; i++) {
    const row = new Float64Array(a);
    for (let j = 0; j < a; j++) row[j] = model.weights.get(i, j);
    weightsCum.push(buildCumulative(row, a));
  }

  const result: SequenceData = [];
  for (let s = 0; s < n; s++) {
    const len = minLen === maxLen
      ? minLen
      : minLen + Math.floor(rng.random() * (maxLen - minLen + 1));

    const seq: (string | null)[] = [];
    let state = sampleCumulative(initsCum, rng);

    for (let t = 0; t < len; t++) {
      seq.push(labels[state]!);
      state = sampleCumulative(weightsCum[state]!, rng);
    }
    result.push(seq);
  }

  return result;
}

/** Build cumulative probability array from a probability vector. */
function buildCumulative(probs: Float64Array, n: number): Float64Array {
  const cum = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += probs[i]!;
    cum[i] = sum;
  }
  // Normalize in case of rounding
  if (sum > 0) {
    for (let i = 0; i < n; i++) cum[i]! /= sum;
  }
  return cum;
}

/** Sample from cumulative probability array. */
function sampleCumulative(cum: Float64Array, rng: SeededRNG): number {
  const u = rng.random();
  for (let i = 0; i < cum.length; i++) {
    if (u < cum[i]!) return i;
  }
  return cum.length - 1;
}
