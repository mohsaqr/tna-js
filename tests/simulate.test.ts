import { describe, it, expect } from 'vitest';
import { tna } from '../src/core/model.js';
import { simulate } from '../src/analysis/simulate.js';

const seqData = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['A', 'C', 'B', 'A', 'C'],
  ['C', 'A', 'B', 'C', 'A'],
];

describe('simulate', () => {
  const model = tna(seqData);

  it('generates correct number of sequences', () => {
    const result = simulate(model, { n: 10, seqLength: 5, seed: 42 });
    expect(result.length).toBe(10);
  });

  it('sequences have correct length', () => {
    const result = simulate(model, { n: 5, seqLength: 8, seed: 42 });
    for (const seq of result) {
      expect(seq.length).toBe(8);
    }
  });

  it('sequences contain only valid labels', () => {
    const result = simulate(model, { n: 20, seqLength: 10, seed: 42 });
    const validLabels = new Set(model.labels);
    for (const seq of result) {
      for (const state of seq) {
        expect(validLabels.has(state as string)).toBe(true);
      }
    }
  });

  it('supports variable length range', () => {
    const result = simulate(model, { n: 50, seqLength: [3, 10], seed: 42 });
    expect(result.length).toBe(50);
    const lengths = result.map(s => s.length);
    expect(Math.min(...lengths)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...lengths)).toBeLessThanOrEqual(10);
  });

  it('is deterministic with same seed', () => {
    const r1 = simulate(model, { n: 5, seqLength: 5, seed: 123 });
    const r2 = simulate(model, { n: 5, seqLength: 5, seed: 123 });
    expect(r1).toEqual(r2);
  });

  it('returns empty for empty labels', () => {
    const emptyModel = { ...model, labels: [], weights: model.weights, inits: new Float64Array(0) };
    const result = simulate(emptyModel);
    expect(result).toEqual([]);
  });
});
