import { describe, it, expect } from 'vitest';
import { tna } from '../src/core/model.js';
import { compareModels } from '../src/analysis/compare.js';
import { RELIABILITY_METRICS } from '../src/analysis/reliability.js';

const seqA = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['A', 'C', 'B', 'A', 'C'],
];

const seqB = [
  ['C', 'A', 'B', 'C', 'A'],
  ['B', 'A', 'C', 'B', 'A'],
  ['A', 'B', 'A', 'C', 'B'],
];

describe('compareModels', () => {
  const modelA = tna(seqA);
  const modelB = tna(seqB);

  it('returns all 22 metrics', () => {
    const result = compareModels(modelA, modelB);
    expect(Object.keys(result).length).toBe(22);
    for (const m of RELIABILITY_METRICS) {
      expect(result).toHaveProperty(m.key);
    }
  });

  it('identical models produce perfect scores', () => {
    const result = compareModels(modelA, modelA);
    expect(result['pearson']).toBeCloseTo(1, 10);
    expect(result['mad']).toBeCloseTo(0, 10);
  });

  it('different models produce non-trivial metrics', () => {
    const result = compareModels(modelA, modelB);
    expect(result['mad']!).toBeGreaterThan(0);
    expect(result['euclidean']!).toBeGreaterThan(0);
  });
});
