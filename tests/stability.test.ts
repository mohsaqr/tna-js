import { describe, it, expect } from 'vitest';
import { tna } from '../src/core/model.js';
import { estimateCS, estimateEdgeStability, estimateNetworkStability } from '../src/analysis/stability.js';

const seqData = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['A', 'C', 'B', 'A', 'C'],
  ['C', 'A', 'B', 'C', 'A'],
  ['B', 'A', 'C', 'B', 'A'],
  ['A', 'B', 'A', 'C', 'B'],
  ['C', 'B', 'A', 'B', 'C'],
  ['B', 'C', 'B', 'A', 'C'],
  ['A', 'A', 'B', 'C', 'A'],
  ['C', 'C', 'A', 'B', 'A'],
];

describe('estimateCS', () => {
  const model = tna(seqData);

  it('returns correct structure', () => {
    const result = estimateCS(model, { iter: 20, seed: 42, measures: ['InStrength', 'OutStrength'] });
    expect(result.dropProps).toEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    expect(result.threshold).toBe(0.7);
    expect(result.certainty).toBe(0.95);
    expect(result.csCoefficients).toHaveProperty('InStrength');
    expect(result.csCoefficients).toHaveProperty('OutStrength');
    expect(result.meanCorrelations).toHaveProperty('InStrength');
    expect(result.meanCorrelations).toHaveProperty('OutStrength');
    expect(result.meanCorrelations['InStrength']!.length).toBe(9);
  });

  it('CS coefficients are in [0, 0.9]', () => {
    const result = estimateCS(model, { iter: 20, seed: 42 });
    for (const val of Object.values(result.csCoefficients)) {
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(0.9);
    }
  });

  it('is deterministic with same seed', () => {
    const r1 = estimateCS(model, { iter: 10, seed: 123, measures: ['InStrength'] });
    const r2 = estimateCS(model, { iter: 10, seed: 123, measures: ['InStrength'] });
    expect(r1.csCoefficients).toEqual(r2.csCoefficients);
  });

  it('spearman correlation method works', () => {
    const result = estimateCS(model, { iter: 10, seed: 42, corrMethod: 'spearman' });
    expect(result.csCoefficients).toBeDefined();
  });

  it('throws without data', () => {
    const noData = { ...model, data: null };
    expect(() => estimateCS(noData)).toThrow('sequence data');
  });
});

describe('estimateEdgeStability', () => {
  const model = tna(seqData);

  it('returns correct structure', () => {
    const result = estimateEdgeStability(model, { iter: 10, seed: 42 });
    expect(result.dropProps.length).toBe(9);
    expect(result.meanCorrelations.length).toBe(9);
    expect(result.csCoefficient).toBeGreaterThanOrEqual(0);
    expect(result.threshold).toBe(0.7);
    expect(result.certainty).toBe(0.95);
  });

  it('correlations decrease as more cases dropped', () => {
    const result = estimateEdgeStability(model, { iter: 20, seed: 42 });
    const corrs = result.meanCorrelations.filter(c => !isNaN(c));
    // Generally, correlations should trend downward
    if (corrs.length > 1) {
      expect(corrs[0]!).toBeGreaterThanOrEqual(corrs[corrs.length - 1]! - 0.1);
    }
  });
});

describe('estimateNetworkStability', () => {
  const model = tna(seqData);

  it('returns correct structure', () => {
    const result = estimateNetworkStability(model, { iter: 10, seed: 42 });
    expect(result.dropProps.length).toBe(9);
    expect(result.densityCorrelations.length).toBe(9);
    expect(result.meanWeightCorrelations.length).toBe(9);
    expect(result.densityCS).toBeGreaterThanOrEqual(0);
    expect(result.meanWeightCS).toBeGreaterThanOrEqual(0);
  });
});
