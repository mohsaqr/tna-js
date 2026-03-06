import { describe, it, expect } from 'vitest';
import { tna } from '../src/core/model.js';
import {
  compareWeightMatrices, reliabilityAnalysis, RELIABILITY_METRICS,
} from '../src/analysis/reliability.js';

const seqData = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['A', 'C', 'B', 'A', 'C'],
  ['C', 'A', 'B', 'C', 'A'],
  ['B', 'A', 'C', 'B', 'A'],
  ['A', 'B', 'A', 'C', 'B'],
  ['C', 'B', 'A', 'B', 'C'],
  ['B', 'C', 'B', 'A', 'C'],
];

describe('compareWeightMatrices', () => {
  it('returns all 22 metrics', () => {
    const modelA = tna(seqData.slice(0, 4));
    const modelB = tna(seqData.slice(4));
    const result = compareWeightMatrices(modelA, modelB);
    expect(Object.keys(result).length).toBe(22);
    for (const m of RELIABILITY_METRICS) {
      expect(result).toHaveProperty(m.key);
      expect(typeof result[m.key]).toBe('number');
    }
  });

  it('identical models: pearson=1, mad=0, cosine=1', () => {
    const model = tna(seqData);
    const result = compareWeightMatrices(model, model);
    expect(result['pearson']).toBeCloseTo(1, 10);
    expect(result['spearman']).toBeCloseTo(1, 10);
    expect(result['cosine']).toBeCloseTo(1, 10);
    expect(result['mad']).toBeCloseTo(0, 10);
    expect(result['euclidean']).toBeCloseTo(0, 10);
    expect(result['rv']).toBeCloseTo(1, 5);
    expect(result['sign_agree']).toBeCloseTo(1, 10);
  });

  it('returns NaN for mismatched sizes', () => {
    const model3 = tna([['A', 'B', 'C']]);
    const model2 = tna([['X', 'Y', 'X']]);
    const result = compareWeightMatrices(model3, model2);
    for (const m of RELIABILITY_METRICS) {
      expect(result[m.key]).toBeNaN();
    }
  });

  it('correlations are bounded', () => {
    const modelA = tna(seqData.slice(0, 4));
    const modelB = tna(seqData.slice(4));
    const result = compareWeightMatrices(modelA, modelB);
    expect(result['pearson']!).toBeGreaterThanOrEqual(-1 - 1e-10);
    expect(result['pearson']!).toBeLessThanOrEqual(1 + 1e-10);
    expect(result['spearman']!).toBeGreaterThanOrEqual(-1 - 1e-10);
    expect(result['spearman']!).toBeLessThanOrEqual(1 + 1e-10);
  });

  it('dissimilarities are non-negative', () => {
    const modelA = tna(seqData.slice(0, 4));
    const modelB = tna(seqData.slice(4));
    const result = compareWeightMatrices(modelA, modelB);
    expect(result['euclidean']!).toBeGreaterThanOrEqual(0);
    expect(result['manhattan']!).toBeGreaterThanOrEqual(0);
    expect(result['braycurtis']!).toBeGreaterThanOrEqual(0);
  });
});

describe('reliabilityAnalysis', () => {
  it('returns correct structure', () => {
    const result = reliabilityAnalysis(seqData, 'tna', { iter: 10, seed: 42 });
    expect(result.iter).toBe(10);
    expect(result.split).toBe(0.5);
    expect(result.modelType).toBe('tna');
    expect(result.summary.length).toBe(22);
    for (const m of RELIABILITY_METRICS) {
      expect(result.iterations).toHaveProperty(m.key);
      expect(result.iterations[m.key]!.length).toBe(10);
    }
  });

  it('summary has valid statistics', () => {
    const result = reliabilityAnalysis(seqData, 'tna', { iter: 20, seed: 42 });
    for (const s of result.summary) {
      expect(typeof s.mean).toBe('number');
      expect(typeof s.sd).toBe('number');
      expect(typeof s.median).toBe('number');
      expect(s.min).toBeLessThanOrEqual(s.max);
      expect(s.q25).toBeLessThanOrEqual(s.q75);
    }
  });

  it('is deterministic with same seed', () => {
    const r1 = reliabilityAnalysis(seqData, 'tna', { iter: 5, seed: 123 });
    const r2 = reliabilityAnalysis(seqData, 'tna', { iter: 5, seed: 123 });
    expect(r1.iterations['pearson']).toEqual(r2.iterations['pearson']);
  });

  it('works with ftna', () => {
    const result = reliabilityAnalysis(seqData, 'ftna', { iter: 5 });
    expect(result.modelType).toBe('ftna');
    expect(result.summary.length).toBe(22);
  });

  it('throws for < 4 sequences', () => {
    expect(() => reliabilityAnalysis(seqData.slice(0, 3), 'tna')).toThrow('at least 4');
  });
});
