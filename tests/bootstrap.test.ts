import { describe, it, expect } from 'vitest';
import { tna, ftna } from '../src/core/model.js';
import { bootstrapTna, bootstrapWtna } from '../src/analysis/bootstrap.js';
import { buildWtnaMatrix } from '../src/analysis/wtna.js';
import type { BootstrapWtnaInput } from '../src/analysis/bootstrap.js';

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

describe('bootstrapTna', () => {
  const model = tna(seqData);

  it('returns correct structure', () => {
    const result = bootstrapTna(model, { iter: 50, seed: 42 });
    expect(result.labels).toEqual(model.labels);
    expect(result.iter).toBe(50);
    expect(result.method).toBe('stability');
    expect(result.level).toBe(0.05);
    expect(result.edges.length).toBeGreaterThan(0);
    expect(result.weightsMean.length).toBe(model.labels.length ** 2);
    expect(result.weightsSd.length).toBe(model.labels.length ** 2);
    expect(result.weightsBias.length).toBe(model.labels.length ** 2);
  });

  it('edges have valid properties', () => {
    const result = bootstrapTna(model, { iter: 50, seed: 42 });
    for (const edge of result.edges) {
      expect(model.labels).toContain(edge.from);
      expect(model.labels).toContain(edge.to);
      expect(edge.weight).toBeGreaterThan(0);
      expect(edge.pValue).toBeGreaterThanOrEqual(0);
      expect(edge.pValue).toBeLessThanOrEqual(1);
      expect(edge.ciLower).toBeLessThanOrEqual(edge.ciUpper);
      expect(typeof edge.significant).toBe('boolean');
    }
  });

  it('model weights match significant edges', () => {
    const result = bootstrapTna(model, { iter: 50, seed: 42 });
    const a = model.labels.length;
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const sigW = result.model.weights.get(i, j);
        if (sigW > 0) {
          // If significant, weight should equal original
          expect(sigW).toBe(model.weights.get(i, j));
        }
      }
    }
  });

  it('is deterministic with same seed', () => {
    const r1 = bootstrapTna(model, { iter: 20, seed: 123 });
    const r2 = bootstrapTna(model, { iter: 20, seed: 123 });
    expect(Array.from(r1.weightsMean)).toEqual(Array.from(r2.weightsMean));
  });

  it('threshold method works', () => {
    const result = bootstrapTna(model, { iter: 50, seed: 42, method: 'threshold' });
    expect(result.method).toBe('threshold');
    expect(result.edges.length).toBeGreaterThan(0);
  });

  it('throws without data', () => {
    const noDataModel = { ...model, data: null };
    expect(() => bootstrapTna(noDataModel)).toThrow('sequence data');
  });
});

describe('bootstrapWtna', () => {
  const codes = ['A', 'B', 'C'];
  const records = [
    { Actor: 'X', A: 1, B: 0, C: 0 },
    { Actor: 'X', A: 0, B: 1, C: 0 },
    { Actor: 'X', A: 0, B: 0, C: 1 },
    { Actor: 'X', A: 1, B: 1, C: 0 },
    { Actor: 'X', A: 0, B: 1, C: 1 },
    { Actor: 'X', A: 1, B: 0, C: 1 },
    { Actor: 'Y', A: 0, B: 1, C: 0 },
    { Actor: 'Y', A: 1, B: 0, C: 0 },
    { Actor: 'Y', A: 0, B: 0, C: 1 },
    { Actor: 'Y', A: 1, B: 0, C: 1 },
    { Actor: 'Y', A: 0, B: 1, C: 1 },
    { Actor: 'Y', A: 1, B: 1, C: 0 },
  ];

  const wtnaOpts = { actor: 'Actor', windowSize: 2 as const };
  const wtnaResult = buildWtnaMatrix(records, codes, wtnaOpts);

  // Build a model from the WTNA matrix
  const model = ftna(wtnaResult.matrix as any, { labels: codes });

  it('returns correct structure', () => {
    const input: BootstrapWtnaInput = {
      originalModel: model,
      records,
      codes,
      wtnaOpts,
      modelType: 'ftna',
      scaling: null,
    };
    const result = bootstrapWtna(input, { iter: 30, seed: 42 });
    expect(result.labels).toEqual(codes);
    expect(result.edges.length).toBeGreaterThan(0);
    expect(result.weightsMean.length).toBe(9);
  });
});
