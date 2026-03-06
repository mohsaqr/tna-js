import { describe, it, expect } from 'vitest';
import { tna } from '../src/core/model.js';
import { permutationTest } from '../src/analysis/permutation.js';

const seqX = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['A', 'C', 'B', 'A', 'C'],
  ['C', 'A', 'B', 'C', 'A'],
];

const seqY = [
  ['B', 'A', 'C', 'B', 'A'],
  ['A', 'B', 'A', 'C', 'B'],
  ['C', 'B', 'A', 'B', 'C'],
  ['B', 'C', 'B', 'A', 'C'],
];

describe('permutationTest', () => {
  const modelX = tna(seqX);
  const modelY = tna(seqY);

  it('returns correct structure', () => {
    const result = permutationTest(modelX, modelY, { iter: 50, seed: 42 });
    const a = modelX.labels.length;
    expect(result.labels).toEqual(modelX.labels);
    expect(result.nStates).toBe(a);
    expect(result.level).toBe(0.05);
    expect(result.diffTrue.length).toBe(a * a);
    expect(result.diffSig.length).toBe(a * a);
    expect(result.pValues.length).toBe(a * a);
    expect(result.edgeStats.length).toBe(a * a);
  });

  it('p-values are in [0, 1]', () => {
    const result = permutationTest(modelX, modelY, { iter: 50, seed: 42 });
    for (let i = 0; i < result.pValues.length; i++) {
      expect(result.pValues[i]!).toBeGreaterThanOrEqual(0);
      expect(result.pValues[i]!).toBeLessThanOrEqual(1);
    }
  });

  it('is deterministic with same seed', () => {
    const r1 = permutationTest(modelX, modelY, { iter: 20, seed: 123 });
    const r2 = permutationTest(modelX, modelY, { iter: 20, seed: 123 });
    expect(Array.from(r1.pValues)).toEqual(Array.from(r2.pValues));
  });

  it('diffSig only contains significant differences', () => {
    const result = permutationTest(modelX, modelY, { iter: 50, seed: 42 });
    for (let i = 0; i < result.diffSig.length; i++) {
      if (result.pValues[i]! >= result.level) {
        expect(result.diffSig[i]).toBe(0);
      }
    }
  });

  it('p-value adjustment works', () => {
    const raw = permutationTest(modelX, modelY, { iter: 50, seed: 42, adjust: 'none' });
    const adj = permutationTest(modelX, modelY, { iter: 50, seed: 42, adjust: 'bonferroni' });
    // Adjusted p-values should be >= raw p-values
    for (let i = 0; i < raw.pValues.length; i++) {
      expect(adj.pValues[i]!).toBeGreaterThanOrEqual(raw.pValues[i]! - 1e-10);
    }
  });

  it('throws for mismatched labels', () => {
    const modelZ = tna([['X', 'Y', 'Z', 'X']]);
    expect(() => permutationTest(modelX, modelZ)).toThrow('same state labels');
  });

  it('throws without data', () => {
    const noData = { ...modelX, data: null };
    expect(() => permutationTest(noData, modelY)).toThrow('sequence data');
  });

  it('paired mode works with equal sizes', () => {
    const result = permutationTest(modelX, modelY, { iter: 20, seed: 42, paired: true });
    expect(result.edgeStats.length).toBeGreaterThan(0);
  });
});
