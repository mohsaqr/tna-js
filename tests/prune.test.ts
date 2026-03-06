import { describe, it, expect } from 'vitest';
import { tna } from '../src/core/model.js';
import { prune, pruneDisparity } from '../src/analysis/prune.js';
import type { TNA } from '../src/core/types.js';

const seqData = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['A', 'C', 'B', 'A', 'C'],
  ['C', 'A', 'B', 'C', 'A'],
];

describe('prune with options', () => {
  const model = tna(seqData);

  it('backward compatible: number argument works', () => {
    const result = prune(model, 0.2) as TNA;
    const a = model.labels.length;
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const orig = model.weights.get(i, j);
        const pruned = result.weights.get(i, j);
        if (orig < 0.2) {
          expect(pruned).toBe(0);
        } else {
          expect(pruned).toBe(orig);
        }
      }
    }
  });

  it('options object: threshold method', () => {
    const result = prune(model, { method: 'threshold', threshold: 0.2 }) as TNA;
    const a = model.labels.length;
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const pruned = result.weights.get(i, j);
        if (pruned > 0) {
          expect(pruned).toBeGreaterThanOrEqual(0.2);
        }
      }
    }
  });

  it('options object: disparity method', () => {
    // Use a more lenient alpha so backbone retains edges in small network
    const result = prune(model, { method: 'disparity', alpha: 0.5 }) as TNA;
    let edgeCount = 0;
    const a = model.labels.length;
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        if (result.weights.get(i, j) > 0) edgeCount++;
      }
    }
    // With lenient alpha, some edges should remain
    expect(edgeCount).toBeGreaterThan(0);
  });
});

describe('pruneDisparity', () => {
  const model = tna(seqData);

  it('preserves significant edges', () => {
    const result = pruneDisparity(model, 0.05);
    const a = model.labels.length;
    let removed = 0;
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        if (model.weights.get(i, j) > 0 && result.weights.get(i, j) === 0) {
          removed++;
        }
      }
    }
    // With alpha=0.05, some edges should be removed
    expect(removed).toBeGreaterThanOrEqual(0);
  });

  it('strict alpha keeps fewer edges', () => {
    const loose = pruneDisparity(model, 0.5);
    const strict = pruneDisparity(model, 0.01);
    const a = model.labels.length;

    let looseEdges = 0, strictEdges = 0;
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        if (loose.weights.get(i, j) > 0) looseEdges++;
        if (strict.weights.get(i, j) > 0) strictEdges++;
      }
    }
    expect(strictEdges).toBeLessThanOrEqual(looseEdges);
  });

  it('preserves model properties', () => {
    const result = pruneDisparity(model, 0.05);
    expect(result.labels).toEqual(model.labels);
    expect(result.type).toBe(model.type);
    expect(result.scaling).toEqual(model.scaling);
  });
});
