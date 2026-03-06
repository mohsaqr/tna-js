import { describe, it, expect } from 'vitest';
import {
  spearmanCorr, spearmanCorrArr, kendallTau, distanceCorr, rvCoefficient,
  rankArray, pAdjust,
} from '../src/stats/index.js';

describe('rankArray', () => {
  it('ranks without ties', () => {
    const arr = new Float64Array([3, 1, 4, 2]);
    const ranks = rankArray(arr);
    expect(Array.from(ranks)).toEqual([3, 1, 4, 2]);
  });

  it('handles tied values with average rank', () => {
    const arr = new Float64Array([3, 1, 3, 2]);
    const ranks = rankArray(arr);
    // Ties at 3: positions 3,4 → average 3.5
    expect(Array.from(ranks)).toEqual([3.5, 1, 3.5, 2]);
  });
});

describe('spearmanCorr', () => {
  it('returns 1 for identical vectors', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([1, 2, 3, 4, 5]);
    expect(spearmanCorr(a, b)).toBeCloseTo(1, 10);
  });

  it('returns -1 for reversed vectors', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([5, 4, 3, 2, 1]);
    expect(spearmanCorr(a, b)).toBeCloseTo(-1, 10);
  });

  it('computes rank correlation for non-linear monotonic relationship', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([1, 4, 9, 16, 25]);
    expect(spearmanCorr(a, b)).toBeCloseTo(1, 10);
  });
});

describe('spearmanCorrArr', () => {
  it('works with number arrays', () => {
    const result = spearmanCorrArr([1, 2, 3], [1, 2, 3]);
    expect(result).toBeCloseTo(1, 10);
  });
});

describe('kendallTau', () => {
  it('returns 1 for identical vectors', () => {
    expect(kendallTau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])).toBeCloseTo(1, 10);
  });

  it('returns -1 for reversed vectors', () => {
    expect(kendallTau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])).toBeCloseTo(-1, 10);
  });

  it('handles ties', () => {
    // With ties, tau-b adjusts denominator
    const result = kendallTau([1, 1, 2, 3], [1, 2, 3, 4]);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(1);
  });

  it('returns NaN for length < 2', () => {
    expect(kendallTau([1], [1])).toBeNaN();
  });
});

describe('distanceCorr', () => {
  it('returns ~1 for perfectly correlated data', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8];
    const y = [2, 4, 6, 8, 10, 12, 14, 16];
    expect(distanceCorr(x, y)).toBeCloseTo(1, 4);
  });

  it('returns NaN for length < 2', () => {
    expect(distanceCorr([1], [1])).toBeNaN();
  });
});

describe('rvCoefficient', () => {
  it('returns 1 for identical matrices', () => {
    const mat = {
      rows: 3,
      get(i: number, j: number) {
        return [[1, 0.5, 0.2], [0.3, 1, 0.4], [0.1, 0.6, 1]][i]![j]!;
      },
    };
    expect(rvCoefficient(mat, mat)).toBeCloseTo(1, 10);
  });

  it('returns value between -1 and 1', () => {
    const a = {
      rows: 3,
      get(i: number, j: number) {
        return [[1, 0.5, 0.2], [0.3, 1, 0.4], [0.1, 0.6, 1]][i]![j]!;
      },
    };
    const b = {
      rows: 3,
      get(i: number, j: number) {
        return [[0.5, 0.1, 0.8], [0.2, 0.7, 0.3], [0.9, 0.4, 0.6]][i]![j]!;
      },
    };
    const rv = rvCoefficient(a, b);
    expect(rv).toBeGreaterThanOrEqual(-1);
    expect(rv).toBeLessThanOrEqual(1);
  });
});

describe('pAdjust', () => {
  it('none: returns original values', () => {
    expect(pAdjust([0.01, 0.04, 0.1], 'none')).toEqual([0.01, 0.04, 0.1]);
  });

  it('bonferroni: multiplies by n', () => {
    const result = pAdjust([0.01, 0.04, 0.5], 'bonferroni');
    expect(result[0]).toBeCloseTo(0.03, 10);
    expect(result[1]).toBeCloseTo(0.12, 10);
    expect(result[2]).toBeCloseTo(1.0, 10); // capped at 1
  });

  it('holm: step-down procedure', () => {
    const result = pAdjust([0.01, 0.04, 0.5], 'holm');
    expect(result[0]).toBeCloseTo(0.03, 10);
    expect(result[1]).toBeCloseTo(0.08, 10);
    expect(result[2]).toBeCloseTo(0.5, 10);
  });

  it('fdr/BH: step-up procedure', () => {
    const result = pAdjust([0.01, 0.04, 0.5], 'fdr');
    expect(result[0]).toBeCloseTo(0.03, 10);
    expect(result[1]).toBeCloseTo(0.06, 10);
    expect(result[2]).toBeCloseTo(0.5, 10);
  });

  it('single value: no adjustment needed', () => {
    expect(pAdjust([0.05], 'bonferroni')).toEqual([0.05]);
  });
});
