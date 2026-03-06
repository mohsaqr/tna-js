import { describe, it, expect } from 'vitest';
import {
  buildWtnaMatrix, toBinaryMatrix, applyWindowing, applyIntervalWindowing,
  computeWtnaTransitions, computeWithinWindow, rowNormalizeWtna,
} from '../src/analysis/wtna.js';

const codes = ['A', 'B', 'C'];

function makeRecords(data: number[][]): Record<string, number>[] {
  return data.map(row => ({ A: row[0]!, B: row[1]!, C: row[2]! }));
}

describe('toBinaryMatrix', () => {
  it('converts records to binary matrix', () => {
    const records = [{ A: 1, B: 0, C: 1 }, { A: 0, B: 1, C: 0 }];
    expect(toBinaryMatrix(records, codes)).toEqual([[1, 0, 1], [0, 1, 0]]);
  });

  it('handles string values', () => {
    const records = [{ A: '1', B: '0', C: '1' }];
    expect(toBinaryMatrix(records, codes)).toEqual([[1, 0, 1]]);
  });
});

describe('applyWindowing', () => {
  it('tumbling: groups into non-overlapping blocks', () => {
    const X = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]];
    const result = applyWindowing(X, 2, 'tumbling');
    expect(result).toEqual([[1, 1, 0], [1, 1, 1]]);
  });

  it('sliding: overlapping windows', () => {
    const X = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const result = applyWindowing(X, 2, 'sliding');
    expect(result.length).toBe(2);
    expect(result[0]).toEqual([1, 1, 0]);
    expect(result[1]).toEqual([0, 1, 1]);
  });

  it('windowSize <= 1: returns input', () => {
    const X = [[1, 0], [0, 1]];
    expect(applyWindowing(X, 1, 'tumbling')).toEqual(X);
  });
});

describe('applyIntervalWindowing', () => {
  it('groups consecutive same-label rows', () => {
    const X = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]];
    const labels = ['AM', 'AM', 'PM', 'PM'];
    const result = applyIntervalWindowing(X, labels);
    expect(result.length).toBe(2);
    expect(result[0]).toEqual([1, 1, 0]); // OR of AM rows
    expect(result[1]).toEqual([1, 0, 1]); // OR of PM rows
  });

  it('reappearing label creates new window', () => {
    const X = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const labels = ['AM', 'PM', 'AM'];
    const result = applyIntervalWindowing(X, labels);
    expect(result.length).toBe(3);
  });
});

describe('computeWtnaTransitions', () => {
  it('computes transitions between consecutive windows', () => {
    const W = [[1, 0, 1], [0, 1, 0], [1, 1, 0]];
    const T = computeWtnaTransitions(W);
    // T[i,j] = Σ_t W[t,i]*W[t+1,j]
    // t=0: W[0]=[1,0,1], W[1]=[0,1,0] → T[0][1]+=1, T[2][1]+=1
    // t=1: W[1]=[0,1,0], W[2]=[1,1,0] → T[1][0]+=1, T[1][1]+=1
    expect(T[0]![1]).toBe(1);
    expect(T[2]![1]).toBe(1);
    expect(T[1]![0]).toBe(1);
    expect(T[1]![1]).toBe(1);
  });

  it('returns zero matrix for single window', () => {
    const T = computeWtnaTransitions([[1, 0, 1]]);
    expect(T).toEqual([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
  });
});

describe('computeWithinWindow', () => {
  it('computes co-occurrence with zero diagonal', () => {
    const W = [[1, 1, 0], [0, 1, 1]];
    const C = computeWithinWindow(W);
    expect(C[0]![1]).toBe(1); // A,B co-occur in row 0
    expect(C[1]![0]).toBe(1);
    expect(C[1]![2]).toBe(1); // B,C co-occur in row 1
    expect(C[0]![0]).toBe(0); // diagonal = 0
  });
});

describe('rowNormalizeWtna', () => {
  it('normalizes rows to sum to 1', () => {
    const M = [[2, 3, 5], [0, 0, 0]];
    const result = rowNormalizeWtna(M);
    expect(result[0]).toEqual([0.2, 0.3, 0.5]);
    expect(result[1]).toEqual([0, 0, 0]); // zero row unchanged
  });
});

describe('buildWtnaMatrix', () => {
  it('builds frequency matrix without actor grouping', () => {
    const records = makeRecords([
      [1, 0, 0], [0, 1, 0], [0, 0, 1],
      [1, 1, 0], [0, 1, 1], [1, 0, 1],
    ]);
    const result = buildWtnaMatrix(records, codes, { windowSize: 2 });
    expect(result.labels).toEqual(codes);
    expect(result.matrix.length).toBe(3);
    expect(result.withinMatrix.length).toBe(3);
  });

  it('builds relative matrix', () => {
    const records = makeRecords([
      [1, 0, 0], [0, 1, 0], [0, 0, 1],
      [1, 1, 0], [0, 1, 1], [1, 0, 1],
    ]);
    const result = buildWtnaMatrix(records, codes, { windowSize: 2, type: 'relative' });
    // Each row should sum to ~1 (or 0 for empty rows)
    for (const row of result.matrix) {
      const sum = row.reduce((a, b) => a + b, 0);
      if (sum > 0) expect(sum).toBeCloseTo(1, 10);
    }
  });

  it('groups by actor', () => {
    const records = [
      { Actor: 'X', A: 1, B: 0, C: 0 },
      { Actor: 'X', A: 0, B: 1, C: 0 },
      { Actor: 'X', A: 0, B: 0, C: 1 },
      { Actor: 'Y', A: 0, B: 1, C: 0 },
      { Actor: 'Y', A: 1, B: 0, C: 0 },
      { Actor: 'Y', A: 0, B: 0, C: 1 },
    ];
    const result = buildWtnaMatrix(records, codes, { actor: 'Actor', windowSize: 2 });
    expect(result.labels).toEqual(codes);
    // Transitions accumulated from both actors
    const totalTrans = result.matrix.reduce(
      (s, row) => s + row.reduce((a, b) => a + b, 0), 0,
    );
    expect(totalTrans).toBeGreaterThan(0);
  });
});
