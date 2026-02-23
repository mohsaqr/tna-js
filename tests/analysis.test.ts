import { describe, it, expect } from 'vitest';
import { prune } from '../src/analysis/prune.js';
import { cliques } from '../src/analysis/cliques.js';
import { communities } from '../src/analysis/communities.js';
import { compareSequences } from '../src/analysis/compare.js';
import { clusterSequences, clusterData } from '../src/analysis/cluster.js';
import { tna } from '../src/core/model.js';
import { groupTna } from '../src/core/group.js';
import type { SequenceData, CommunityResult } from '../src/core/types.js';
import fixture from './fixtures/ground_truth.json';

const smallData: SequenceData = fixture.small_data;

describe('prune', () => {
  it('removes edges below threshold (R ground truth)', () => {
    const model = tna(smallData);
    const pruned = prune(model, 0.05);
    if ('weights' in pruned) {
      const actual = pruned.weights.to2D();
      const expected = fixture.pruned_weights_005;
      for (let i = 0; i < actual.length; i++) {
        for (let j = 0; j < actual[i]!.length; j++) {
          expect(actual[i]![j]).toBeCloseTo(expected[i]![j]!, 10);
        }
      }
    }
  });

  it('sets sub-threshold edges to zero', () => {
    const model = tna(smallData);
    const pruned = prune(model, 0.3);
    if ('weights' in pruned) {
      const w = pruned.weights;
      for (let i = 0; i < w.rows; i++) {
        for (let j = 0; j < w.cols; j++) {
          expect(w.get(i, j) === 0 || w.get(i, j) >= 0.3).toBe(true);
        }
      }
    }
  });
});

describe('cliques', () => {
  it('finds directed cliques (R ground truth)', () => {
    const model = tna(smallData);
    const result = cliques(model);
    if ('labels' in result) {
      const foundLabels = result.labels.map((c) => [...c].sort());
      const expectedLabels = fixture.clique_labels.map((c: string[]) => [...c].sort());
      expect(foundLabels.length).toBe(expectedLabels.length);

      for (const expected of expectedLabels) {
        const found = foundLabels.some(
          (f) => f.length === expected.length && f.every((v, i) => v === expected[i]),
        );
        expect(found).toBe(true);
      }
    }
  });

  it('returns correct number of cliques', () => {
    const model = tna(smallData);
    const result = cliques(model);
    if ('labels' in result) {
      expect(result.labels.length).toBe(14);
    }
  });
});

describe('communities', () => {
  it('detects communities with leading_eigen', () => {
    const model = tna(smallData);
    const result = communities(model) as CommunityResult;

    const commAssign = result.assignments['leading_eigen']!;
    expect(commAssign.length).toBe(fixture.labels.length);
    // Should find at least 2 communities
    const numComms = new Set(commAssign).size;
    expect(numComms).toBeGreaterThanOrEqual(2);
    // All assignments should be valid non-negative integers
    for (const v of commAssign) {
      expect(typeof v).toBe('number');
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  it('detects communities with louvain', () => {
    const model = tna(smallData);
    const result = communities(model, { methods: 'louvain' }) as CommunityResult;
    const commAssign = result.assignments['louvain']!;
    expect(commAssign.length).toBe(fixture.labels.length);
    for (const v of commAssign) {
      expect(typeof v).toBe('number');
    }
  });

  it('detects communities with label_prop', () => {
    const model = tna(smallData);
    const result = communities(model, { methods: 'label_prop' }) as CommunityResult;
    const commAssign = result.assignments['label_prop']!;
    expect(commAssign.length).toBe(fixture.labels.length);
    for (const v of commAssign) {
      expect(typeof v).toBe('number');
    }
  });

  it('reports community counts', () => {
    const model = tna(smallData);
    const result = communities(model) as CommunityResult;
    expect(result.counts['leading_eigen']).toBeGreaterThanOrEqual(2);
  });
});

describe('clusterSequences', () => {
  const clusterData: SequenceData = smallData.slice(0, 10);

  it('clusters with hamming distance', () => {
    const result = clusterSequences(clusterData, 2);
    expect(result.k).toBe(2);
    expect(result.assignments.length).toBe(10);
    expect(result.sizes.length).toBe(2);
    expect(result.sizes.reduce((a, b) => a + b, 0)).toBe(10);
    expect(typeof result.silhouette).toBe('number');
  });

  it('clusters with levenshtein distance', () => {
    const result = clusterSequences(clusterData, 2, { dissimilarity: 'lv' });
    expect(result.assignments.length).toBe(10);
    expect(result.dissimilarity).toBe('lv');
  });

  it('clusters with osa distance', () => {
    const result = clusterSequences(clusterData, 2, { dissimilarity: 'osa' });
    expect(result.assignments.length).toBe(10);
  });

  it('clusters with lcs distance', () => {
    const result = clusterSequences(clusterData, 2, { dissimilarity: 'lcs' });
    expect(result.assignments.length).toBe(10);
  });

  it('clusters with hierarchical method', () => {
    const result = clusterSequences(clusterData, 3, { method: 'hierarchical' });
    expect(result.k).toBe(3);
    expect(result.sizes.length).toBe(3);
    expect(result.method).toBe('hierarchical');
  });

  it('throws for k < 2', () => {
    expect(() => clusterSequences(clusterData, 1)).toThrow();
  });

  it('throws for k > n', () => {
    expect(() => clusterSequences(clusterData, 100)).toThrow();
  });
});

// ---- R ground truth clustering tests ----
// 6 sequences: [A,B,C], [C,A,B], [A,B,A], [B,C,A], [A,C,B], [C,B,A]
// R encoding (seq2chr): A→a, C→b, B→c → "acb","bac","aca","cba","abc","bca"
// All ground truth values generated with R stringdist + cluster packages.

const clusterTestData: SequenceData = [
  ['A', 'B', 'C'],
  ['C', 'A', 'B'],
  ['A', 'B', 'A'],
  ['B', 'C', 'A'],
  ['A', 'C', 'B'],
  ['C', 'B', 'A'],
];

// R distance matrices (exact values from stringdist)
const R_DISTANCES: Record<string, number[][]> = {
  hamming: [
    [0, 3, 1, 3, 2, 2],
    [3, 0, 3, 3, 2, 2],
    [1, 3, 0, 2, 2, 1],
    [3, 3, 2, 0, 2, 2],
    [2, 2, 2, 2, 0, 3],
    [2, 2, 1, 2, 3, 0],
  ],
  lv: [
    [0, 2, 1, 2, 2, 2],
    [2, 0, 2, 2, 2, 2],
    [1, 2, 0, 2, 2, 1],
    [2, 2, 2, 0, 2, 2],
    [2, 2, 2, 2, 0, 2],
    [2, 2, 1, 2, 2, 0],
  ],
  osa: [
    [0, 2, 1, 2, 1, 2],
    [2, 0, 2, 2, 1, 1],
    [1, 2, 0, 2, 2, 1],
    [2, 2, 2, 0, 2, 1],
    [1, 1, 2, 2, 0, 2],
    [2, 1, 1, 1, 2, 0],
  ],
  dl: [
    [0, 2, 1, 2, 1, 2],
    [2, 0, 2, 2, 1, 1],
    [1, 2, 0, 2, 2, 1],
    [2, 2, 2, 0, 2, 1],
    [1, 1, 2, 2, 0, 2],
    [2, 1, 1, 1, 2, 0],
  ],
  lcs: [
    [0, 2, 2, 2, 2, 4],
    [2, 0, 2, 2, 2, 2],
    [2, 2, 0, 2, 2, 2],
    [2, 2, 2, 0, 4, 2],
    [2, 2, 2, 4, 0, 2],
    [4, 2, 2, 2, 2, 0],
  ],
  qgram: [
    [0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [2, 2, 0, 2, 2, 2],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
  ],
  cosine: [
    [0, 0, 0.225403330758517, 0, 0, 0],
    [0, 0, 0.225403330758517, 0, 0, 0],
    [0.225403330758517, 0.225403330758517, 0, 0.225403330758517, 0.225403330758517, 0.225403330758517],
    [0, 0, 0.225403330758517, 0, 0, 0],
    [0, 0, 0.225403330758517, 0, 0, 0],
    [0, 0, 0.225403330758517, 0, 0, 0],
  ],
  jaccard: [
    [0, 0, 1 / 3, 0, 0, 0],
    [0, 0, 1 / 3, 0, 0, 0],
    [1 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3],
    [0, 0, 1 / 3, 0, 0, 0],
    [0, 0, 1 / 3, 0, 0, 0],
    [0, 0, 1 / 3, 0, 0, 0],
  ],
  jw: [
    [0, 1, 2 / 9, 1, 4 / 9, 4 / 9],
    [1, 0, 1, 1, 4 / 9, 4 / 9],
    [2 / 9, 1, 0, 4 / 9, 4 / 9, 2 / 9],
    [1, 1, 4 / 9, 0, 4 / 9, 4 / 9],
    [4 / 9, 4 / 9, 4 / 9, 4 / 9, 0, 1],
    [4 / 9, 4 / 9, 2 / 9, 4 / 9, 1, 0],
  ],
};

const R_WEIGHTED_HAMMING = [
  [0, 1.503214724408055, 0.135335283236613, 1.503214724408055, 0.503214724408055, 1.135335283236613],
  [1.503214724408055, 0, 1.503214724408055, 1.503214724408055, 1.367879441171442, 0.503214724408055],
  [0.135335283236613, 1.503214724408055, 0, 1.367879441171442, 0.503214724408055, 1.0],
  [1.503214724408055, 1.503214724408055, 1.367879441171442, 0, 1.135335283236613, 1.367879441171442],
  [0.503214724408055, 1.367879441171442, 0.503214724408055, 1.135335283236613, 0, 1.503214724408055],
  [1.135335283236613, 0.503214724408055, 1.0, 1.367879441171442, 1.503214724408055, 0],
];

// Helper to compare distance matrices
function expectDistMatrix(actual: { get(i: number, j: number): number; rows: number }, expected: number[][]) {
  for (let i = 0; i < expected.length; i++) {
    for (let j = 0; j < expected[i]!.length; j++) {
      expect(actual.get(i, j)).toBeCloseTo(expected[i]![j]!, 10);
    }
  }
}

// Canonicalize cluster labels: first-seen cluster = 1, second-seen = 2, etc.
// This makes partition comparison independent of arbitrary label numbering.
function canonicalize(assignments: number[]): number[] {
  const mapping = new Map<number, number>();
  let nextLabel = 1;
  return assignments.map((a) => {
    if (!mapping.has(a)) mapping.set(a, nextLabel++);
    return mapping.get(a)!;
  });
}

describe('clustering — R ground truth distance matrices', () => {
  for (const metric of Object.keys(R_DISTANCES) as (keyof typeof R_DISTANCES)[]) {
    it(`${metric} distance matrix matches R`, () => {
      const result = clusterSequences(clusterTestData, 2, { dissimilarity: metric as any });
      expectDistMatrix(result.distance, R_DISTANCES[metric]!);
    });
  }

  it('weighted hamming (lambda=1) distance matrix matches R', () => {
    const result = clusterSequences(clusterTestData, 2, {
      dissimilarity: 'hamming',
      weighted: true,
      lambda: 1,
    });
    expectDistMatrix(result.distance, R_WEIGHTED_HAMMING);
  });
});

describe('clustering — R ground truth PAM k=2', () => {
  const PAM_RESULTS: Record<string, { assignments: number[]; silhouette: number }> = {
    hamming: { assignments: [1, 2, 1, 1, 2, 1], silhouette: 0.241750841750842 },
    lv: { assignments: [1, 1, 1, 1, 2, 1], silhouette: 0.083333333333333 },
    osa: { assignments: [1, 1, 2, 2, 1, 2], silhouette: 0.241666666666667 },
    dl: { assignments: [1, 1, 2, 2, 1, 2], silhouette: 0.241666666666667 },
    lcs: { assignments: [1, 1, 1, 1, 1, 2], silhouette: 0.016666666666667 },
    qgram: { assignments: [1, 1, 2, 1, 1, 1], silhouette: 0.833333333333333 },
    cosine: { assignments: [1, 1, 2, 1, 1, 1], silhouette: 0.833333333333333 },
    jaccard: { assignments: [1, 1, 2, 1, 1, 1], silhouette: 0.833333333333333 },
    jw: { assignments: [1, 2, 1, 1, 2, 1], silhouette: 0.359643940289102 },
  };

  for (const [metric, expected] of Object.entries(PAM_RESULTS)) {
    it(`PAM k=2 ${metric}: partition and silhouette match R`, () => {
      const result = clusterSequences(clusterTestData, 2, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(canonicalize(expected.assignments));
      expect(result.silhouette).toBeCloseTo(expected.silhouette, 6);
    });
  }

  it('PAM k=2 weighted hamming (lambda=1) matches R', () => {
    const result = clusterSequences(clusterTestData, 2, {
      dissimilarity: 'hamming',
      weighted: true,
      lambda: 1,
    });
    expect(canonicalize(result.assignments)).toEqual(canonicalize([1, 2, 1, 1, 1, 2]));
    expect(result.silhouette).toBeCloseTo(0.458727121405005, 6);
  });
});

describe('clustering — R ground truth hierarchical k=2 (hamming)', () => {
  const HIER_RESULTS: Record<string, { assignments: number[]; silhouette: number }> = {
    complete: { assignments: [1, 1, 1, 2, 1, 1], silhouette: 0.085648148148148 },
    average: { assignments: [1, 2, 1, 1, 2, 1], silhouette: 0.241750841750842 },
    single: { assignments: [1, 1, 1, 1, 2, 1], silhouette: -0.007028619528620 },
    'ward.D': { assignments: [1, 2, 1, 2, 2, 1], silhouette: 0.250793650793651 },
    'ward.D2': { assignments: [1, 2, 1, 2, 2, 1], silhouette: 0.250793650793651 },
    mcquitty: { assignments: [1, 2, 1, 1, 2, 1], silhouette: 0.241750841750842 },
    median: { assignments: [1, 2, 1, 1, 1, 1], silhouette: 0.162037037037037 },
    centroid: { assignments: [1, 2, 1, 1, 1, 1], silhouette: 0.162037037037037 },
  };

  for (const [method, expected] of Object.entries(HIER_RESULTS)) {
    it(`hierarchical ${method}: partition and silhouette match R`, () => {
      const result = clusterSequences(clusterTestData, 2, {
        dissimilarity: 'hamming',
        method,
      });
      expect(canonicalize(result.assignments)).toEqual(canonicalize(expected.assignments));
      expect(result.silhouette).toBeCloseTo(expected.silhouette, 6);
    });
  }
});

describe('clustering — R ground truth PAM k=3 (hamming)', () => {
  it('PAM k=3 hamming matches R', () => {
    const result = clusterSequences(clusterTestData, 3, { dissimilarity: 'hamming' });
    expect(canonicalize(result.assignments)).toEqual(canonicalize([1, 2, 1, 3, 2, 1]));
    expect(result.silhouette).toBeCloseTo(0.233333333333333, 6);
  });
});

describe('clustering — clusterData (numeric)', () => {
  it('clusters well-separated numeric data', () => {
    const numData = [
      [0, 0], [0.1, 0.1], [0.2, 0],    // cluster 1
      [10, 10], [10.1, 10.2], [9.9, 10], // cluster 2
    ];
    const result = clusterData(numData, 2);
    expect(result.k).toBe(2);
    expect(result.assignments.length).toBe(6);
    // First 3 should be in same cluster, last 3 in another
    expect(result.assignments[0]).toBe(result.assignments[1]);
    expect(result.assignments[0]).toBe(result.assignments[2]);
    expect(result.assignments[3]).toBe(result.assignments[4]);
    expect(result.assignments[3]).toBe(result.assignments[5]);
    expect(result.assignments[0]).not.toBe(result.assignments[3]);
    expect(result.silhouette).toBeGreaterThan(0.9);
  });

  it('clusters with manhattan distance', () => {
    const numData = [
      [0, 0], [1, 0],
      [10, 10], [11, 10],
    ];
    const result = clusterData(numData, 2, { dissimilarity: 'manhattan' });
    expect(result.assignments[0]).toBe(result.assignments[1]);
    expect(result.assignments[2]).toBe(result.assignments[3]);
    expect(result.assignments[0]).not.toBe(result.assignments[2]);
  });

  it('clusters with hierarchical method', () => {
    const numData = [
      [0, 0], [0.5, 0],
      [10, 10], [10.5, 10],
    ];
    const result = clusterData(numData, 2, { method: 'complete' });
    expect(result.assignments[0]).toBe(result.assignments[1]);
    expect(result.assignments[2]).toBe(result.assignments[3]);
  });

  it('throws for k < 2', () => {
    expect(() => clusterData([[1], [2]], 1)).toThrow();
  });

  it('throws for k > n', () => {
    expect(() => clusterData([[1], [2]], 5)).toThrow();
  });
});

describe('compareSequences', () => {
  it('compares patterns across groups', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'High' : 'Low'));
    const gmodel = groupTna(smallData, groups);
    const result = compareSequences(gmodel, { sub: [1, 2], minFreq: 1 });

    expect(result.length).toBeGreaterThan(0);
    for (const row of result) {
      expect(typeof row.pattern).toBe('string');
      expect(row.frequencies['High']).toBeDefined();
      expect(row.frequencies['Low']).toBeDefined();
      expect(row.proportions['High']).toBeDefined();
      expect(row.proportions['Low']).toBeDefined();
    }
  });

  it('runs permutation test', () => {
    const groups = smallData.map((_, i) => (i < 10 ? 'High' : 'Low'));
    const gmodel = groupTna(smallData, groups);
    const result = compareSequences(gmodel, {
      sub: [1],
      minFreq: 1,
      test: true,
      iter: 100,
      seed: 42,
    });

    expect(result.length).toBeGreaterThan(0);
    for (const row of result) {
      expect(typeof row.effectSize).toBe('number');
      expect(typeof row.pValue).toBe('number');
      expect(row.pValue!).toBeGreaterThanOrEqual(0);
      expect(row.pValue!).toBeLessThanOrEqual(1);
    }
  });
});
