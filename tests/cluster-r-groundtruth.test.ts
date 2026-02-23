/**
 * 100 R Ground Truth Clustering Tests
 *
 * All expected values generated from R using:
 *   - stringdist::stringdistmatrix() for sequence distances
 *   - stats::dist() for numeric distances
 *   - cluster::pam() for PAM clustering
 *   - stats::hclust() + stats::cutree() for hierarchical clustering
 *   - cluster::silhouette() for silhouette scores
 *
 * Assignments are canonicalized (first-seen label = 1, second = 2, etc.)
 * so partition comparison is independent of arbitrary label numbering.
 */

import { describe, it, expect } from 'vitest';
import { clusterSequences, clusterData } from '../src/analysis/cluster.js';
import type { SequenceData } from '../src/core/types.js';

// ============================================================
// Canonicalize helper
// ============================================================

function canonicalize(assignments: number[]): number[] {
  const mapping = new Map<number, number>();
  let nextLabel = 1;
  return assignments.map((a) => {
    if (!mapping.has(a)) mapping.set(a, nextLabel++);
    return mapping.get(a)!;
  });
}

// ============================================================
// SEQUENCE DATASETS
// ============================================================

const DS_A: SequenceData = [
  ['A', 'B', 'C', 'D'],
  ['D', 'C', 'B', 'A'],
  ['A', 'B', 'A', 'B'],
  ['C', 'D', 'C', 'D'],
  ['B', 'A', 'D', 'C'],
  ['A', 'C', 'B', 'D'],
  ['D', 'B', 'C', 'A'],
  ['C', 'A', 'D', 'B'],
];

const DS_B: SequenceData = [
  ['X', 'Y', 'Z'],
  ['Y', 'Z', 'X'],
  ['Z', 'X', 'Y'],
  ['X', 'X', 'Y'],
  ['Y', 'Y', 'Z'],
  ['Z', 'Z', 'X'],
  ['X', 'Y', 'X'],
  ['Y', 'Z', 'Y'],
  ['Z', 'X', 'Z'],
  ['X', 'Z', 'Y'],
];

const DS_C: SequenceData = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'B', 'C'],
  ['C', 'A', 'B', 'C', 'A'],
  ['A', 'A', 'B', 'B', 'C'],
  ['B', 'B', 'C', 'C', 'A'],
  ['C', 'C', 'A', 'A', 'B'],
  ['A', 'B', 'A', 'C', 'B'],
  ['B', 'C', 'B', 'A', 'C'],
  ['C', 'A', 'C', 'B', 'A'],
  ['A', 'A', 'C', 'B', 'B'],
  ['B', 'B', 'C', 'A', 'A'],
  ['C', 'C', 'A', 'B', 'B'],
];

const DS_D: SequenceData = [
  ['P', 'Q', 'R', 'S'],
  ['S', 'R', 'Q', 'P'],
  ['P', 'Q', 'P', 'Q'],
  ['R', 'S', 'R', 'S'],
  ['Q', 'P', 'S', 'R'],
  ['S', 'P', 'R', 'Q'],
];

// ============================================================
// NUMERIC DATASETS
// ============================================================

const OH_A: number[][] = [
  [1, 0, 0, 1],
  [0, 1, 1, 0],
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 1, 0, 0],
  [0, 0, 1, 1],
  [1, 0, 0, 0],
  [0, 1, 1, 1],
];

const OH_B: number[][] = [
  [1, 2, 0, 0, 1, 0],
  [0, 1, 2, 1, 0, 0],
  [2, 0, 1, 0, 0, 1],
  [1, 1, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 1],
  [2, 2, 0, 0, 0, 0],
  [0, 0, 2, 2, 0, 0],
  [0, 0, 0, 0, 2, 2],
  [1, 0, 1, 0, 1, 0],
  [0, 1, 0, 1, 0, 1],
];

const OH_C: number[][] = [
  [0.1, 0.2, 0.1],
  [0.2, 0.1, 0.3],
  [0.0, 0.3, 0.2],
  [0.3, 0.1, 0.1],
  [0.1, 0.2, 0.2],
  [4.9, 5.1, 5.0],
  [5.0, 4.8, 5.2],
  [5.1, 5.0, 4.9],
  [4.8, 5.2, 5.1],
  [5.2, 4.9, 5.0],
  [9.8, 10.1, 10.0],
  [10.0, 9.9, 10.2],
  [10.1, 10.0, 9.8],
  [9.9, 10.2, 10.1],
  [10.2, 9.8, 10.0],
];

// ============================================================
// DS_A TESTS (30)
// ============================================================

describe('DS_A (8 seqs, len 4) — PAM k=2, all metrics', () => {
  const cases: [string, number[], number][] = [
    ['hamming', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['lv', [1, 1, 1, 1, 1, 1, 1, 2], -0.2916666666666666],
    ['osa', [1, 1, 1, 1, 1, 1, 1, 2], -0.2916666666666666],
    ['dl', [1, 2, 1, 1, 1, 1, 2, 1], 0.2609523809523809],
    ['lcs', [1, 2, 1, 1, 1, 1, 2, 1], 0.162948717948718],
    ['qgram', [1, 1, 1, 2, 1, 1, 1, 1], 0.6875],
    ['cosine', [1, 1, 1, 2, 1, 1, 1, 1], 0.7133883476483184],
    ['jaccard', [1, 1, 1, 2, 1, 1, 1, 1], 0.6875],
    ['jw', [1, 2, 1, 1, 1, 1, 2, 1], 0.383298319327731],
  ];

  for (const [metric, expectedAssign, expectedSil] of cases) {
    it(`PAM k=2 ${metric}`, () => {
      const result = clusterSequences(DS_A, 2, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('DS_A (8 seqs, len 4) — PAM k=3, all metrics', () => {
  const cases: [string, number[], number][] = [
    ['hamming', [1, 2, 1, 1, 3, 1, 2, 3], 0.3025793650793651],
    ['lv', [1, 1, 1, 1, 1, 1, 1, 2], -0.2916666666666666],
    ['osa', [1, 1, 1, 1, 1, 1, 1, 2], -0.2916666666666666],
    ['dl', [1, 2, 1, 1, 1, 1, 2, 3], 0.1420995670995671],
    ['lcs', [1, 2, 1, 1, 1, 1, 2, 3], 0.10625],
    ['qgram', [1, 1, 2, 3, 1, 1, 1, 1], 0.75],
    ['cosine', [1, 1, 2, 3, 1, 1, 1, 1], 0.75],
    ['jaccard', [1, 1, 2, 3, 1, 1, 1, 1], 0.75],
    ['jw', [1, 2, 1, 3, 3, 1, 2, 1], 0.2888888888888889],
  ];

  for (const [metric, expectedAssign, expectedSil] of cases) {
    it(`PAM k=3 ${metric}`, () => {
      const result = clusterSequences(DS_A, 3, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('DS_A (8 seqs, len 4) — hierarchical hamming k=2, all methods', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['single', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['average', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['ward.D', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['ward.D2', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['mcquitty', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['median', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
    ['centroid', [1, 1, 1, 1, 2, 1, 1, 2], 0.2907467532467533],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterSequences(DS_A, 2, { dissimilarity: 'hamming', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('DS_A (8 seqs, len 4) — hierarchical lv k=3', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 1, 2, 2, 1, 3, 3, 3], 0.75],
    ['ward.D', [1, 1, 2, 2, 1, 3, 3, 3], 0.75],
    ['ward.D2', [1, 1, 2, 2, 1, 3, 3, 3], 0.75],
    ['average', [1, 1, 2, 2, 1, 3, 3, 3], 0.75],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterSequences(DS_A, 3, { dissimilarity: 'lv', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

// ============================================================
// DS_B TESTS (18)
// ============================================================

describe('DS_B (10 seqs, len 3) — PAM k=2, all metrics', () => {
  const cases: [string, number[], number][] = [
    ['hamming', [1, 2, 2, 2, 1, 2, 1, 2, 1, 2], 0.2507150072150072],
    ['lv', [1, 2, 1, 1, 1, 1, 2, 2, 2, 2], 0.2441666666666667],
    ['osa', [1, 2, 1, 1, 1, 1, 2, 2, 2, 2], 0.2441666666666667],
    ['dl', [1, 2, 1, 1, 1, 2, 2, 1, 2, 1], 0.1591269841269841],
    ['lcs', [1, 1, 1, 1, 2, 1, 1, 2, 1, 1], 0.210515873015873],
    ['qgram', [1, 1, 1, 1, 1, 2, 1, 1, 2, 1], 0.5428571428571429],
    ['cosine', [1, 1, 1, 1, 1, 2, 1, 1, 2, 1], 0.5712749216158032],
    ['jaccard', [1, 1, 1, 1, 1, 2, 1, 1, 2, 1], 0.5428571428571429],
    ['jw', [1, 2, 2, 2, 1, 2, 1, 2, 1, 2], 0.334268414681946],
  ];

  for (const [metric, expectedAssign, expectedSil] of cases) {
    it(`PAM k=2 ${metric}`, () => {
      const result = clusterSequences(DS_B, 2, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('DS_B (10 seqs, len 3) — PAM k=4, selected metrics', () => {
  const cases: [string, number[], number][] = [
    ['hamming', [1, 2, 3, 4, 1, 2, 1, 4, 3, 4], 0.3692857142857143],
    ['lv', [1, 2, 3, 1, 1, 1, 3, 3, 3, 2], 0.07222222222222223],
    ['osa', [1, 2, 3, 1, 1, 1, 3, 3, 3, 2], 0.07222222222222223],
    ['lcs', [1, 1, 1, 1, 2, 1, 1, 2, 3, 4], -0.07607142857142855],
    ['jw', [1, 2, 3, 4, 1, 2, 1, 4, 3, 4], 0.408257918552036],
  ];

  for (const [metric, expectedAssign, expectedSil] of cases) {
    it(`PAM k=4 ${metric}`, () => {
      const result = clusterSequences(DS_B, 4, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('DS_B (10 seqs, len 3) — hierarchical osa k=2', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 1, 1, 2, 2, 2, 2, 2, 2, 1], 0.424],
    ['single', [1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 0.05],
    ['average', [1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 0.05],
    ['ward.D2', [1, 1, 1, 2, 2, 2, 2, 2, 2, 1], 0.424],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterSequences(DS_B, 2, { dissimilarity: 'osa', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

// ============================================================
// DS_C TESTS (13)
// ============================================================

describe('DS_C (12 seqs, len 5) — PAM k=3, all metrics', () => {
  const cases: [string, number[], number][] = [
    ['hamming', [1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2], 0.2616221741221741],
    ['lv', [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 2, 1], 0.4921957671957672],
    ['osa', [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 2, 1], 0.4777281746031747],
    ['dl', [1, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3], 0.253167492480573],
    ['lcs', [1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 2, 3], 0.2217631497043262],
    ['qgram', [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2], 1],
    ['cosine', [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2], 1],
    ['jaccard', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3], 0],
    ['jw', [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 0.3865623670226488],
  ];

  for (const [metric, expectedAssign, expectedSil] of cases) {
    it(`PAM k=3 ${metric}`, () => {
      const result = clusterSequences(DS_C, 3, { dissimilarity: metric as any });
      // jaccard: all DS_C sequences share alphabet {A,B,C} so all distances are 0;
      // partition is arbitrary, only check silhouette for this degenerate case
      if (metric !== 'jaccard') {
        expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      }
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('DS_C (12 seqs, len 5) — hierarchical hamming k=3', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 0.1971153846153846],
    ['average', [1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2], 0.3205128205128205],
    ['ward.D', [1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2], 0.3205128205128205],
    ['mcquitty', [1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2], 0.3205128205128205],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterSequences(DS_C, 3, { dissimilarity: 'hamming', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

// ============================================================
// DS_D TESTS (9)
// ============================================================

describe('DS_D (6 seqs, len 4) — PAM k=2, all metrics', () => {
  const cases: [string, number[], number][] = [
    ['hamming', [1, 2, 1, 1, 2, 2], 0.178030303030303],
    ['lv', [1, 1, 1, 1, 1, 2], 0.5],
    ['osa', [1, 1, 1, 1, 1, 2], 0.5],
    ['dl', [1, 2, 1, 1, 1, 2], 0.2148962148962149],
    ['lcs', [1, 2, 1, 1, 1, 2], 0.1259259259259259],
    ['qgram', [1, 1, 1, 2, 1, 1], 0.5833333333333334],
    ['cosine', [1, 1, 1, 2, 1, 1], 0.6178511301977579],
    ['jaccard', [1, 1, 1, 2, 1, 1], 0.5833333333333334],
    ['jw', [1, 2, 2, 1, 1, 2], 0.4553571428571427],
  ];

  for (const [metric, expectedAssign, expectedSil] of cases) {
    it(`PAM k=2 ${metric}`, () => {
      const result = clusterSequences(DS_D, 2, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

// ============================================================
// OH_A TESTS (12)
// ============================================================

describe('OH_A (8x4 binary) — PAM', () => {
  const cases: [string, number, number[], number][] = [
    ['euclidean', 2, [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['manhattan', 2, [1, 2, 1, 2, 1, 2, 1, 2], 0.4685314685314685],
    ['euclidean', 3, [1, 2, 1, 2, 1, 3, 1, 2], 0.1339157161471588],
    ['manhattan', 3, [1, 2, 1, 2, 1, 3, 1, 2], 0.2232142857142857],
  ];

  for (const [metric, k, expectedAssign, expectedSil] of cases) {
    it(`PAM k=${k} ${metric}`, () => {
      const result = clusterData(OH_A, k, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('OH_A (8x4 binary) — hierarchical euclidean k=2', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['single', [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['average', [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['ward.D', [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['ward.D2', [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['mcquitty', [1, 2, 1, 2, 1, 2, 1, 2], 0.2774776708579856],
    ['median', [1, 1, 1, 1, 1, 2, 1, 1], -0.009902025392744429],
    ['centroid', [1, 1, 1, 1, 1, 2, 1, 1], -0.009902025392744429],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterData(OH_A, 2, { dissimilarity: 'euclidean', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

// ============================================================
// OH_B TESTS (10)
// ============================================================

describe('OH_B (10x6 numeric) — PAM', () => {
  const cases: [string, number, number[], number][] = [
    ['euclidean', 2, [1, 1, 1, 1, 2, 1, 1, 2, 1, 2], 0.1967878117570472],
    ['euclidean', 3, [1, 2, 1, 1, 3, 1, 2, 3, 1, 3], 0.3103448335017324],
    ['euclidean', 4, [1, 2, 3, 3, 4, 1, 2, 4, 3, 4], 0.3155376933724527],
    ['manhattan', 2, [1, 1, 1, 1, 2, 1, 1, 2, 1, 2], 0.2606334810746575],
    ['manhattan', 3, [1, 2, 1, 1, 3, 1, 2, 3, 1, 3], 0.4013301282051282],
    ['manhattan', 4, [1, 2, 3, 3, 4, 1, 2, 4, 3, 4], 0.4096428571428571],
  ];

  for (const [metric, k, expectedAssign, expectedSil] of cases) {
    it(`PAM k=${k} ${metric}`, () => {
      const result = clusterData(OH_B, k, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('OH_B (10x6 numeric) — hierarchical euclidean k=3', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 2, 1, 1, 2, 1, 2, 3, 1, 2], 0.165910165935545],
    ['average', [1, 2, 1, 1, 3, 1, 2, 3, 1, 3], 0.3103448335017324],
    ['ward.D', [1, 2, 1, 1, 3, 1, 2, 3, 1, 3], 0.3103448335017324],
    ['ward.D2', [1, 2, 1, 1, 3, 1, 2, 3, 1, 3], 0.3103448335017324],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterData(OH_B, 3, { dissimilarity: 'euclidean', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

// ============================================================
// OH_C TESTS (8)
// ============================================================

describe('OH_C (15x3 point cloud) — PAM', () => {
  const cases: [string, number, number[], number][] = [
    ['euclidean', 2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 0.7118118434053579],
    ['euclidean', 3, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], 0.9637102990152355],
    ['manhattan', 2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 0.7133003275053851],
    ['manhattan', 3, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], 0.9669167786452151],
  ];

  for (const [metric, k, expectedAssign, expectedSil] of cases) {
    it(`PAM k=${k} ${metric}`, () => {
      const result = clusterData(OH_C, k, { dissimilarity: metric as any });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});

describe('OH_C (15x3 point cloud) — hierarchical manhattan k=2', () => {
  const cases: [string, number[], number][] = [
    ['complete', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 0.7133003275053851],
    ['single', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 0.7133003275053851],
    ['average', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 0.7133003275053851],
    ['ward.D2', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 0.7133003275053851],
  ];

  for (const [method, expectedAssign, expectedSil] of cases) {
    it(`hclust ${method}`, () => {
      const result = clusterData(OH_C, 2, { dissimilarity: 'manhattan', method });
      expect(canonicalize(result.assignments)).toEqual(expectedAssign);
      expect(result.silhouette).toBeCloseTo(expectedSil, 6);
    });
  }
});
