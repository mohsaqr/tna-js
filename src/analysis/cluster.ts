/**
 * Sequence clustering functions.
 * Port of R TNA clustering with full distance metric and linkage method support.
 */
import { Matrix } from '../core/matrix.js';
import type { SequenceData, ClusterResult, TNAData } from '../core/types.js';

const SENTINEL = '\0__NA__';

// ---- Distance functions ----

function toTokenLists(data: SequenceData, naSyms = ['*', '%']): string[][] {
  const naSet = new Set(naSyms);
  return data.map((row) =>
    row.map((val) => {
      if (val === null || val === undefined || val === '') return SENTINEL;
      if (naSet.has(val)) return SENTINEL;
      return val;
    }),
  );
}

/** Position of the last non-sentinel token (matching R's seq2chr 'len'). */
function effectiveLength(seq: string[]): number {
  let last = 0;
  for (let i = 0; i < seq.length; i++) {
    if (seq[i] !== SENTINEL) last = i + 1;
  }
  return last;
}

function hammingDistance(
  a: string[], b: string[],
  weighted = false, lambda_ = 1,
): number {
  const maxLen = Math.max(a.length, b.length);
  const aPad = [...a, ...new Array(maxLen - a.length).fill(SENTINEL)];
  const bPad = [...b, ...new Array(maxLen - b.length).fill(SENTINEL)];

  let dist = 0;
  for (let i = 0; i < maxLen; i++) {
    if (aPad[i] !== bPad[i]) {
      dist += weighted ? Math.exp(-lambda_ * i) : 1;
    }
  }
  return dist;
}

/**
 * Levenshtein edit distance.
 * Note: substitution cost is **inverted** (1 for match, 0 for mismatch)
 * to replicate R TNA's internal `levenshtein_dist` C function which uses
 * `cost = 0L + 1L * (x[i] == y[j])`.
 */
function levenshteinDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const m = lenA ?? a.length;
  const n = lenB ?? b.length;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array(n + 1).fill(0);

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      // R TNA inverted cost: match=1, mismatch=0
      const cost = a[i - 1] === b[j - 1] ? 1 : 0;
      curr[j] = Math.min(
        prev[j]! + 1,
        curr[j - 1]! + 1,
        prev[j - 1]! + cost,
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n]!;
}

/**
 * Optimal String Alignment distance.
 * Note: substitution/transposition cost is **inverted** (1 for match, 0 for
 * mismatch) to replicate R TNA's internal `osa_dist` C function.
 */
function osaDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const m = lenA ?? a.length;
  const n = lenB ?? b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  const d: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) d[i]![0] = i;
  for (let j = 0; j <= n; j++) d[0]![j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      // R TNA inverted cost: match=1, mismatch=0
      const cost = a[i - 1] === b[j - 1] ? 1 : 0;
      d[i]![j] = Math.min(
        d[i - 1]![j]! + 1,
        d[i]![j - 1]! + 1,
        d[i - 1]![j - 1]! + cost,
      );
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i]![j] = Math.min(d[i]![j]!, d[i - 2]![j - 2]! + cost);
      }
    }
  }
  return d[m]![n]!;
}

/** Full Damerau-Levenshtein with alphabet-indexed transposition tracking. */
function dlDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const m = lenA ?? a.length;
  const n = lenB ?? b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  const maxDist = m + n;
  // d is (m+2) x (n+2), with d[0] as the sentinel row
  const d: number[][] = Array.from({ length: m + 2 }, () => new Array(n + 2).fill(0));
  d[0]![0] = maxDist;
  for (let i = 0; i <= m; i++) {
    d[i + 1]![0] = maxDist;
    d[i + 1]![1] = i;
  }
  for (let j = 0; j <= n; j++) {
    d[0]![j + 1] = maxDist;
    d[1]![j + 1] = j;
  }

  // da: last row where each token was seen
  const da = new Map<string, number>();

  for (let i = 1; i <= m; i++) {
    let db = 0; // last column where a[i-1] matched b[j-1]
    for (let j = 1; j <= n; j++) {
      const i1 = da.get(b[j - 1]!) ?? 0;
      const j1 = db;
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      if (cost === 0) db = j;

      d[i + 1]![j + 1] = Math.min(
        d[i]![j]! + cost,        // substitution
        d[i + 1]![j]! + 1,       // insertion
        d[i]![j + 1]! + 1,       // deletion
        d[i1]![j1]! + (i - i1 - 1) + 1 + (j - j1 - 1), // transposition
      );
    }
    da.set(a[i - 1]!, i);
  }
  return d[m + 1]![n + 1]!;
}

/** LCS distance: max(m, n) - lcs_length (matches R TNA package). */
function lcsDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const m = lenA ?? a.length;
  const n = lenB ?? b.length;
  let prev = new Array(n + 1).fill(0);
  let curr = new Array(n + 1).fill(0);

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        curr[j] = prev[j - 1]! + 1;
      } else {
        curr[j] = Math.max(prev[j]!, curr[j - 1]!);
      }
    }
    [prev, curr] = [curr, new Array(n + 1).fill(0)];
  }
  return Math.max(m, n) - prev[n]!;
}

/** Build q-gram frequency profile (default q=1 = unigrams, matching R). */
function getQgrams(seq: string[], len?: number, q = 1): Map<string, number> {
  const n = len ?? seq.length;
  const profile = new Map<string, number>();
  for (let i = 0; i <= n - q; i++) {
    const gram = seq.slice(i, i + q).join('\0');
    profile.set(gram, (profile.get(gram) ?? 0) + 1);
  }
  return profile;
}

/** Q-gram distance: sum of |freq_a - freq_b| over all q-grams. */
function qgramDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const pa = getQgrams(a, lenA);
  const pb = getQgrams(b, lenB);
  const allKeys = new Set([...pa.keys(), ...pb.keys()]);
  let dist = 0;
  for (const key of allKeys) {
    dist += Math.abs((pa.get(key) ?? 0) - (pb.get(key) ?? 0));
  }
  return dist;
}

/** Cosine distance: 1 - cos(v_a, v_b) on q-gram frequency vectors. */
function cosineDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const pa = getQgrams(a, lenA);
  const pb = getQgrams(b, lenB);
  const allKeys = new Set([...pa.keys(), ...pb.keys()]);

  let dot = 0, normA = 0, normB = 0;
  for (const key of allKeys) {
    const va = pa.get(key) ?? 0;
    const vb = pb.get(key) ?? 0;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }

  if (normA === 0 || normB === 0) return 1;
  return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Jaccard distance: 1 - |intersection|/|union| on q-gram type sets. */
function jaccardDistance(a: string[], b: string[], lenA?: number, lenB?: number): number {
  const setA = new Set(getQgrams(a, lenA).keys());
  const setB = new Set(getQgrams(b, lenB).keys());

  let intersection = 0;
  for (const key of setA) {
    if (setB.has(key)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  if (union === 0) return 0;
  return 1 - intersection / union;
}

/** Jaro distance (p=0 matches R stringdist default for "jw" method). */
function jaroWinklerDistance(a: string[], b: string[], p = 0, lenA?: number, lenB?: number): number {
  const m = lenA ?? a.length;
  const n = lenB ?? b.length;
  if (m === 0 && n === 0) return 0;
  if (m === 0 || n === 0) return 1;

  const matchWindow = Math.max(0, Math.floor(Math.max(m, n) / 2) - 1);

  const aMatched = new Array(m).fill(false);
  const bMatched = new Array(n).fill(false);
  let matches = 0;

  // Find matches
  for (let i = 0; i < m; i++) {
    const lo = Math.max(0, i - matchWindow);
    const hi = Math.min(n - 1, i + matchWindow);
    for (let j = lo; j <= hi; j++) {
      if (!bMatched[j] && a[i] === b[j]) {
        aMatched[i] = true;
        bMatched[j] = true;
        matches++;
        break;
      }
    }
  }

  if (matches === 0) return 1;

  // Count transpositions
  let transpositions = 0;
  let bIdx = 0;
  for (let i = 0; i < m; i++) {
    if (!aMatched[i]) continue;
    while (!bMatched[bIdx]) bIdx++;
    if (a[i] !== b[bIdx]) transpositions++;
    bIdx++;
  }

  const jaroSim = (matches / m + matches / n + (matches - transpositions / 2) / matches) / 3;

  if (p === 0) return 1 - jaroSim;

  // Winkler prefix bonus (up to 4 characters)
  let prefix = 0;
  const maxPrefix = Math.min(4, Math.min(m, n));
  for (let i = 0; i < maxPrefix; i++) {
    if (a[i] === b[i]) prefix++;
    else break;
  }

  return 1 - (jaroSim + prefix * p * (1 - jaroSim));
}

type DistFunc = (a: string[], b: string[], lenA?: number, lenB?: number) => number;

const DISTANCE_FUNCS: Record<string, DistFunc> = {
  hamming: (a, b) => hammingDistance(a, b),
  lv: levenshteinDistance,
  osa: osaDistance,
  dl: dlDistance,
  lcs: lcsDistance,
  qgram: qgramDistance,
  cosine: cosineDistance,
  jaccard: jaccardDistance,
  jw: (a, b, lenA, lenB) => jaroWinklerDistance(a, b, 0, lenA, lenB),
};

function computeDistanceMatrix(
  sequences: string[][],
  dissimilarity: string,
  weighted = false,
  lambda_ = 1,
): Matrix {
  const n = sequences.length;
  const dist = Matrix.zeros(n, n);

  if (dissimilarity === 'hamming') {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = hammingDistance(sequences[i]!, sequences[j]!, weighted, lambda_);
        dist.set(i, j, d);
        dist.set(j, i, d);
      }
    }
  } else {
    const func = DISTANCE_FUNCS[dissimilarity];
    if (!func) throw new Error(`Unknown dissimilarity: ${dissimilarity}`);
    // Pre-compute effective lengths (last non-NA position, matching R's seq2chr)
    const effLens = sequences.map(effectiveLength);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = func(sequences[i]!, sequences[j]!, effLens[i], effLens[j]);
        dist.set(i, j, d);
        dist.set(j, i, d);
      }
    }
  }

  return dist;
}

// ---- Numeric distance functions (for clusterData) ----

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function manhattanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i]! - b[i]!);
  }
  return sum;
}

function computeNumericDistanceMatrix(data: number[][], metric: string): Matrix {
  const n = data.length;
  const dist = Matrix.zeros(n, n);
  const func = metric === 'manhattan' ? manhattanDistance : euclideanDistance;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = func(data[i]!, data[j]!);
      dist.set(i, j, d);
      dist.set(j, i, d);
    }
  }
  return dist;
}

// ---- Silhouette ----

function silhouetteScore(dist: Matrix, labels: number[]): number {
  const n = labels.length;
  const uniqueClusters = [...new Set(labels)];
  if (uniqueClusters.length < 2) return 0;

  let totalScore = 0;
  for (let i = 0; i < n; i++) {
    const ci = labels[i]!;

    // a(i): mean distance to same-cluster members
    let sumSame = 0;
    let countSame = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && labels[j] === ci) {
        sumSame += dist.get(i, j);
        countSame++;
      }
    }
    if (countSame === 0) continue;
    const ai = sumSame / countSame;

    // b(i): min over other clusters of mean distance
    let bi = Infinity;
    for (const c of uniqueClusters) {
      if (c === ci) continue;
      let sumOther = 0;
      let countOther = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === c) {
          sumOther += dist.get(i, j);
          countOther++;
        }
      }
      if (countOther > 0) {
        bi = Math.min(bi, sumOther / countOther);
      }
    }

    const maxAB = Math.max(ai, bi);
    totalScore += maxAB > 0 ? (bi - ai) / maxAB : 0;
  }

  return totalScore / n;
}

// ---- PAM (Partitioning Around Medoids) ----

function pam(dist: Matrix, k: number): number[] {
  const n = dist.rows;

  // BUILD phase: select medoids greedily
  const medoids: number[] = [];
  const totalDists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) totalDists[i] = totalDists[i]! + dist.get(i, j);
  }

  // First medoid: point with minimum total distance (last-wins tie-breaking, matches R)
  let bestIdx = 0;
  for (let i = 1; i < n; i++) {
    if (totalDists[i]! <= totalDists[bestIdx]!) bestIdx = i;
  }
  medoids.push(bestIdx);

  const nearestDist = new Float64Array(n);
  for (let i = 0; i < n; i++) nearestDist[i] = dist.get(i, medoids[0]!);

  // Subsequent medoids: maximize gain (last-wins tie-breaking, matches R)
  for (let m = 1; m < k; m++) {
    let bestGain = -Infinity;
    let bestCandidate = -1;
    for (let c = 0; c < n; c++) {
      if (medoids.includes(c)) continue;
      let gain = 0;
      for (let i = 0; i < n; i++) {
        gain += Math.max(0, nearestDist[i]! - dist.get(i, c));
      }
      if (gain >= bestGain) {
        bestGain = gain;
        bestCandidate = c;
      }
    }
    medoids.push(bestCandidate);
    for (let i = 0; i < n; i++) {
      nearestDist[i] = Math.min(nearestDist[i]!, dist.get(i, bestCandidate));
    }
  }

  // SWAP phase: evaluate all (medoid, non-medoid) pairs simultaneously
  const medoidsArr = [...medoids];
  for (let iter = 0; iter < 100; iter++) {
    let currentCost = 0;
    for (let i = 0; i < n; i++) {
      let minD = Infinity;
      for (const m of medoidsArr) minD = Math.min(minD, dist.get(i, m));
      currentCost += minD;
    }

    let bestChange = 0;
    let bestMIdx = -1;
    let bestSwap = -1;

    for (let mIdx = 0; mIdx < k; mIdx++) {
      for (let c = 0; c < n; c++) {
        if (medoidsArr.includes(c)) continue;
        const trial = [...medoidsArr];
        trial[mIdx] = c;
        let trialCost = 0;
        for (let i = 0; i < n; i++) {
          let minD = Infinity;
          for (const m of trial) minD = Math.min(minD, dist.get(i, m));
          trialCost += minD;
        }
        const change = trialCost - currentCost;
        if (change < bestChange) {
          bestChange = change;
          bestMIdx = mIdx;
          bestSwap = c;
        }
      }
    }

    if (bestMIdx >= 0) {
      medoidsArr[bestMIdx] = bestSwap;
    } else {
      break;
    }
  }

  // Sort medoids by index for consistent cluster labeling (matches R)
  const sortedMedoids = [...medoidsArr].sort((a, b) => a - b);

  // Assign (1-indexed): ties go to lower-index medoid (first in sorted order)
  return Array.from({ length: n }, (_, i) => {
    let minD = Infinity;
    let bestM = 0;
    for (let m = 0; m < k; m++) {
      const d = dist.get(i, sortedMedoids[m]!);
      if (d < minD) {
        minD = d;
        bestM = m;
      }
    }
    return bestM + 1;
  });
}

// ---- Hierarchical clustering (Lance-Williams) ----

/** Hierarchical agglomerative clustering with Lance-Williams formula. */
function hierarchical(dist: Matrix, k: number, method: string): number[] {
  const n = dist.rows;

  // Initialize: each point is its own cluster
  const clusters: number[][] = Array.from({ length: n }, (_, i) => [i]);
  const active = new Set<number>(Array.from({ length: n }, (_, i) => i));
  const sizes = new Array(n).fill(1) as number[];

  // Mutable distance matrix
  const d: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => dist.get(i, j)),
  );

  // For ward.D2, square distances first
  if (method === 'ward.D2') {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        d[i]![j]! *= d[i]![j]!;
      }
    }
  }

  while (active.size > k) {
    // Find closest pair
    let bestDist = Infinity;
    let bestI = -1;
    let bestJ = -1;

    const activeArr = [...active];
    for (let a = 0; a < activeArr.length; a++) {
      for (let b = a + 1; b < activeArr.length; b++) {
        const ci = activeArr[a]!;
        const cj = activeArr[b]!;
        if (d[ci]![cj]! < bestDist) {
          bestDist = d[ci]![cj]!;
          bestI = ci;
          bestJ = cj;
        }
      }
    }

    const ni = sizes[bestI]!;
    const nj = sizes[bestJ]!;
    const dij = d[bestI]![bestJ]!;

    // Update distances using Lance-Williams formula
    for (const ck of active) {
      if (ck === bestI || ck === bestJ) continue;
      const nk = sizes[ck]!;
      const dik = d[bestI]![ck]!;
      const djk = d[bestJ]![ck]!;

      let newDist: number;
      switch (method) {
        case 'single':
          newDist = Math.min(dik, djk);
          break;
        case 'complete':
          newDist = Math.max(dik, djk);
          break;
        case 'mcquitty':
          newDist = (dik + djk) / 2;
          break;
        case 'median':
          newDist = (dik + djk) / 2 - dij / 4;
          break;
        case 'centroid':
          newDist = (ni * dik + nj * djk) / (ni + nj)
            - (ni * nj * dij) / ((ni + nj) * (ni + nj));
          break;
        case 'ward.D':
        case 'ward.D2':
          newDist = ((ni + nk) * dik + (nj + nk) * djk - nk * dij) / (ni + nj + nk);
          break;
        default: // average (UPGMA)
          newDist = (ni * dik + nj * djk) / (ni + nj);
          break;
      }

      d[bestI]![ck] = newDist;
      d[ck]![bestI] = newDist;
    }

    // Merge bestJ into bestI
    clusters[bestI] = [...clusters[bestI]!, ...clusters[bestJ]!];
    sizes[bestI] = ni + nj;
    active.delete(bestJ);
  }

  // Assign cluster labels (1-indexed)
  const assignments = new Array(n).fill(0) as number[];
  let clusterIdx = 1;
  for (const ci of active) {
    for (const point of clusters[ci]!) {
      assignments[point] = clusterIdx;
    }
    clusterIdx++;
  }

  return assignments;
}

// ---- Type detection ----

function isNumericData(data: unknown): data is number[][] {
  if (!Array.isArray(data) || data.length === 0) return false;
  const firstRow = data[0];
  return Array.isArray(firstRow) && firstRow.length > 0 && typeof firstRow[0] === 'number';
}

// ---- Main functions ----

/**
 * Unified clustering function. Auto-detects input type:
 * - TNAData (has .sequenceData) → string distance metrics
 * - number[][] → numeric distance metrics (euclidean/manhattan)
 * - SequenceData (string[][]) → string distance metrics
 */
export function clusterData(
  data: SequenceData | TNAData | number[][],
  k: number,
  options?: {
    dissimilarity?: string;
    method?: string;
    naSyms?: string[];
    weighted?: boolean;
    lambda?: number;
  },
): ClusterResult {
  const method = options?.method ?? 'pam';

  // Path 1: TNAData — extract sequenceData, use string metrics
  if (typeof data === 'object' && data !== null && 'sequenceData' in data) {
    const seqData = (data as TNAData).sequenceData;
    return clusterStringData(seqData, k, method, options);
  }

  // Path 2: numeric data
  if (isNumericData(data)) {
    const dissimilarity = options?.dissimilarity ?? 'euclidean';

    if (k < 2) throw new Error('k must be >= 2');
    if (k > data.length) throw new Error(`k=${k} exceeds number of observations (${data.length})`);

    const dist = computeNumericDistanceMatrix(data, dissimilarity);

    let assignments: number[];
    if (method === 'pam') {
      assignments = pam(dist, k);
    } else {
      assignments = hierarchical(dist, k, method);
    }

    const sil = silhouetteScore(dist, assignments);

    const sizes: number[] = [];
    for (let c = 1; c <= k; c++) {
      sizes.push(assignments.filter((a) => a === c).length);
    }

    // Convert numeric data to string form for ClusterResult compatibility
    const seqData: SequenceData = data.map((row) => row.map((v) => String(v)));

    return {
      data: seqData,
      k,
      assignments,
      silhouette: sil,
      sizes,
      method,
      distance: dist,
      dissimilarity,
    };
  }

  // Path 3: SequenceData (string[][])
  return clusterStringData(data as SequenceData, k, method, options);
}

/** Internal: cluster string sequence data with string distance metrics. */
function clusterStringData(
  seqData: SequenceData,
  k: number,
  method: string,
  options?: {
    dissimilarity?: string;
    naSyms?: string[];
    weighted?: boolean;
    lambda?: number;
  },
): ClusterResult {
  const dissimilarity = options?.dissimilarity ?? 'hamming';
  const naSyms = options?.naSyms ?? ['*', '%'];
  const weighted = options?.weighted ?? false;
  const lambda_ = options?.lambda ?? 1;

  if (k < 2) throw new Error('k must be >= 2');
  if (k > seqData.length) throw new Error(`k=${k} exceeds number of sequences (${seqData.length})`);

  const sequences = toTokenLists(seqData, naSyms);
  const dist = computeDistanceMatrix(sequences, dissimilarity, weighted, lambda_);

  let assignments: number[];
  if (method === 'pam') {
    assignments = pam(dist, k);
  } else {
    assignments = hierarchical(dist, k, method);
  }

  const sil = silhouetteScore(dist, assignments);

  const sizes: number[] = [];
  for (let c = 1; c <= k; c++) {
    sizes.push(assignments.filter((a) => a === c).length);
  }

  return {
    data: seqData,
    k,
    assignments,
    silhouette: sil,
    sizes,
    method,
    distance: dist,
    dissimilarity,
  };
}

/** @deprecated Use clusterData() instead. */
export function clusterSequences(
  data: SequenceData | TNAData,
  k: number,
  options?: {
    dissimilarity?: 'hamming' | 'lv' | 'osa' | 'dl' | 'lcs' | 'qgram' | 'cosine' | 'jaccard' | 'jw';
    method?: string;
    naSyms?: string[];
    weighted?: boolean;
    lambda?: number;
  },
): ClusterResult {
  return clusterData(data, k, options);
}
