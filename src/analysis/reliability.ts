/**
 * Reliability analysis: split-half comparison of TNA weight matrices.
 * Port of Desktop analysis/reliability.ts.
 *
 * All 22 metrics match R's implementation exactly. Key behaviour notes:
 *  - All vector-level metrics operate on the FULL n×n weight matrix
 *    (including diagonal), flattened column-major (same as R as.vector()).
 *  - Rank Agreement uses matrix row-differences, matching R's diff(matrix).
 *  - RV coefficient uses column-centred tcrossprod formula.
 *  - Distance correlation matches R's biased estimator (can be negative).
 */
import type { TNA, SequenceData } from '../core/types.js';
import { tna, ftna, ctna, atna } from '../core/model.js';
import { SeededRNG } from '../core/rng.js';
import { spearmanCorrArr, kendallTau, distanceCorr, rvCoefficient } from '../stats/correlation.js';

// ── Metric definitions ───────────────────────────────────────────────────────

export interface MetricDef {
  key: string;
  label: string;
  category: 'Deviations' | 'Correlations' | 'Dissimilarities' | 'Similarities' | 'Pattern';
}

export const RELIABILITY_METRICS: MetricDef[] = [
  { key: 'mad',        label: 'Mean Abs. Diff.',   category: 'Deviations' },
  { key: 'median_ad',  label: 'Median Abs. Diff.', category: 'Deviations' },
  { key: 'rmsd',       label: 'RMS Diff.',          category: 'Deviations' },
  { key: 'max_ad',     label: 'Max Abs. Diff.',     category: 'Deviations' },
  { key: 'rel_mad',    label: 'Rel. MAD',           category: 'Deviations' },
  { key: 'cv_ratio',   label: 'CV Ratio',           category: 'Deviations' },
  { key: 'pearson',    label: 'Pearson',            category: 'Correlations' },
  { key: 'spearman',   label: 'Spearman',           category: 'Correlations' },
  { key: 'kendall',    label: 'Kendall',            category: 'Correlations' },
  { key: 'dcor',       label: 'Distance Corr.',     category: 'Correlations' },
  { key: 'euclidean',  label: 'Euclidean',          category: 'Dissimilarities' },
  { key: 'manhattan',  label: 'Manhattan',          category: 'Dissimilarities' },
  { key: 'canberra',   label: 'Canberra',           category: 'Dissimilarities' },
  { key: 'braycurtis', label: 'Bray-Curtis',        category: 'Dissimilarities' },
  { key: 'frobenius',  label: 'Frobenius',          category: 'Dissimilarities' },
  { key: 'cosine',     label: 'Cosine',             category: 'Similarities' },
  { key: 'jaccard',    label: 'Jaccard',            category: 'Similarities' },
  { key: 'dice',       label: 'Dice',               category: 'Similarities' },
  { key: 'overlap',    label: 'Overlap',            category: 'Similarities' },
  { key: 'rv',         label: 'RV',                 category: 'Similarities' },
  { key: 'rank_agree', label: 'Rank Agreement',     category: 'Pattern' },
  { key: 'sign_agree', label: 'Sign Agreement',     category: 'Pattern' },
];

// ── Result types ─────────────────────────────────────────────────────────────

export interface ReliabilityMetricSummary {
  metric: string;
  category: string;
  mean: number;
  sd: number;
  median: number;
  min: number;
  max: number;
  q25: number;
  q75: number;
}

export interface ReliabilityResult {
  iterations: Record<string, number[]>;
  summary: ReliabilityMetricSummary[];
  iter: number;
  split: number;
  modelType: string;
}

// ── Internal math helpers ────────────────────────────────────────────────────

function arrMean(a: number[]): number {
  if (a.length === 0) return NaN;
  return a.reduce((s, v) => s + v, 0) / a.length;
}

function arrStd(a: number[], ddof = 1): number {
  if (a.length < ddof + 1) return NaN;
  const m = arrMean(a);
  const variance = a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - ddof);
  return Math.sqrt(variance);
}

function arrMedian(a: number[]): number {
  if (a.length === 0) return NaN;
  const sorted = [...a].sort((x, y) => x - y);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

function arrQuantile(a: number[], p: number): number {
  if (a.length === 0) return NaN;
  const sorted = [...a].sort((x, y) => x - y);
  const pos = p * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! * (hi - pos) + sorted[hi]! * (pos - lo);
}

function pearsonCorrArr(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return NaN;
  const mx = arrMean(x);
  const my = arrMean(y);
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i]! - mx;
    const dy = y[i]! - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom < 1e-14 ? NaN : num / denom;
}

// ── Matrix comparison ────────────────────────────────────────────────────────

/** Flatten n×n matrix in column-major order (same as R's as.vector). */
function flattenColMajor(w: TNA['weights']): number[] {
  const n = w.rows;
  const out: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      out.push(w.get(i, j));
    }
  }
  return out;
}

/**
 * Compare two TNA weight matrices using all 22 metrics.
 * Matches R's tna:::compare_ output exactly.
 */
export function compareWeightMatrices(a: TNA, b: TNA): Record<string, number> {
  const nanResult: Record<string, number> = {};
  for (const m of RELIABILITY_METRICS) nanResult[m.key] = NaN;

  if (a.labels.length !== b.labels.length) return nanResult;

  const n = a.labels.length;
  if (n === 0) return nanResult;

  const xv = flattenColMajor(a.weights);
  const yv = flattenColMajor(b.weights);
  const m2 = xv.length;

  const absX    = xv.map(v => Math.abs(v));
  const absY    = yv.map(v => Math.abs(v));
  const absDiff = xv.map((v, i) => Math.abs(v - yv[i]!));

  const meanX  = arrMean(xv);
  const meanY  = arrMean(yv);
  const stdX   = arrStd(xv);
  const stdY   = arrStd(yv);
  const meanAY = arrMean(absY);

  // Deviations
  const mad        = arrMean(absDiff);
  const median_ad  = arrMedian(absDiff);
  const rmsd       = Math.sqrt(arrMean(absDiff.map(d => d * d)));
  const max_ad     = Math.max(...absDiff);
  const rel_mad    = meanAY > 1e-14 ? mad / meanAY : NaN;
  const cv_ratio   = (Math.abs(meanX) > 1e-14 && Math.abs(stdY) > 1e-14)
    ? (stdX * meanY) / (meanX * stdY)
    : NaN;

  // Correlations
  const pearson  = pearsonCorrArr(xv, yv);
  const spearman = spearmanCorrArr(xv, yv);
  const kendall  = kendallTau(xv, yv);
  const dcor     = distanceCorr(xv, yv);

  // Dissimilarities
  const euclidean = Math.sqrt(absDiff.reduce((s, d) => s + d * d, 0));
  const manhattan = absDiff.reduce((s, d) => s + d, 0);

  let canberraSum = 0;
  for (let i = 0; i < m2; i++) {
    if (absX[i]! > 0 && absY[i]! > 0) {
      canberraSum += absDiff[i]! / (absX[i]! + absY[i]!);
    }
  }
  const canberra = canberraSum;

  const sumAbsXY = absX.reduce((s, v, i) => s + v + absY[i]!, 0);
  const braycurtis = sumAbsXY > 1e-14 ? manhattan / sumAbsXY : 0;

  const frobenius = Math.sqrt(n / 2) > 1e-14
    ? euclidean / Math.sqrt(n / 2)
    : NaN;

  // Similarities
  let dotXY = 0, dotXX = 0, dotYY = 0;
  for (let i = 0; i < m2; i++) {
    dotXY += xv[i]! * yv[i]!;
    dotXX += xv[i]! * xv[i]!;
    dotYY += yv[i]! * yv[i]!;
  }
  const cosine = Math.sqrt(dotXX * dotYY) > 1e-14
    ? dotXY / Math.sqrt(dotXX * dotYY)
    : NaN;

  let minSum = 0, maxSum = 0, sumAbsX = 0, sumAbsY = 0;
  for (let i = 0; i < m2; i++) {
    minSum  += Math.min(absX[i]!, absY[i]!);
    maxSum  += Math.max(absX[i]!, absY[i]!);
    sumAbsX += absX[i]!;
    sumAbsY += absY[i]!;
  }
  const jaccard = maxSum > 1e-14 ? minSum / maxSum : NaN;
  const dice    = (sumAbsX + sumAbsY) > 1e-14 ? 2 * minSum / (sumAbsX + sumAbsY) : NaN;
  const overlap = Math.min(sumAbsX, sumAbsY) > 1e-14
    ? minSum / Math.min(sumAbsX, sumAbsY)
    : NaN;

  const rv = rvCoefficient(a.weights, b.weights);

  // Pattern
  let matchCount = 0, totalDiff = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n; j++) {
      const dA = a.weights.get(i + 1, j) - a.weights.get(i, j);
      const dB = b.weights.get(i + 1, j) - b.weights.get(i, j);
      if (Math.sign(dA) === Math.sign(dB)) matchCount++;
      totalDiff++;
    }
  }
  const rank_agree = totalDiff > 0 ? matchCount / totalDiff : NaN;

  const sameSign  = xv.filter((v, i) => Math.sign(v) === Math.sign(yv[i]!)).length;
  const sign_agree = m2 > 0 ? sameSign / m2 : NaN;

  return {
    mad, median_ad, rmsd, max_ad, rel_mad, cv_ratio,
    pearson, spearman, kendall, dcor,
    euclidean, manhattan, canberra, braycurtis, frobenius,
    cosine, jaccard, dice, overlap, rv,
    rank_agree, sign_agree,
  };
}

// ── Builder map ──────────────────────────────────────────────────────────────

const BUILDERS: Record<string, (data: SequenceData, opts: Record<string, unknown>) => TNA> = {
  tna:  (d, o) => tna(d as any, o as any),
  ftna: (d, o) => ftna(d as any, o as any),
  ctna: (d, o) => ctna(d as any, o as any),
  atna: (d, o) => atna(d as any, o as any),
};

// ── Main function ────────────────────────────────────────────────────────────

/**
 * Perform split-half reliability analysis.
 *
 * Repeatedly splits the sequence data into two halves, builds a model on
 * each half, and compares the resulting weight matrices using 22 metrics
 * that exactly match R's tna:::compare_ output.
 */
export function reliabilityAnalysis(
  sequenceData: SequenceData,
  modelType: 'tna' | 'ftna' | 'ctna' | 'atna',
  opts: {
    iter?: number;
    split?: number;
    atnaBeta?: number;
    seed?: number;
    scaling?: string;
    addStartState?: boolean;
    startStateLabel?: string;
    addEndState?: boolean;
    endStateLabel?: string;
  } = {},
): ReliabilityResult {
  if (sequenceData.length < 4) {
    throw new Error('Need at least 4 sequences for reliability analysis');
  }

  const {
    iter = 100, split = 0.5, atnaBeta = 0.1, seed = 42,
    scaling, addStartState, startStateLabel, addEndState, endStateLabel,
  } = opts;
  const n  = sequenceData.length;
  const nA = Math.floor(n * split);

  if (nA < 2 || n - nA < 2) {
    throw new Error('Each split half must have at least 2 sequences');
  }

  const applyStartEnd = (seqs: SequenceData): SequenceData => {
    if (!addStartState && !addEndState) return seqs;
    return seqs.map(seq => {
      let last = seq.length - 1;
      while (last >= 0 && seq[last] === null) last--;
      const trimmed: (string | null)[] = seq.slice(0, last + 1);
      if (addStartState) trimmed.unshift(startStateLabel || 'Start');
      if (addEndState)   trimmed.push(endStateLabel   || 'End');
      return trimmed;
    });
  };

  const rng = new SeededRNG(seed);
  const builder = BUILDERS[modelType]!;
  const buildOpts: Record<string, unknown> = {};
  if (scaling)              buildOpts.scaling = scaling;
  if (modelType === 'atna') buildOpts.beta    = atnaBeta;

  const iterations: Record<string, number[]> = {};
  for (const m of RELIABILITY_METRICS) iterations[m.key] = [];

  for (let it = 0; it < iter; it++) {
    const indicesA = rng.choiceWithoutReplacement(n, nA);
    const setA = new Set(indicesA);
    const indicesB = Array.from({ length: n }, (_, i) => i).filter(i => !setA.has(i));

    const seqA: SequenceData = applyStartEnd(indicesA.map(i => sequenceData[i]!));
    const seqB: SequenceData = applyStartEnd(indicesB.map(i => sequenceData[i]!));

    try {
      const modelA = builder(seqA, buildOpts);
      const modelB = builder(seqB, buildOpts);
      const metrics = compareWeightMatrices(modelA, modelB);
      for (const m of RELIABILITY_METRICS) {
        iterations[m.key]!.push(metrics[m.key]!);
      }
    } catch {
      for (const m of RELIABILITY_METRICS) {
        iterations[m.key]!.push(NaN);
      }
    }
  }

  const summary: ReliabilityMetricSummary[] = RELIABILITY_METRICS.map(metDef => {
    const raw  = iterations[metDef.key] ?? [];
    const vals = raw.filter(v => isFinite(v));
    return {
      metric:   metDef.label,
      category: metDef.category,
      mean:     arrMean(vals),
      sd:       arrStd(vals),
      median:   arrMedian(vals),
      min:      vals.length > 0 ? Math.min(...vals) : NaN,
      max:      vals.length > 0 ? Math.max(...vals) : NaN,
      q25:      arrQuantile(vals, 0.25),
      q75:      arrQuantile(vals, 0.75),
    };
  });

  return { iterations, summary, iter, split, modelType };
}
