/**
 * Centrality stability estimation via case-dropping bootstrap.
 * Port of Desktop analysis/stability.ts.
 *
 * Also includes edge-level and network-level stability (Phase 5).
 */
import type { TNA, CentralityMeasure } from '../core/types.js';
import type { Matrix } from '../core/matrix.js';
import { computeTransitions3D, computeWeightsFrom3D } from '../core/transitions.js';
import { createTNA, buildModel as tnajBuildModel } from '../core/model.js';
import { pearsonCorr } from '../core/matrix.js';
import { SeededRNG } from '../core/rng.js';
import { centralities } from './centralities.js';
import { spearmanCorr } from '../stats/correlation.js';
import {
  toBinaryMatrix, applyWindowing, applyIntervalWindowing,
  computeWtnaTransitions, rowNormalizeWtna,
} from './wtna.js';
import type { WtnaOptions } from './wtna.js';

export interface StabilityResult {
  csCoefficients: Record<string, number>;
  meanCorrelations: Record<string, number[]>;
  dropProps: number[];
  threshold: number;
  certainty: number;
}

export interface StabilityOptions {
  measures?: CentralityMeasure[];
  iter?: number;
  dropProps?: number[];
  threshold?: number;
  certainty?: number;
  seed?: number;
  corrMethod?: 'pearson' | 'spearman';
}

/**
 * Estimate centrality stability using case-dropping bootstrap.
 */
export function estimateCS(
  model: TNA,
  options: StabilityOptions = {},
): StabilityResult {
  const {
    measures = ['InStrength', 'OutStrength', 'Betweenness'] as CentralityMeasure[],
    iter = 500,
    dropProps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    threshold = 0.7,
    certainty = 0.95,
    seed = 42,
    corrMethod = 'pearson',
  } = options;

  if (!model.data) {
    throw new Error('TNA model must have sequence data for centrality stability');
  }

  const labels = model.labels;
  const a = labels.length;
  const seqData = model.data;
  const n = seqData.length;
  const modelType = model.type;
  const modelScaling = model.scaling.length > 0 ? model.scaling : null;

  const rng = new SeededRNG(seed);

  // Pad sequences to uniform length
  let maxLen = 0;
  for (const seq of seqData) { if (seq.length > maxLen) maxLen = seq.length; }
  const padded = seqData.map(seq => {
    if (seq.length >= maxLen) return seq;
    const pad: (string | null)[] = new Array(maxLen - seq.length).fill(null);
    return [...seq, ...pad];
  });

  const trans = computeTransitions3D(padded, labels, modelType, model.params);
  const origCent = centralities(model, { measures });

  // Filter measures with non-zero variance
  const validMeasures: CentralityMeasure[] = [];
  for (const m of measures) {
    const vals = origCent.measures[m];
    if (!vals) continue;
    let mean = 0;
    for (let i = 0; i < a; i++) mean += vals[i]!;
    mean /= a;
    let variance = 0;
    for (let i = 0; i < a; i++) variance += (vals[i]! - mean) ** 2;
    if (variance > 0) validMeasures.push(m);
  }

  const correlations: Record<string, number[][]> = {};
  for (const m of validMeasures) {
    correlations[m] = dropProps.map(() => []);
  }

  for (let j = 0; j < dropProps.length; j++) {
    const dp = dropProps[j]!;
    const nDrop = Math.floor(n * dp);
    const nKeep = n - nDrop;
    if (nDrop === 0 || nKeep < 2) continue;

    for (let it = 0; it < iter; it++) {
      const keepIdx = rng.choiceWithoutReplacement(n, nKeep);
      const transSub: Matrix[] = [];
      for (const idx of keepIdx) {
        transSub.push(trans[idx]!);
      }

      const weightsSub = computeWeightsFrom3D(transSub, modelType, modelScaling);
      const subModel = createTNA(weightsSub, model.inits, labels, null, modelType, model.scaling);
      const subCent = centralities(subModel, { measures: validMeasures });

      for (const m of validMeasures) {
        const origVals = origCent.measures[m]!;
        const subVals = subCent.measures[m]!;
        const corr = corrMethod === 'spearman'
          ? spearmanCorr(origVals, subVals)
          : pearsonCorr(origVals, subVals);
        correlations[m]![j]!.push(corr);
      }
    }
  }

  return computeCsResult(measures, validMeasures, correlations, dropProps, threshold, certainty);
}

/** Input for WTNA centrality stability. */
export interface StabilityWtnaInput {
  records: Record<string, string | number>[];
  codes: string[];
  wtnaOpts: WtnaOptions;
  modelType: string;
  scaling: string | null | '';
}

/**
 * Centrality stability for WTNA models via case-dropping bootstrap.
 */
export function estimateCsWtna(
  input: StabilityWtnaInput,
  originalModel: TNA,
  options: StabilityOptions = {},
): StabilityResult {
  const {
    measures = ['InStrength', 'OutStrength', 'Betweenness'] as CentralityMeasure[],
    iter = 500,
    dropProps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    threshold = 0.7,
    certainty = 0.95,
    seed = 42,
    corrMethod = 'pearson',
  } = options;

  const { records, codes, wtnaOpts } = input;
  const labels = codes;
  const a = codes.length;
  const modelType = input.modelType;
  const scaling = input.scaling || null;

  // Build windowed groups
  const actorKey = wtnaOpts.actor;
  const sessionKey = wtnaOpts.session;
  const windowSize = Math.max(1, wtnaOpts.windowSize ?? 3);
  const windowType = wtnaOpts.windowType ?? 'tumbling';

  const groupMap = new Map<string, Record<string, string | number>[]>();
  for (const rec of records) {
    const actor = actorKey ? String(rec[actorKey] ?? '') : '__all__';
    if (!groupMap.has(actor)) groupMap.set(actor, []);
    groupMap.get(actor)!.push(rec);
  }

  const windowedGroups: number[][][] = [];
  for (const grp of groupMap.values()) {
    const X = toBinaryMatrix(grp, codes);
    let W: number[][];
    if (sessionKey) {
      const lbls = grp.map(rec => String(rec[sessionKey] ?? ''));
      W = applyIntervalWindowing(X, lbls);
    } else {
      W = applyWindowing(X, windowSize, windowType);
    }
    if (W.length >= 2) windowedGroups.push(W);
  }

  const n = windowedGroups.length;
  const rng = new SeededRNG(seed);

  const origCent = centralities(originalModel, { measures });

  const validMeasures: CentralityMeasure[] = [];
  for (const m of measures) {
    const vals = origCent.measures[m];
    if (!vals) continue;
    let mean = 0;
    for (let i = 0; i < a; i++) mean += vals[i]!;
    mean /= a;
    let variance = 0;
    for (let i = 0; i < a; i++) variance += (vals[i]! - mean) ** 2;
    if (variance > 0) validMeasures.push(m);
  }

  const correlations: Record<string, number[][]> = {};
  for (const m of validMeasures) {
    correlations[m] = dropProps.map(() => []);
  }

  for (let j = 0; j < dropProps.length; j++) {
    const dp = dropProps[j]!;
    const nDrop = Math.floor(n * dp);
    const nKeep = n - nDrop;
    if (nDrop === 0 || nKeep < 2) continue;

    for (let it = 0; it < iter; it++) {
      const keepIdx = rng.choiceWithoutReplacement(n, nKeep);

      const M: number[][] = Array.from({ length: a }, () => new Array<number>(a).fill(0));
      for (const idx of keepIdx) {
        const W = windowedGroups[idx]!;
        const T = computeWtnaTransitions(W);
        for (let i = 0; i < a; i++) {
          for (let jj = 0; jj < a; jj++) {
            M[i]![jj]! += T[i]?.[jj] ?? 0;
          }
        }
      }

      let matrix = M;
      if (modelType === 'tna') matrix = rowNormalizeWtna(matrix);

      const tempType = (modelType === 'tna' || !!scaling) ? 'relative' : 'frequency';
      const tempOpts: Record<string, unknown> = { type: tempType, labels };
      if (scaling) tempOpts.scaling = scaling;
      const subModel = tnajBuildModel(matrix, tempOpts as any) as TNA;
      const subCent = centralities(subModel, { measures: validMeasures });

      for (const m of validMeasures) {
        const origVals = origCent.measures[m]!;
        const subVals = subCent.measures[m]!;
        const corr = corrMethod === 'spearman'
          ? spearmanCorr(origVals, subVals)
          : pearsonCorr(origVals, subVals);
        correlations[m]![j]!.push(corr);
      }
    }
  }

  return computeCsResult(measures, validMeasures, correlations, dropProps, threshold, certainty);
}

/** Shared logic for computing CS coefficients from correlation data. */
function computeCsResult(
  measures: CentralityMeasure[],
  validMeasures: CentralityMeasure[],
  correlations: Record<string, number[][]>,
  dropProps: number[],
  threshold: number,
  certainty: number,
): StabilityResult {
  const meanCorrelations: Record<string, number[]> = {};
  const csCoefficients: Record<string, number> = {};

  for (const m of measures) {
    if (validMeasures.includes(m)) {
      const means: number[] = [];
      for (let j = 0; j < dropProps.length; j++) {
        const corrs = correlations[m]![j]!;
        const valid = corrs.filter(c => !isNaN(c));
        if (valid.length === 0) {
          means.push(NaN);
          continue;
        }
        means.push(valid.reduce((s, v) => s + v, 0) / valid.length);
      }
      meanCorrelations[m] = means;

      let cs = 0;
      for (let j = 0; j < dropProps.length; j++) {
        const corrs = correlations[m]![j]!;
        const validCorrs = corrs.filter(c => !isNaN(c));
        if (validCorrs.length === 0) continue;
        const aboveThreshold = validCorrs.filter(c => c >= threshold).length / validCorrs.length;
        if (aboveThreshold >= certainty) {
          cs = dropProps[j]!;
        }
      }
      csCoefficients[m] = cs;
    } else {
      meanCorrelations[m] = dropProps.map(() => NaN);
      csCoefficients[m] = 0;
    }
  }

  return { csCoefficients, meanCorrelations, dropProps, threshold, certainty };
}

// ── Phase 5: Edge-Level and Network-Level Stability ──────────────────────────

export interface EdgeStabilityResult {
  /** Correlation per (dropProp): mean correlation of flattened edge weight vectors. */
  meanCorrelations: number[];
  /** CS coefficient for edge stability. */
  csCoefficient: number;
  dropProps: number[];
  threshold: number;
  certainty: number;
}

/**
 * Estimate edge-level stability using case-dropping bootstrap.
 * Same approach as estimateCS, but correlates edge weight vectors
 * instead of centrality vectors.
 */
export function estimateEdgeStability(
  model: TNA,
  options: StabilityOptions = {},
): EdgeStabilityResult {
  const {
    iter = 500,
    dropProps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    threshold = 0.7,
    certainty = 0.95,
    seed = 42,
    corrMethod = 'pearson',
  } = options;

  if (!model.data) {
    throw new Error('TNA model must have sequence data for edge stability');
  }

  const labels = model.labels;
  const a = labels.length;
  const seqData = model.data;
  const n = seqData.length;
  const modelType = model.type;
  const modelScaling = model.scaling.length > 0 ? model.scaling : null;

  const rng = new SeededRNG(seed);

  let maxLen = 0;
  for (const seq of seqData) { if (seq.length > maxLen) maxLen = seq.length; }
  const padded = seqData.map(seq => {
    if (seq.length >= maxLen) return seq;
    const pad: (string | null)[] = new Array(maxLen - seq.length).fill(null);
    return [...seq, ...pad];
  });

  const trans = computeTransitions3D(padded, labels, modelType, model.params);

  // Original weight vector (flattened)
  const origWeightVec = new Float64Array(a * a);
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < a; j++) {
      origWeightVec[i * a + j] = model.weights.get(i, j);
    }
  }

  const corrsByDrop: number[][] = dropProps.map(() => []);

  for (let dp = 0; dp < dropProps.length; dp++) {
    const nDrop = Math.floor(n * dropProps[dp]!);
    const nKeep = n - nDrop;
    if (nDrop === 0 || nKeep < 2) continue;

    for (let it = 0; it < iter; it++) {
      const keepIdx = rng.choiceWithoutReplacement(n, nKeep);
      const transSub: Matrix[] = [];
      for (const idx of keepIdx) transSub.push(trans[idx]!);

      const weightsSub = computeWeightsFrom3D(transSub, modelType, modelScaling);

      const subWeightVec = new Float64Array(a * a);
      for (let i = 0; i < a; i++) {
        for (let j = 0; j < a; j++) {
          subWeightVec[i * a + j] = weightsSub.get(i, j);
        }
      }

      const corr = corrMethod === 'spearman'
        ? spearmanCorr(origWeightVec, subWeightVec)
        : pearsonCorr(origWeightVec, subWeightVec);
      corrsByDrop[dp]!.push(corr);
    }
  }

  const meanCorrelations: number[] = [];
  let csCoefficient = 0;

  for (let dp = 0; dp < dropProps.length; dp++) {
    const corrs = corrsByDrop[dp]!;
    const valid = corrs.filter(c => !isNaN(c));
    if (valid.length === 0) {
      meanCorrelations.push(NaN);
      continue;
    }
    meanCorrelations.push(valid.reduce((s, v) => s + v, 0) / valid.length);

    const aboveThreshold = valid.filter(c => c >= threshold).length / valid.length;
    if (aboveThreshold >= certainty) {
      csCoefficient = dropProps[dp]!;
    }
  }

  return { meanCorrelations, csCoefficient, dropProps, threshold, certainty };
}

export interface NetworkStabilityResult {
  /** Mean density correlation across drop proportions. */
  densityCorrelations: number[];
  /** Mean weight correlation across drop proportions. */
  meanWeightCorrelations: number[];
  /** CS coefficient for density stability. */
  densityCS: number;
  /** CS coefficient for mean weight stability. */
  meanWeightCS: number;
  dropProps: number[];
  threshold: number;
  certainty: number;
}

/**
 * Estimate network-level stability using case-dropping bootstrap.
 * Tracks global metrics (density, mean weight) across subsamples.
 */
export function estimateNetworkStability(
  model: TNA,
  options: StabilityOptions = {},
): NetworkStabilityResult {
  const {
    iter = 500,
    dropProps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    threshold = 0.7,
    certainty = 0.95,
    seed = 42,
  } = options;

  if (!model.data) {
    throw new Error('TNA model must have sequence data for network stability');
  }

  const labels = model.labels;
  const a = labels.length;
  const seqData = model.data;
  const n = seqData.length;
  const modelType = model.type;
  const modelScaling = model.scaling.length > 0 ? model.scaling : null;

  const rng = new SeededRNG(seed);

  let maxLen = 0;
  for (const seq of seqData) { if (seq.length > maxLen) maxLen = seq.length; }
  const padded = seqData.map(seq => {
    if (seq.length >= maxLen) return seq;
    const pad: (string | null)[] = new Array(maxLen - seq.length).fill(null);
    return [...seq, ...pad];
  });

  const trans = computeTransitions3D(padded, labels, modelType, model.params);

  // Original network metrics
  const origDensity = computeDensity(model.weights, a);
  const origMeanWeight = computeMeanWeight(model.weights, a);

  const densityDiffs: number[][] = dropProps.map(() => []);
  const meanWeightDiffs: number[][] = dropProps.map(() => []);

  for (let dp = 0; dp < dropProps.length; dp++) {
    const nDrop = Math.floor(n * dropProps[dp]!);
    const nKeep = n - nDrop;
    if (nDrop === 0 || nKeep < 2) continue;

    for (let it = 0; it < iter; it++) {
      const keepIdx = rng.choiceWithoutReplacement(n, nKeep);
      const transSub: Matrix[] = [];
      for (const idx of keepIdx) transSub.push(trans[idx]!);

      const weightsSub = computeWeightsFrom3D(transSub, modelType, modelScaling);

      const subDensity = computeDensity(weightsSub, a);
      const subMeanWeight = computeMeanWeight(weightsSub, a);

      // Store absolute difference (lower = more stable)
      densityDiffs[dp]!.push(Math.abs(subDensity - origDensity));
      meanWeightDiffs[dp]!.push(Math.abs(subMeanWeight - origMeanWeight));
    }
  }

  const densityCorrelations: number[] = [];
  const meanWeightCorrelations: number[] = [];
  let densityCS = 0;
  let meanWeightCS = 0;

  for (let dp = 0; dp < dropProps.length; dp++) {
    const dDiffs = densityDiffs[dp]!;
    const mDiffs = meanWeightDiffs[dp]!;

    if (dDiffs.length === 0) {
      densityCorrelations.push(NaN);
      meanWeightCorrelations.push(NaN);
      continue;
    }

    // Convert to stability score: 1 - normalized mean diff
    const dMean = dDiffs.reduce((s, v) => s + v, 0) / dDiffs.length;
    const mMean = mDiffs.reduce((s, v) => s + v, 0) / mDiffs.length;

    const dStability = origDensity > 0 ? 1 - dMean / origDensity : 1;
    const mStability = origMeanWeight > 0 ? 1 - mMean / origMeanWeight : 1;

    densityCorrelations.push(Math.max(0, dStability));
    meanWeightCorrelations.push(Math.max(0, mStability));

    if (Math.max(0, dStability) >= threshold) {
      const aboveThreshold = dDiffs.filter(d =>
        origDensity > 0 ? (1 - d / origDensity) >= threshold : true,
      ).length / dDiffs.length;
      if (aboveThreshold >= certainty) densityCS = dropProps[dp]!;
    }

    if (Math.max(0, mStability) >= threshold) {
      const aboveThreshold = mDiffs.filter(d =>
        origMeanWeight > 0 ? (1 - d / origMeanWeight) >= threshold : true,
      ).length / mDiffs.length;
      if (aboveThreshold >= certainty) meanWeightCS = dropProps[dp]!;
    }
  }

  return {
    densityCorrelations,
    meanWeightCorrelations,
    densityCS,
    meanWeightCS,
    dropProps,
    threshold,
    certainty,
  };
}

/** Compute network density (proportion of non-zero edges, excluding diagonal). */
function computeDensity(weights: { rows: number; get(i: number, j: number): number }, a: number): number {
  let nonZero = 0;
  const maxEdges = a * (a - 1);
  if (maxEdges === 0) return 0;
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < a; j++) {
      if (i !== j && weights.get(i, j) > 0) nonZero++;
    }
  }
  return nonZero / maxEdges;
}

/** Compute mean non-zero edge weight (excluding diagonal). */
function computeMeanWeight(weights: { rows: number; get(i: number, j: number): number }, a: number): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < a; j++) {
      if (i !== j) {
        const w = weights.get(i, j);
        if (w > 0) { sum += w; count++; }
      }
    }
  }
  return count > 0 ? sum / count : 0;
}
