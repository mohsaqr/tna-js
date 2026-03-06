/**
 * Permutation test for comparing two TNA models.
 * Port of Desktop analysis/permutation.ts.
 * Uses tnaj's computeTransitions3D / computeWeightsFrom3D for exact R equivalence.
 */
import type { TNA } from '../core/types.js';
import type { Matrix } from '../core/matrix.js';
import { computeTransitions3D, computeWeightsFrom3D } from '../core/transitions.js';
import { buildModel as tnajBuildModel } from '../core/model.js';
import { SeededRNG } from '../core/rng.js';
import { pAdjust } from '../stats/pAdjust.js';
import type { PAdjustMethod } from '../stats/pAdjust.js';
import {
  toBinaryMatrix, applyWindowing, applyIntervalWindowing,
  computeWtnaTransitions, rowNormalizeWtna,
} from './wtna.js';
import type { WtnaOptions } from './wtna.js';

export interface EdgeStat {
  from: string;
  to: string;
  diffTrue: number;
  effectSize: number;
  pValue: number;
}

export interface PermutationResult {
  edgeStats: EdgeStat[];
  diffTrue: Float64Array;
  diffSig: Float64Array;
  pValues: Float64Array;
  labels: string[];
  nStates: number;
  level: number;
}

export interface PermutationOptions {
  iter?: number;
  adjust?: PAdjustMethod;
  level?: number;
  seed?: number;
  paired?: boolean;
}

/**
 * Permutation test comparing two TNA models.
 * Both models must have sequence data (model.data) and identical labels.
 */
export function permutationTest(
  x: TNA,
  y: TNA,
  options: PermutationOptions = {},
): PermutationResult {
  const { iter = 1000, adjust = 'none', level = 0.05, seed = 42, paired = false } = options;

  if (!x.data || !y.data) {
    throw new Error('Both TNA models must have sequence data for permutation test');
  }

  const labels = x.labels;
  const a = labels.length;

  if (a !== y.labels.length || !labels.every((l, i) => l === y.labels[i])) {
    throw new Error('Both models must have the same state labels');
  }

  const modelType = x.type;
  const modelScaling = x.scaling.length > 0 ? x.scaling : null;

  const dataX = x.data;
  const dataY = y.data;
  const nX = dataX.length;
  const nY = dataY.length;
  const combined = [...dataX, ...dataY];
  const nXY = nX + nY;

  // Pad sequences to uniform length
  let maxLen = 0;
  for (const seq of combined) {
    if (seq.length > maxLen) maxLen = seq.length;
  }
  const padded = combined.map(seq => {
    if (seq.length >= maxLen) return seq;
    const pad: (string | null)[] = new Array(maxLen - seq.length).fill(null);
    return [...seq, ...pad];
  });

  const combinedTrans = computeTransitions3D(padded, labels, modelType, x.params);

  // True differences from model weights
  const trueDiff = new Float64Array(a * a);
  const absTrueDiff = new Float64Array(a * a);
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < a; j++) {
      const idx = i * a + j;
      trueDiff[idx] = x.weights.get(i, j) - y.weights.get(i, j);
      absTrueDiff[idx] = Math.abs(trueDiff[idx]!);
    }
  }

  if (paired && nX !== nY) {
    throw new Error('Paired permutation test requires equal group sizes');
  }

  const rng = new SeededRNG(seed);
  const edgePCounts = new Float64Array(a * a);
  const permDiffSums = new Float64Array(a * a);
  const permDiffSqSums = new Float64Array(a * a);

  for (let it = 0; it < iter; it++) {
    let permIdx: number[];
    if (paired) {
      permIdx = Array.from({ length: nXY }, (_, i) => i);
      for (let p = 0; p < nX; p++) {
        if (rng.random() < 0.5) {
          [permIdx[p], permIdx[nX + p]] = [permIdx[nX + p]!, permIdx[p]!];
        }
      }
    } else {
      permIdx = rng.permutation(nXY);
    }

    const transPermX: Matrix[] = [];
    const transPermY: Matrix[] = [];
    for (let i = 0; i < nX; i++) {
      transPermX.push(combinedTrans[permIdx[i]!]!);
    }
    for (let i = nX; i < nXY; i++) {
      transPermY.push(combinedTrans[permIdx[i]!]!);
    }

    const wPermX = computeWeightsFrom3D(transPermX, modelType, modelScaling);
    const wPermY = computeWeightsFrom3D(transPermY, modelType, modelScaling);

    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const idx = i * a + j;
        const diff = wPermX.get(i, j) - wPermY.get(i, j);
        permDiffSums[idx]! += diff;
        permDiffSqSums[idx]! += diff * diff;
        if (Math.abs(diff) >= absTrueDiff[idx]!) {
          edgePCounts[idx]!++;
        }
      }
    }
  }

  // P-values
  const rawPValues = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    rawPValues[i] = (edgePCounts[i]! + 1) / (iter + 1);
  }

  // Adjust p-values (column-major flatten to match R)
  const colMajorP: number[] = [];
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      colMajorP.push(rawPValues[i * a + j]!);
    }
  }
  const adjustedColMajor = pAdjust(colMajorP, adjust);

  const adjustedP = new Float64Array(a * a);
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      adjustedP[i * a + j] = adjustedColMajor[j * a + i]!;
    }
  }

  // Effect sizes
  const effectSizes = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    const mean = permDiffSums[i]! / iter;
    const variance = (permDiffSqSums[i]! / iter) - mean * mean;
    const sd = iter > 1 ? Math.sqrt((variance * iter) / (iter - 1)) : 0;
    effectSizes[i] = sd > 0 ? trueDiff[i]! / sd : NaN;
  }

  // Significant-only differences
  const diffSig = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    diffSig[i] = adjustedP[i]! < level ? trueDiff[i]! : 0;
  }

  // Edge stats (column-major order to match R)
  const edgeStats: EdgeStat[] = [];
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      const idx = i * a + j;
      edgeStats.push({
        from: labels[i]!,
        to: labels[j]!,
        diffTrue: trueDiff[idx]!,
        effectSize: effectSizes[idx]!,
        pValue: adjustedP[idx]!,
      });
    }
  }

  return { edgeStats, diffTrue: trueDiff, diffSig, pValues: adjustedP, labels, nStates: a, level };
}

/** Input for WTNA permutation test. */
export interface PermutationWtnaInput {
  records: Record<string, string | number>[];
  codes: string[];
  wtnaOpts: WtnaOptions;
  modelType: string;
  scaling: string | null | '';
}

/**
 * Permutation test for WTNA models.
 * Pools windowed groups from both inputs, randomly partitions,
 * rebuilds WTNA matrices per iteration, compares weight differences.
 */
export function permutationTestWtna(
  inputX: PermutationWtnaInput,
  inputY: PermutationWtnaInput,
  modelX: TNA,
  modelY: TNA,
  options: PermutationOptions = {},
): PermutationResult {
  const { iter = 1000, adjust = 'none', level = 0.05, seed = 42 } = options;

  const labels = inputX.codes;
  const a = labels.length;

  if (a !== inputY.codes.length || !labels.every((l, i) => l === inputY.codes[i])) {
    throw new Error('Both WTNA inputs must have the same state codes');
  }

  const modelType = inputX.modelType;
  const scaling = inputX.scaling || null;

  const groupsX = buildWindowedGroups(inputX);
  const groupsY = buildWindowedGroups(inputY);
  const nX = groupsX.length;
  const nY = groupsY.length;
  const combined = [...groupsX, ...groupsY];
  const nXY = nX + nY;

  const trueDiff = new Float64Array(a * a);
  const absTrueDiff = new Float64Array(a * a);
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < a; j++) {
      const idx = i * a + j;
      trueDiff[idx] = modelX.weights.get(i, j) - modelY.weights.get(i, j);
      absTrueDiff[idx] = Math.abs(trueDiff[idx]!);
    }
  }

  const rng = new SeededRNG(seed);
  const edgePCounts = new Float64Array(a * a);
  const permDiffSums = new Float64Array(a * a);
  const permDiffSqSums = new Float64Array(a * a);

  for (let it = 0; it < iter; it++) {
    const permIdx = rng.permutation(nXY);

    const mX = buildMatrixFromGroups(permIdx.slice(0, nX).map(i => combined[i]!), a);
    const mY = buildMatrixFromGroups(permIdx.slice(nX).map(i => combined[i]!), a);

    const matX = modelType === 'tna' ? rowNormalizeWtna(mX) : mX;
    const matY = modelType === 'tna' ? rowNormalizeWtna(mY) : mY;

    const wtnaType = (modelType === 'tna' || !!scaling) ? 'relative' : 'frequency';
    const optsX: Record<string, unknown> = { type: wtnaType, labels };
    const optsY: Record<string, unknown> = { type: wtnaType, labels };
    if (scaling) { optsX.scaling = scaling; optsY.scaling = scaling; }

    const tempX = tnajBuildModel(matX, optsX as any) as TNA;
    const tempY = tnajBuildModel(matY, optsY as any) as TNA;

    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const idx = i * a + j;
        const diff = tempX.weights.get(i, j) - tempY.weights.get(i, j);
        permDiffSums[idx]! += diff;
        permDiffSqSums[idx]! += diff * diff;
        if (Math.abs(diff) >= absTrueDiff[idx]!) {
          edgePCounts[idx]!++;
        }
      }
    }
  }

  const rawPValues = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    rawPValues[i] = (edgePCounts[i]! + 1) / (iter + 1);
  }

  const colMajorP: number[] = [];
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      colMajorP.push(rawPValues[i * a + j]!);
    }
  }
  const adjustedColMajor = pAdjust(colMajorP, adjust);

  const adjustedP = new Float64Array(a * a);
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      adjustedP[i * a + j] = adjustedColMajor[j * a + i]!;
    }
  }

  const effectSizes = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    const mean = permDiffSums[i]! / iter;
    const variance = (permDiffSqSums[i]! / iter) - mean * mean;
    const sd = iter > 1 ? Math.sqrt((variance * iter) / (iter - 1)) : 0;
    effectSizes[i] = sd > 0 ? trueDiff[i]! / sd : NaN;
  }

  const diffSig = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    diffSig[i] = adjustedP[i]! < level ? trueDiff[i]! : 0;
  }

  const edgeStats: EdgeStat[] = [];
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      const idx = i * a + j;
      edgeStats.push({
        from: labels[i]!,
        to: labels[j]!,
        diffTrue: trueDiff[idx]!,
        effectSize: effectSizes[idx]!,
        pValue: adjustedP[idx]!,
      });
    }
  }

  return { edgeStats, diffTrue: trueDiff, diffSig, pValues: adjustedP, labels, nStates: a, level };
}

/** Build windowed groups from a WTNA input for permutation resampling. */
function buildWindowedGroups(input: PermutationWtnaInput): number[][][] {
  const { records, codes, wtnaOpts } = input;
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

  const groups: number[][][] = [];
  for (const grp of groupMap.values()) {
    const X = toBinaryMatrix(grp, codes);
    let W: number[][];
    if (sessionKey) {
      const lbls = grp.map(rec => String(rec[sessionKey] ?? ''));
      W = applyIntervalWindowing(X, lbls);
    } else {
      W = applyWindowing(X, windowSize, windowType);
    }
    if (W.length >= 2) groups.push(W);
  }
  return groups;
}

/** Build a raw frequency transition matrix from selected windowed groups. */
function buildMatrixFromGroups(groups: number[][][], k: number): number[][] {
  const M: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  for (const W of groups) {
    const T = computeWtnaTransitions(W);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        M[i]![j]! += T[i]?.[j] ?? 0;
      }
    }
  }
  return M;
}
