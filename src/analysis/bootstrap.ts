/**
 * Bootstrap resampling for TNA model stability testing.
 * Port of Desktop analysis/bootstrap.ts.
 * Uses tnaj's computeTransitions3D / computeWeightsFrom3D for R equivalence.
 */
import type { TNA } from '../core/types.js';
import type { Matrix } from '../core/matrix.js';
import { computeTransitions3D, computeWeightsFrom3D } from '../core/transitions.js';
import { createTNA, buildModel as tnajBuildModel } from '../core/model.js';
import { SeededRNG } from '../core/rng.js';
import { arrayQuantile } from '../core/matrix.js';
import {
  toBinaryMatrix, applyWindowing, applyIntervalWindowing,
  computeWtnaTransitions, rowNormalizeWtna,
} from './wtna.js';
import type { WtnaOptions } from './wtna.js';

export interface BootstrapEdge {
  from: string;
  to: string;
  weight: number;
  bootstrapMean: number;
  bias: number;
  pValue: number;
  significant: boolean;
  crLower: number;
  crUpper: number;
  ciLower: number;
  ciUpper: number;
}

export interface BootstrapResult {
  edges: BootstrapEdge[];
  model: TNA;
  labels: string[];
  method: string;
  iter: number;
  level: number;
  weightsMean: Float64Array;
  weightsSd: Float64Array;
  weightsBias: Float64Array;
}

export interface BootstrapOptions {
  iter?: number;
  level?: number;
  method?: 'stability' | 'threshold';
  threshold?: number;
  consistencyRange?: [number, number];
  seed?: number;
}

/**
 * Bootstrap a TNA model to assess edge stability.
 * The model must have sequence data (model.data).
 */
export function bootstrapTna(
  model: TNA,
  options: BootstrapOptions = {},
): BootstrapResult {
  const {
    iter = 1000,
    level = 0.05,
    method = 'stability',
    consistencyRange = [0.75, 1.25],
    seed = 42,
  } = options;

  if (!model.data) {
    throw new Error('TNA model must have sequence data for bootstrap');
  }

  const labels = model.labels;
  const a = labels.length;
  const seqData = model.data;
  const n = seqData.length;
  const modelType = model.type;
  const modelScaling = model.scaling.length > 0 ? model.scaling : null;

  // Pad sequences to uniform length
  let maxLen = 0;
  for (const seq of seqData) { if (seq.length > maxLen) maxLen = seq.length; }
  const padded = seqData.map(seq => {
    if (seq.length >= maxLen) return seq;
    const pad: (string | null)[] = new Array(maxLen - seq.length).fill(null);
    return [...seq, ...pad];
  });

  const trans = computeTransitions3D(padded, labels, modelType, model.params);
  const weights = computeWeightsFrom3D(trans, modelType, modelScaling);

  // Default threshold: 10th percentile of weights
  let threshold = options.threshold;
  if (threshold === undefined) {
    const allW: number[] = [];
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        allW.push(weights.get(i, j));
      }
    }
    allW.sort((x, y) => x - y);
    const p10Idx = Math.floor(allW.length * 0.1);
    threshold = allW[p10Idx] ?? 0;
  }

  const rng = new SeededRNG(seed);

  // Per-sequence row totals for stability NA handling
  const seqRowTotals = new Float64Array(n * a);
  for (let seqIdx = 0; seqIdx < n; seqIdx++) {
    const t = trans[seqIdx]!;
    for (let i = 0; i < a; i++) {
      let rowSum = 0;
      for (let j = 0; j < a; j++) rowSum += t.get(i, j);
      seqRowTotals[seqIdx * a + i] = rowSum;
    }
  }

  const pCounts = new Float64Array(a * a);
  const bootSums = new Float64Array(a * a);
  const bootSqSums = new Float64Array(a * a);
  const bootWeights: Float64Array[] = [];
  for (let i = 0; i < a * a; i++) {
    bootWeights.push(new Float64Array(iter));
  }

  for (let it = 0; it < iter; it++) {
    const bootIdx = rng.choice(n, n);
    const transBoot: Matrix[] = [];
    for (let i = 0; i < n; i++) {
      transBoot.push(trans[bootIdx[i]!]!);
    }

    const wBoot = computeWeightsFrom3D(transBoot, modelType, modelScaling);

    const bootRowTotals = new Float64Array(a);
    for (let k = 0; k < n; k++) {
      const seqIdx = bootIdx[k]!;
      for (let i = 0; i < a; i++) {
        bootRowTotals[i]! += seqRowTotals[seqIdx * a + i]!;
      }
    }

    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const idx = i * a + j;
        const wb = wBoot.get(i, j);
        const wo = weights.get(i, j);

        bootWeights[idx]![it] = wb;
        bootSums[idx]! += wb;
        bootSqSums[idx]! += wb * wb;

        if (method === 'stability') {
          if (bootRowTotals[i]! === 0) continue;
          if (wb <= wo * consistencyRange[0]! || wb >= wo * consistencyRange[1]!) {
            pCounts[idx]!++;
          }
        } else {
          if (wb < threshold) {
            pCounts[idx]!++;
          }
        }
      }
    }
  }

  // P-values
  const pValues = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    pValues[i] = (pCounts[i]! + 1) / (iter + 1);
  }

  // Confidence intervals
  const ciLower = new Float64Array(a * a);
  const ciUpper = new Float64Array(a * a);
  const halfLevel = level / 2;
  for (let idx = 0; idx < a * a; idx++) {
    ciLower[idx] = arrayQuantile(bootWeights[idx]!, halfLevel);
    ciUpper[idx] = arrayQuantile(bootWeights[idx]!, 1 - halfLevel);
  }

  // Mean and SD
  const weightsMean = new Float64Array(a * a);
  const weightsSd = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    const mean = bootSums[i]! / iter;
    weightsMean[i] = mean;
    const variance = (bootSqSums[i]! / iter) - mean * mean;
    weightsSd[i] = iter > 1 ? Math.sqrt((variance * iter) / (iter - 1)) : 0;
  }

  // Bias
  const weightsBias = new Float64Array(a * a);
  for (let i = 0; i < a; i++)
    for (let j = 0; j < a; j++) {
      const idx = i * a + j;
      weightsBias[idx] = weightsMean[idx]! - weights.get(i, j);
    }

  // Edge stats (column-major to match R ordering)
  const edges: BootstrapEdge[] = [];
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      const idx = i * a + j;
      const w = weights.get(i, j);
      if (w <= 0) continue;
      edges.push({
        from: labels[i]!,
        to: labels[j]!,
        weight: w,
        bootstrapMean: weightsMean[idx]!,
        bias: weightsBias[idx]!,
        pValue: pValues[idx]!,
        significant: pValues[idx]! < level,
        crLower: w * consistencyRange[0],
        crUpper: w * consistencyRange[1],
        ciLower: ciLower[idx]!,
        ciUpper: ciUpper[idx]!,
      });
    }
  }

  // Significant-only model
  const sigModel = createTNA(weights, model.inits, labels, model.data, model.type, model.scaling);
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < a; j++) {
      const idx = i * a + j;
      sigModel.weights.set(i, j, pValues[idx]! < level ? weights.get(i, j) : 0);
    }
  }

  return { edges, model: sigModel, labels, method, iter, level, weightsMean, weightsSd, weightsBias };
}

/** Input for WTNA bootstrap. */
export interface BootstrapWtnaInput {
  originalModel: TNA;
  records: Record<string, string | number>[];
  codes: string[];
  wtnaOpts: WtnaOptions;
  modelType: 'tna' | 'ftna';
  scaling: string | null | '';
}

/**
 * Bootstrap a WTNA model to assess edge stability via row-level resampling.
 */
export function bootstrapWtna(
  input: BootstrapWtnaInput,
  options: BootstrapOptions = {},
): BootstrapResult {
  const { originalModel, records, codes, wtnaOpts, modelType, scaling } = input;
  const {
    iter = 1000,
    level = 0.05,
    method = 'stability',
    consistencyRange = [0.75, 1.25],
    seed = 42,
  } = options;

  const labels = codes;
  const a = codes.length;
  const actorKey   = wtnaOpts.actor;
  const sessionKey = wtnaOpts.session;
  const windowSize = Math.max(1, wtnaOpts.windowSize ?? 3);
  const windowType = wtnaOpts.windowType ?? 'tumbling';

  // Build resample units
  const windowedGroups: number[][][] = [];
  {
    const groupMap = new Map<string, Record<string, string | number>[]>();
    for (const rec of records) {
      const actor = actorKey ? String(rec[actorKey] ?? '') : '__all__';
      if (!groupMap.has(actor)) groupMap.set(actor, []);
      groupMap.get(actor)!.push(rec);
    }
    if (actorKey) {
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
    } else {
      const allRecs = groupMap.get('__all__') ?? [];
      const X = toBinaryMatrix(allRecs, codes);
      let W: number[][];
      if (sessionKey) {
        const lbls = allRecs.map(rec => String(rec[sessionKey] ?? ''));
        W = applyIntervalWindowing(X, lbls);
      } else {
        W = applyWindowing(X, windowSize, windowType);
      }
      for (let t = 0; t < W.length - 1; t++) {
        windowedGroups.push([W[t]!, W[t + 1]!]);
      }
    }
  }

  const n = windowedGroups.length;
  const rng = new SeededRNG(seed);

  let threshold = options.threshold;
  if (threshold === undefined) {
    const allW: number[] = [];
    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const w = originalModel.weights.get(i, j);
        if (w > 0) allW.push(w);
      }
    }
    allW.sort((x, y) => x - y);
    threshold = allW[Math.floor(allW.length * 0.1)] ?? 0;
  }

  const pCounts   = new Float64Array(a * a);
  const bootSums  = new Float64Array(a * a);
  const bootSqSums = new Float64Array(a * a);
  const bootWeightsArr: Float64Array[] = [];
  for (let i = 0; i < a * a; i++) bootWeightsArr.push(new Float64Array(iter));

  for (let it = 0; it < iter; it++) {
    const bootIdx = rng.choice(n, n);

    const M: number[][] = Array.from({ length: a }, () => new Array<number>(a).fill(0));
    for (let k = 0; k < n; k++) {
      const W = windowedGroups[bootIdx[k]!]!;
      const T = computeWtnaTransitions(W);
      for (let i = 0; i < a; i++) {
        for (let j = 0; j < a; j++) {
          M[i]![j]! += T[i]?.[j] ?? 0;
        }
      }
    }
    let matrix = M;
    if (modelType === 'tna') matrix = rowNormalizeWtna(matrix);

    const tempType = modelType === 'tna' ? 'relative' : 'frequency';
    const tempOpts: Record<string, unknown> = { type: tempType, labels };
    if (scaling) tempOpts.scaling = scaling;
    const tempModel = tnajBuildModel(matrix, tempOpts as any) as TNA;

    for (let i = 0; i < a; i++) {
      for (let j = 0; j < a; j++) {
        const idx = i * a + j;
        const wb = tempModel.weights.get(i, j);
        const wo = originalModel.weights.get(i, j);

        bootWeightsArr[idx]![it] = wb;
        bootSums[idx]!  += wb;
        bootSqSums[idx]! += wb * wb;

        if (method === 'stability') {
          if (wb <= wo * consistencyRange[0]! || wb >= wo * consistencyRange[1]!) pCounts[idx]!++;
        } else {
          if (wb < threshold) pCounts[idx]!++;
        }
      }
    }
  }

  const pValues = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) pValues[i] = (pCounts[i]! + 1) / (iter + 1);

  const ciLower = new Float64Array(a * a);
  const ciUpper = new Float64Array(a * a);
  const halfLevel = level / 2;
  for (let idx = 0; idx < a * a; idx++) {
    ciLower[idx] = arrayQuantile(bootWeightsArr[idx]!, halfLevel);
    ciUpper[idx] = arrayQuantile(bootWeightsArr[idx]!, 1 - halfLevel);
  }

  const weightsMean = new Float64Array(a * a);
  const weightsSd   = new Float64Array(a * a);
  for (let i = 0; i < a * a; i++) {
    const mean = bootSums[i]! / iter;
    weightsMean[i] = mean;
    const variance = (bootSqSums[i]! / iter) - mean * mean;
    weightsSd[i] = iter > 1 ? Math.sqrt((variance * iter) / (iter - 1)) : 0;
  }

  const weightsBias = new Float64Array(a * a);
  for (let i = 0; i < a; i++)
    for (let j = 0; j < a; j++) {
      const idx = i * a + j;
      weightsBias[idx] = weightsMean[idx]! - originalModel.weights.get(i, j);
    }

  const edges: BootstrapEdge[] = [];
  for (let j = 0; j < a; j++) {
    for (let i = 0; i < a; i++) {
      const idx = i * a + j;
      const w = originalModel.weights.get(i, j);
      if (w <= 0) continue;
      edges.push({
        from: labels[i]!,
        to: labels[j]!,
        weight: w,
        bootstrapMean: weightsMean[idx]!,
        bias: weightsBias[idx]!,
        pValue: pValues[idx]!,
        significant: pValues[idx]! < level,
        crLower: w * consistencyRange[0],
        crUpper: w * consistencyRange[1],
        ciLower: ciLower[idx]!,
        ciUpper: ciUpper[idx]!,
      });
    }
  }

  // Significant-only model
  const sigWeights2D: number[][] = [];
  for (let i = 0; i < a; i++) {
    const row: number[] = [];
    for (let j = 0; j < a; j++) {
      row.push(pValues[i * a + j]! < level ? originalModel.weights.get(i, j) : 0);
    }
    sigWeights2D.push(row);
  }
  const sigModel = tnajBuildModel(sigWeights2D, { type: originalModel.type, labels }) as TNA;

  return { edges, model: sigModel, labels, method, iter, level, weightsMean, weightsSd, weightsBias };
}
