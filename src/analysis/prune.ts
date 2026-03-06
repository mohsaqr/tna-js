/**
 * Pruning functions for TNA models.
 * Port of Python tna/prune.py + disparity filter (Serrano et al. 2009).
 */
import type { TNA, GroupTNA } from '../core/types.js';
import { isGroupTNA, groupEntries } from '../core/group.js';

export interface PruneOptions {
  method?: 'threshold' | 'disparity';
  /** Minimum edge weight for threshold method (default 0.1). */
  threshold?: number;
  /** Significance level for disparity filter (default 0.05). */
  alpha?: number;
}

/**
 * Prune edges below a weight threshold.
 * Accepts either a simple threshold number (backward compatible)
 * or an options object for method selection.
 */
export function prune(
  model: TNA | GroupTNA,
  thresholdOrOptions: number | PruneOptions = 0.1,
): TNA | Record<string, TNA> {
  if (isGroupTNA(model)) {
    const result: Record<string, TNA> = {};
    for (const [name, m] of groupEntries(model)) {
      result[name] = prune(m, thresholdOrOptions) as TNA;
    }
    return result;
  }

  const tnaModel = model as TNA;
  const opts: PruneOptions = typeof thresholdOrOptions === 'number'
    ? { method: 'threshold', threshold: thresholdOrOptions }
    : thresholdOrOptions;

  const method = opts.method ?? 'threshold';

  if (method === 'disparity') {
    return pruneDisparity(tnaModel, opts.alpha ?? 0.05);
  }

  const threshold = opts.threshold ?? 0.1;
  const weights = tnaModel.weights.map((v) => (v < threshold ? 0 : v));

  return {
    weights,
    inits: new Float64Array(tnaModel.inits),
    labels: [...tnaModel.labels],
    data: tnaModel.data,
    type: tnaModel.type,
    scaling: [...tnaModel.scaling],
  };
}

/**
 * Disparity filter backbone extraction (Serrano et al. 2009).
 *
 * For each edge (i,j), tests whether the normalized weight p_ij = w_ij / s_i
 * is significant against a uniform null model where k_i-1 is the degree of node i.
 * The p-value is (1 - p_ij)^(k_i - 1). Edges with p-value >= alpha are removed.
 */
export function pruneDisparity(model: TNA, alpha = 0.05): TNA {
  const n = model.weights.rows;
  const weights = model.weights.clone();

  for (let i = 0; i < n; i++) {
    // Compute strength and degree of node i
    let strength = 0;
    let degree = 0;
    for (let j = 0; j < n; j++) {
      const w = weights.get(i, j);
      if (w > 0) { strength += w; degree++; }
    }

    if (degree <= 1 || strength <= 0) continue;

    for (let j = 0; j < n; j++) {
      const w = weights.get(i, j);
      if (w <= 0) continue;

      const pij = w / strength;
      // Significance: probability of observing this weight under uniform null
      const pValue = Math.pow(1 - pij, degree - 1);

      if (pValue >= alpha) {
        weights.set(i, j, 0);
      }
    }
  }

  return {
    weights,
    inits: new Float64Array(model.inits),
    labels: [...model.labels],
    data: model.data,
    type: model.type,
    scaling: [...model.scaling],
  };
}
