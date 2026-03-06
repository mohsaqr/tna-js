/**
 * TNA - Transition Network Analysis for JavaScript/TypeScript.
 *
 * @example
 * ```ts
 * import { tna, centralities, prune } from 'tna';
 *
 * const data = [
 *   ['A', 'B', 'C', 'A'],
 *   ['B', 'C', 'A', 'B'],
 *   ['A', 'C', 'B', 'A'],
 * ];
 *
 * const model = tna(data);
 * const cent = centralities(model);
 * const pruned = prune(model, 0.1);
 * ```
 */

// Core
export {
  // Matrix
  Matrix, rowNormalize, minmaxScale, maxScale, rankScale, applyScaling,
  arrayMean, arrayStd, pearsonCorr, arrayQuantile,
  // Model
  createTNA, buildModel, tna, ftna, ctna, atna, summary,
  // Prepare
  createSeqdata, prepareData, importOnehot, type OnehotSequenceData,
  // Transitions
  computeTransitions, computeTransitions3D, computeWeightsFrom3D, computeWeightsFromMatrix,
  // Group
  isGroupTNA, createGroupTNA, groupNames, groupEntries, groupApply, renameGroups,
  groupTna, groupFtna, groupCtna, groupAtna,
  // Colors
  colorPalette, DEFAULT_COLORS, ACCENT_PALETTE, SET3_PALETTE,
  hexToRgb, rgbToHex, lightenColor, darkenColor, createColorMap,
  // RNG
  SeededRNG,
} from './core/index.js';

// Stats
export {
  spearmanCorr, spearmanCorrArr, kendallTau, distanceCorr, rvCoefficient, rankArray,
  pAdjust, type PAdjustMethod,
} from './stats/index.js';

// Analysis
export {
  centralities, betweennessNetwork, AVAILABLE_MEASURES,
  prune, pruneDisparity, type PruneOptions,
  cliques,
  communities, AVAILABLE_METHODS,
  compareSequences, compareModels,
  clusterSequences, clusterData,
  // WTNA
  buildWtnaMatrix, type WtnaOptions, type WtnaResult,
  toBinaryMatrix, applyWindowing, applyIntervalWindowing,
  computeWtnaTransitions, computeWithinWindow, rowNormalizeWtna,
  // Bootstrap
  bootstrapTna, bootstrapWtna,
  type BootstrapEdge, type BootstrapResult, type BootstrapOptions, type BootstrapWtnaInput,
  // Permutation
  permutationTest, permutationTestWtna,
  type EdgeStat, type PermutationResult, type PermutationOptions, type PermutationWtnaInput,
  // Stability
  estimateCS, estimateCsWtna, estimateEdgeStability, estimateNetworkStability,
  type StabilityResult, type StabilityOptions, type StabilityWtnaInput,
  type EdgeStabilityResult, type NetworkStabilityResult,
  // Reliability
  compareWeightMatrices, reliabilityAnalysis, RELIABILITY_METRICS,
  type MetricDef, type ReliabilityMetricSummary, type ReliabilityResult,
  // Simulate
  simulate, type SimulateOptions,
} from './analysis/index.js';

// Types
export type {
  TNA, GroupTNA, TNAData, SequenceData, Sequence, ModelType, TransitionParams, BuildModelOptions,
  CentralityMeasure, CentralityResult, CliqueResult, CommunityResult, CommunityMethod,
  ClusterResult, CompareRow,
} from './core/types.js';
