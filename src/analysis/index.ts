/**
 * Analysis module: centralities, pruning, cliques, communities, compare, cluster,
 * WTNA, bootstrap, permutation, stability, reliability, simulate.
 */

export { centralities, betweennessNetwork, AVAILABLE_MEASURES } from './centralities.js';
export { prune, pruneDisparity, type PruneOptions } from './prune.js';
export { cliques } from './cliques.js';
export { communities, AVAILABLE_METHODS } from './communities.js';
export { compareSequences, compareModels } from './compare.js';
export { clusterSequences, clusterData } from './cluster.js';

// WTNA
export {
  buildWtnaMatrix,
  toBinaryMatrix, applyWindowing, applyIntervalWindowing,
  computeWtnaTransitions, computeWithinWindow, rowNormalizeWtna,
  type WtnaOptions, type WtnaResult,
} from './wtna.js';

// Bootstrap
export {
  bootstrapTna, bootstrapWtna,
  type BootstrapEdge, type BootstrapResult, type BootstrapOptions,
  type BootstrapWtnaInput,
} from './bootstrap.js';

// Permutation
export {
  permutationTest, permutationTestWtna,
  type EdgeStat, type PermutationResult, type PermutationOptions,
  type PermutationWtnaInput,
} from './permutation.js';

// Stability
export {
  estimateCS, estimateCsWtna,
  estimateEdgeStability, estimateNetworkStability,
  type StabilityResult, type StabilityOptions,
  type StabilityWtnaInput,
  type EdgeStabilityResult, type NetworkStabilityResult,
} from './stability.js';

// Reliability
export {
  compareWeightMatrices, reliabilityAnalysis,
  RELIABILITY_METRICS,
  type MetricDef, type ReliabilityMetricSummary, type ReliabilityResult,
} from './reliability.js';

// Simulate
export { simulate, type SimulateOptions } from './simulate.js';
