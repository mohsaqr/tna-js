/**
 * Statistics module: correlation measures and p-value adjustment.
 */
export {
  spearmanCorr, spearmanCorrArr, kendallTau, distanceCorr, rvCoefficient,
  rankArray,
} from './correlation.js';

export { pAdjust, type PAdjustMethod } from './pAdjust.js';
