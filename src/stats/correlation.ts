/**
 * Correlation and association measures for TNA.
 * Ported from Desktop analysis/stability.ts and analysis/reliability.ts.
 */
import { pearsonCorr } from '../core/matrix.js';

/**
 * Rank array values (1-based, average ties).
 * Used internally by spearmanCorr and kendallTau.
 */
export function rankArray(arr: Float64Array): Float64Array {
  const indexed = Array.from(arr, (v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Float64Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j]!.v === indexed[i]!.v) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) ranks[indexed[k]!.i] = avgRank;
    i = j;
  }
  return ranks;
}

/** Rank number[] values (1-based, average ties). */
function rankArr(a: number[]): number[] {
  const indexed = a.map((v, i) => ({ v, i })).sort((x, y) => x.v - y.v);
  const ranks = new Array<number>(a.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j]!.v === indexed[i]!.v) j++;
    const avg = (i + j + 1) / 2;
    for (let k = i; k < j; k++) ranks[indexed[k]!.i] = avg;
    i = j;
  }
  return ranks;
}

function arrMean(a: number[]): number {
  if (a.length === 0) return NaN;
  return a.reduce((s, v) => s + v, 0) / a.length;
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

/**
 * Spearman rank correlation = Pearson correlation on ranks.
 * Accepts Float64Array (from centrality vectors).
 */
export function spearmanCorr(a: Float64Array, b: Float64Array): number {
  return pearsonCorr(rankArray(a), rankArray(b));
}

/**
 * Spearman rank correlation for number[] arrays.
 * Used by reliability metrics.
 */
export function spearmanCorrArr(x: number[], y: number[]): number {
  return pearsonCorrArr(rankArr(x), rankArr(y));
}

/**
 * Kendall's tau-b (matches R's cor(x, y, method='kendall')).
 * tau-b = (C - D) / sqrt((n0 - Tx) * (n0 - Ty))
 */
export function kendallTau(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return NaN;
  let concordant = 0, discordant = 0, tx = 0, ty = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const sx = Math.sign(x[i]! - x[j]!);
      const sy = Math.sign(y[i]! - y[j]!);
      if (sx === sy && sx !== 0) concordant++;
      else if (sx !== 0 && sy !== 0) discordant++;
      if (sx === 0) tx++;
      if (sy === 0) ty++;
    }
  }
  const n0 = n * (n - 1) / 2;
  const denom = Math.sqrt((n0 - tx) * (n0 - ty));
  return denom < 1e-14 ? NaN : (concordant - discordant) / denom;
}

/**
 * Distance correlation matching R's tna:::distance_correlation.
 * Returns v_xy / sqrt(v_x * v_y) (biased estimator; can be negative).
 */
export function distanceCorr(x: number[], y: number[]): number {
  const m = x.length;
  if (m < 2) return NaN;

  const center = (vals: number[]): number[][] => {
    const d = Array.from({ length: m }, (_, i) =>
      Array.from({ length: m }, (__, j) => Math.abs(vals[i]! - vals[j]!)),
    );
    const rowMeans = d.map(row => arrMean(row));
    const grandMean = arrMean(rowMeans);
    return d.map((row, i) =>
      row.map((v, j) => v - rowMeans[i]! - rowMeans[j]! + grandMean),
    );
  };

  const A = center(x);
  const B = center(y);

  let vXY = 0, vX = 0, vY = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      vXY += A[i]![j]! * B[i]![j]!;
      vX  += A[i]![j]! * A[i]![j]!;
      vY  += B[i]![j]! * B[i]![j]!;
    }
  }
  const n2 = m * m;
  vXY /= n2;
  vX  /= n2;
  vY  /= n2;

  const denom = Math.sqrt(vX * vY);
  return denom < 1e-14 ? NaN : vXY / denom;
}

/**
 * RV coefficient matching R's tna:::rv_coefficient.
 * Uses column-centred matrices and tcrossprod formula:
 * RV = trace(XX' * YY') / sqrt(trace(XX' * XX') * trace(YY' * YY'))
 *
 * Accepts any object with .rows, .get(i,j) — compatible with tnaj Matrix.
 */
export function rvCoefficient(
  a: { rows: number; get(i: number, j: number): number },
  b: { rows: number; get(i: number, j: number): number },
): number {
  const n = a.rows;

  const colCenter = (w: { rows: number; get(i: number, j: number): number }): number[][] => {
    const result: number[][] = [];
    for (let i = 0; i < n; i++) result.push(new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      let colSum = 0;
      for (let i = 0; i < n; i++) colSum += w.get(i, j);
      const colMean = colSum / n;
      for (let i = 0; i < n; i++) result[i]![j] = w.get(i, j) - colMean;
    }
    return result;
  };

  const tcrossprod = (x: number[][]): number[][] => {
    const mat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += x[i]![k]! * x[j]![k]!;
        mat[i]![j] = s;
      }
    }
    return mat;
  };

  const traceMul = (P: number[][], Q: number[][]): number => {
    let tr = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        tr += P[i]![j]! * Q[j]![i]!;
    return tr;
  };

  const xc = colCenter(a);
  const yc = colCenter(b);
  const xx = tcrossprod(xc);
  const yy = tcrossprod(yc);

  const trXXYY = traceMul(xx, yy);
  const trXXXX = traceMul(xx, xx);
  const trYYYY = traceMul(yy, yy);

  const denom = Math.sqrt(trXXXX * trYYYY);
  return denom < 1e-14 ? NaN : trXXYY / denom;
}
