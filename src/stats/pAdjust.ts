/**
 * P-value adjustment methods matching R's p.adjust().
 * Ported from Desktop analysis/permutation.ts.
 */

export type PAdjustMethod = 'none' | 'bonferroni' | 'holm' | 'fdr' | 'BH';

/**
 * Adjust p-values for multiple comparisons.
 * Matches R's p.adjust() for supported methods.
 */
export function pAdjust(pvals: number[], method: PAdjustMethod): number[] {
  const n = pvals.length;
  if (method === 'none' || n <= 1) return pvals.slice();

  if (method === 'bonferroni') {
    return pvals.map(p => Math.min(p * n, 1));
  }

  if (method === 'holm') {
    const indexed = pvals.map((p, i) => ({ p, i }));
    indexed.sort((a, b) => a.p - b.p);
    const adjusted = new Array<number>(n);
    let cummax = 0;
    for (let k = 0; k < n; k++) {
      const adj = indexed[k]!.p * (n - k);
      cummax = Math.max(cummax, adj);
      adjusted[indexed[k]!.i] = Math.min(cummax, 1);
    }
    return adjusted;
  }

  if (method === 'fdr' || method === 'BH') {
    const indexed = pvals.map((p, i) => ({ p, i }));
    indexed.sort((a, b) => a.p - b.p);
    const adjusted = new Array<number>(n);
    let cummin = 1;
    for (let k = n - 1; k >= 0; k--) {
      const adj = (indexed[k]!.p * n) / (k + 1);
      cummin = Math.min(cummin, adj);
      adjusted[indexed[k]!.i] = Math.min(cummin, 1);
    }
    return adjusted;
  }

  return pvals.slice();
}
