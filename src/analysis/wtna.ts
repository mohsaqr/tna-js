/**
 * Window-based Transition Network Analysis (WTNA).
 * Port of R wtna() function / Desktop analysis/wtna.ts.
 *
 * Computes a directed transition matrix M where M[i,j] counts how often
 * state i in window t leads to state j in window t+1, accumulated across all actors.
 *
 * R parameter mapping:
 *   actor=   → groups data; each actor is one independent stream
 *   session= → variable-size windowing column
 *   interval → fixed window size (default 3)
 */

export interface WtnaOptions {
  /** Grouping column. Each unique actor value is one independent stream. */
  actor?: string;
  /** Variable-size windowing column (R's session=). */
  session?: string;
  /** Fixed window size in rows (R's interval=, default 3). */
  windowSize?: number;
  /** Windowing strategy for fixed-size windows. */
  windowType?: 'tumbling' | 'sliding';
  type?: 'frequency' | 'relative';
}

export interface WtnaResult {
  matrix: number[][];
  withinMatrix: number[][];
  labels: string[];
}

/** Build a binary matrix (n_records × n_codes) from records. */
export function toBinaryMatrix(
  records: Record<string, string | number>[],
  codes: string[],
): number[][] {
  return records.map(rec =>
    codes.map(c => {
      const v = rec[c];
      return typeof v === 'number' ? (v > 0 ? 1 : 0) : (parseInt(String(v), 10) > 0 ? 1 : 0);
    }),
  );
}

/**
 * Apply fixed-size windowing to a binary matrix X (n_rows × n_cols).
 * Tumbling: non-overlapping blocks. Sliding: overlapping windows.
 */
export function applyWindowing(
  X: number[][],
  windowSize: number,
  mode: 'tumbling' | 'sliding',
): number[][] {
  const n = X.length;
  const k = X[0]?.length ?? 0;
  if (windowSize <= 1) return X;

  if (mode === 'tumbling') {
    const nBlocks = Math.ceil(n / windowSize);
    const result: number[][] = Array.from({ length: nBlocks }, () => new Array<number>(k).fill(0));
    for (let i = 0; i < n; i++) {
      const block = Math.floor(i / windowSize);
      for (let j = 0; j < k; j++) {
        result[block]![j] = (result[block]![j]! | (X[i]?.[j] ?? 0));
      }
    }
    return result;
  } else {
    const nWindows = Math.max(0, n - windowSize + 1);
    const result: number[][] = Array.from({ length: nWindows }, () => new Array<number>(k).fill(0));
    for (let i = 0; i < nWindows; i++) {
      for (let t = i; t < i + windowSize; t++) {
        for (let j = 0; j < k; j++) {
          result[i]![j] = (result[i]![j]! | (X[t]?.[j] ?? 0));
        }
      }
    }
    return result;
  }
}

/**
 * Apply variable-size windowing using row labels (R's session= parameter).
 * Consecutive rows sharing the same label are OR-reduced into one window.
 */
export function applyIntervalWindowing(X: number[][], labels: string[]): number[][] {
  const k = X[0]?.length ?? 0;
  const result: number[][] = [];
  let prevLabel: string | null = null;
  for (let i = 0; i < X.length; i++) {
    const lbl = labels[i] ?? '';
    if (lbl !== prevLabel) {
      result.push(new Array<number>(k).fill(0));
      prevLabel = lbl;
    }
    const win = result[result.length - 1]!;
    for (let j = 0; j < k; j++) {
      win[j]! |= X[i]?.[j] ?? 0;
    }
  }
  return result;
}

/**
 * Compute the transition matrix T from windowed binary matrix W.
 * T[i,j] = Σ_t W[t,i] * W[t+1,j]
 *
 * NOTE: This is WTNA-specific, operating on plain number[][].
 * Not to be confused with core/transitions.ts computeTransitions (SequenceData → Matrix).
 */
export function computeWtnaTransitions(W: number[][]): number[][] {
  const n = W.length;
  const k = W[0]?.length ?? 0;
  const T: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  if (n < 2) return T;
  for (let t = 0; t < n - 1; t++) {
    const rowT  = W[t]!;
    const rowT1 = W[t + 1]!;
    for (let i = 0; i < k; i++) {
      if (!rowT[i]) continue;
      for (let j = 0; j < k; j++) {
        T[i]![j]! += rowT[i]! * rowT1[j]!;
      }
    }
  }
  return T;
}

/**
 * Compute within-window co-occurrence matrix from windowed binary matrix W.
 * C[i,j] = Σ_t W[t,i] · W[t,j] for i ≠ j (zero diagonal).
 */
export function computeWithinWindow(W: number[][]): number[][] {
  const n = W.length;
  const k = W[0]?.length ?? 0;
  const C: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  for (let t = 0; t < n; t++) {
    const row = W[t]!;
    for (let i = 0; i < k; i++) {
      if (!row[i]) continue;
      for (let j = 0; j < k; j++) {
        if (i !== j && row[j]) C[i]![j]!++;
      }
    }
  }
  return C;
}

/** Add matrix B into matrix A in-place (same dimensions). */
function addMatrix(A: number[][], B: number[][]): void {
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < (A[i]?.length ?? 0); j++) {
      A[i]![j]! += B[i]?.[j] ?? 0;
    }
  }
}

/** Row-normalize matrix M (type = 'relative'). */
export function rowNormalizeWtna(M: number[][]): number[][] {
  return M.map(row => {
    const s = row.reduce((a, b) => a + b, 0);
    return s > 0 ? row.map(v => v / s) : row.slice();
  });
}

/**
 * Compute the WTNA transition matrix from one-hot records.
 *
 * @param records Flat array of data rows (binary cols + optional actor/session strings).
 * @param codes   Column names for the binary state columns.
 * @param opts    Actor/session column names, window settings, weight type.
 */
export function buildWtnaMatrix(
  records: Record<string, string | number>[],
  codes: string[],
  opts: WtnaOptions = {},
): WtnaResult {
  const windowSize = Math.max(1, opts.windowSize ?? 3);
  const windowType = opts.windowType ?? 'tumbling';
  const type       = opts.type ?? 'frequency';
  const actorKey   = opts.actor;
  const sessionKey = opts.session;
  const k          = codes.length;

  const M:  number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const Mc: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));

  const groups = new Map<string, Record<string, string | number>[]>();
  for (const rec of records) {
    const actor = actorKey ? String(rec[actorKey] ?? '') : '__all__';
    if (!groups.has(actor)) groups.set(actor, []);
    groups.get(actor)!.push(rec);
  }

  for (const grp of groups.values()) {
    const X = toBinaryMatrix(grp, codes);
    let W: number[][];
    if (sessionKey) {
      const labels = grp.map(rec => String(rec[sessionKey] ?? ''));
      W = applyIntervalWindowing(X, labels);
    } else {
      W = applyWindowing(X, windowSize, windowType);
    }
    const T  = computeWtnaTransitions(W);
    const Cw = computeWithinWindow(W);
    addMatrix(M,  T);
    addMatrix(Mc, Cw);
  }

  const matrix = type === 'relative' ? rowNormalizeWtna(M) : M;
  return { matrix, withinMatrix: Mc, labels: codes };
}
