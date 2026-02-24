/**
 * Data preparation functions.
 * Port of Python tna/prepare.py
 */
import type { SequenceData, TNAData } from './types.js';

/** Result of importOnehot with window metadata for windowed co-occurrence. */
export interface OnehotSequenceData {
  sequences: SequenceData;
  windowSize: number;
  windowSpan: number;
}

/**
 * Create sequence data from a 2D string array (wide format).
 * Extracts unique state labels and optionally adds begin/end states.
 */
export function createSeqdata(
  data: SequenceData,
  options?: {
    beginState?: string;
    endState?: string;
  },
): { data: SequenceData; labels: string[] } {
  // Get unique states
  const stateSet = new Set<string>();
  for (const row of data) {
    for (const val of row) {
      if (val !== null && val !== undefined && val !== '') {
        stateSet.add(val);
      }
    }
  }
  const labels = Array.from(stateSet).sort();

  // Add begin/end states
  if (options?.beginState && !labels.includes(options.beginState)) {
    labels.unshift(options.beginState);
  }
  if (options?.endState && !labels.includes(options.endState)) {
    labels.push(options.endState);
  }

  let result = data;

  // Prepend begin state
  if (options?.beginState) {
    result = result.map((row) => [options.beginState!, ...row]);
  }

  // Append end state
  if (options?.endState) {
    result = result.map((row) => [...row, options.endState!]);
  }

  return { data: result, labels };
}

/**
 * Parse wide-format data into a TNAData object.
 * Input: array of arrays where each inner array is a sequence.
 */
export function prepareData(
  data: SequenceData,
  options?: {
    beginState?: string;
    endState?: string;
  },
): TNAData {
  const { data: seqData, labels } = createSeqdata(data, options);

  // Compute statistics
  const actionCounts = new Map<string, number>();
  let totalActions = 0;
  let totalLength = 0;
  let maxLen = 0;

  for (const row of seqData) {
    let rowLen = 0;
    for (const val of row) {
      if (val !== null && val !== undefined && val !== '') {
        actionCounts.set(val, (actionCounts.get(val) ?? 0) + 1);
        totalActions++;
        rowLen++;
      }
    }
    totalLength += rowLen;
    if (rowLen > maxLen) maxLen = rowLen;
  }

  return {
    sequenceData: seqData,
    labels,
    statistics: {
      nSessions: seqData.length,
      nUniqueActions: labels.length,
      uniqueActions: labels,
      maxSequenceLength: maxLen,
      meanSequenceLength: seqData.length > 0 ? totalLength / seqData.length : 0,
    },
  };
}

/**
 * Convert one-hot encoded data into wide-format sequence data.
 *
 * Matches R's import_onehot() from the tna package (dev branch).
 *
 * Two-level windowing hierarchy:
 *   - windowSize (R's window_size): rows per sub-window (default 1).
 *   - interval   (R's interval):    sub-windows per output sequence
 *                                   (default = all sub-windows in one sequence).
 *
 * With windowSize=1, interval=3: every 3 consecutive rows form one output
 * sequence, within which each row is treated as its own co-occurrence window.
 * This matches R's default when window_size=1 and interval=3.
 *
 * @param data - Array of records with 0/1 values for each column
 * @param cols - Column names that are one-hot encoded state indicators
 * @param options - windowing options
 * @returns OnehotSequenceData with sequences and window metadata
 */
export function importOnehot(
  data: Record<string, number | string>[],
  cols: string[],
  options?: {
    actor?: string;
    session?: string;
    /** R's window_size: rows per sub-window (default 1). */
    windowSize?: number;
    /** R's interval: sub-windows per output sequence (default = all sub-windows). */
    interval?: number;
    windowType?: 'tumbling' | 'sliding';
    aggregate?: boolean;
  },
): OnehotSequenceData {
  const windowSize = options?.windowSize ?? 1;
  const windowType = options?.windowType ?? 'tumbling';
  const aggregate = options?.aggregate ?? false;

  // Decode: 1 -> column name, 0 -> null
  const decoded: (string | null)[][] = data.map((row) =>
    cols.map((col) => (row[col] === 1 ? col : null)),
  );

  // Group by actor/session if provided
  const groups: (string | null)[][][] = [];
  if (options?.actor || options?.session) {
    const groupMap = new Map<string, (string | null)[][]>();
    for (let i = 0; i < data.length; i++) {
      const parts: string[] = [];
      if (options?.actor) parts.push(String(data[i]![options.actor] ?? ''));
      if (options?.session) parts.push(String(data[i]![options.session] ?? ''));
      const key = parts.join('_');
      if (!groupMap.has(key)) groupMap.set(key, []);
      groupMap.get(key)!.push(decoded[i]!);
    }
    for (const rows of groupMap.values()) {
      groups.push(rows);
    }
  } else {
    groups.push(decoded);
  }

  // Process each group into interval-grouped output sequences
  const result: (string | null)[][] = [];

  for (const groupRows of groups) {
    const nRows = groupRows.length;

    // ── Step 1: Build all sub-windows for this group ──────────────────────
    // Each sub-window is an array of (string | null) values (flattened cols).
    const allWindows: (string | null)[][] = [];

    if (windowType === 'sliding') {
      // R's sliding window: iterative lag expansion per column,
      // then remove first row. Each remaining row becomes one window.
      const active: boolean[][] = cols.map((_, c) =>
        Array.from({ length: nRows }, (__, r) => groupRows[r]![c] !== null),
      );

      // R: seq(1, ws-1) for ws=1 gives c(1,0) — always applies lag(1).
      const maxW = Math.max(windowSize, 2);
      for (let w = 1; w < maxW; w++) {
        for (let c = 0; c < cols.length; c++) {
          const prev = active[c]!.slice();
          for (let r = 0; r < nRows; r++) {
            active[c]![r] = prev[r]! || (r >= w && prev[r - w]!);
          }
        }
      }

      // Each remaining row (after first) is a window of ncols entries
      for (let r = 1; r < nRows; r++) {
        const windowVals: (string | null)[] = [];
        for (let c = 0; c < cols.length; c++) {
          windowVals.push(active[c]![r] ? cols[c]! : null);
        }
        allWindows.push(windowVals);
      }
    } else {
      // Tumbling windows: chunk rows into windowSize-row sub-windows
      for (let start = 0; start < nRows; start += windowSize) {
        const windowRows = groupRows.slice(start, Math.min(start + windowSize, nRows));
        const windowVals: (string | null)[] = [];

        if (aggregate) {
          // One slot per column: first non-null value in the sub-window
          for (let c = 0; c < cols.length; c++) {
            let firstVal: string | null = null;
            for (const r of windowRows) {
              if (r[c] !== null) {
                firstVal = r[c]!;
                break;
              }
            }
            windowVals.push(firstVal);
          }
        } else {
          // Expand all rows × cols within this sub-window
          for (const r of windowRows) {
            for (let c = 0; c < cols.length; c++) {
              windowVals.push(r[c]!);
            }
          }
        }
        allWindows.push(windowVals);
      }
    }

    // ── Step 2: Group sub-windows by interval into output sequences ────────
    // R: .window_grp = .window %/% interval  →  every `interval` windows = one row
    // Default (no interval specified): all windows in one sequence (R's default of
    // ncol*nrow effectively puts all windows in one group for typical datasets).
    const effectiveInterval = options?.interval ?? allWindows.length;
    const safeInterval = Math.max(1, effectiveInterval);

    if (allWindows.length === 0) {
      // Empty group: preserve structure with one empty sequence
      result.push([]);
    } else {
      const nGroups = Math.ceil(allWindows.length / safeInterval);
      for (let g = 0; g < nGroups; g++) {
        const groupWindows = allWindows.slice(g * safeInterval, (g + 1) * safeInterval);
        const rowValues: (string | null)[] = [];
        for (const w of groupWindows) {
          for (const v of w) rowValues.push(v);
        }
        result.push(rowValues);
      }
    }
  }

  // R: window_size attr = window_size^(!aggregate) → ws when not aggregate, 1 when aggregate
  const wsAttr = aggregate ? 1 : windowSize;

  return {
    sequences: result,
    windowSize: wsAttr,
    windowSpan: cols.length,
  };
}
