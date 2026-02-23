/**
 * 1000 R Ground Truth One-Hot CTNA Tests
 *
 * All expected values generated from R TNA 1.2.0 using:
 *   - tna::import_onehot() for one-hot → sequence conversion
 *   - tna::ctna() for windowed co-occurrence CTNA
 *
 * Tests verify weights, inits, and labels match R exactly.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { importOnehot, ctna } from '../src/index.js';

interface Config {
  params: {
    n_actors: number;
    rows_per_actor: number;
    n_sessions: number;
    n_cols: number;
    window_size: number;
    window_type: 'tumbling' | 'sliding';
    aggregate: boolean;
    seed: number;
    cols: string[];
  };
  raw_data: Record<string, number | string>[];
  ctna_weights: number[] | number;
  ctna_inits: (number | null)[] | number | null;
  ctna_labels: string[] | string;
  n_seqs: number;
}

const groundtruthPath = resolve(__dirname, 'fixtures/onehot_ctna_groundtruth.json');
const groundtruth: Config[] = JSON.parse(readFileSync(groundtruthPath, 'utf-8'));

describe('One-hot CTNA R equivalence', () => {
  for (const cfg of groundtruth) {
    const { params: p } = cfg;
    const id = `seed${p.seed}_${p.n_cols}cols_ws${p.window_size}_${p.window_type}_agg${p.aggregate}`;

    it(id, () => {
      const data = cfg.raw_data;
      const result = importOnehot(data, p.cols, {
        actor: p.n_actors > 1 ? 'Actor' : undefined,
        session: p.n_sessions > 1 ? 'Session' : undefined,
        windowSize: p.window_size,
        windowType: p.window_type,
        aggregate: p.aggregate,
      });

      const m = ctna(result);

      // R's auto_unbox may turn single-element vectors into scalars
      const labels = Array.isArray(cfg.ctna_labels) ? cfg.ctna_labels : [cfg.ctna_labels];
      const weights = Array.isArray(cfg.ctna_weights) ? cfg.ctna_weights : [cfg.ctna_weights];
      const inits = Array.isArray(cfg.ctna_inits) ? cfg.ctna_inits : [cfg.ctna_inits];

      // Check labels
      expect(m.labels).toEqual(labels);

      // Check weights
      const n = labels.length;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const expected = weights[i * n + j]!;
          expect(m.weights.get(i, j)).toBeCloseTo(expected, 10);
        }
      }

      // Check inits
      for (let i = 0; i < n; i++) {
        const expected = inits[i];
        if (expected === null) {
          expect(m.inits[i]).toBeNaN();
        } else {
          expect(m.inits[i]).toBeCloseTo(expected, 10);
        }
      }
    });
  }
});
