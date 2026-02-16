# tnaj

**Transition Network Analysis for JavaScript/TypeScript**

A zero-dependency, pure TypeScript implementation of Transition Network Analysis (TNA) for modeling sequential data as weighted directed networks. Works in Node.js, Deno, Bun, and browsers.

Ported from the [R TNA package](https://cran.r-project.org/package=TNA) and the [Python tna package](https://github.com/mohsaqr/tnapy), with numerical equivalence to R TNA validated to machine epsilon (~1e-15).

**Live demo:** [https://saqr.me/tna-js/](https://saqr.me/tna-js/)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Model Building](#model-building)
  - [Centralities](#centralities)
  - [Pruning](#pruning)
  - [Community Detection](#community-detection)
  - [Clique Detection](#clique-detection)
  - [Sequence Clustering](#sequence-clustering)
  - [Sequence Comparison](#sequence-comparison)
  - [Group Models](#group-models)
  - [Data Preparation](#data-preparation)
  - [Utility Functions](#utility-functions)
- [Using in a Website](#using-in-a-website)
  - [With a Bundler (Vite, Webpack, etc.)](#with-a-bundler)
  - [From a CDN (ESM)](#from-a-cdn)
  - [Server-Side (Node.js / Bun / Deno)](#server-side)
- [Data Format](#data-format)
- [Types Reference](#types-reference)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
# npm
npm install tnaj

# yarn
yarn add tnaj

# pnpm
pnpm add tnaj

# bun
bun add tnaj
```

Or install from GitHub:

```bash
npm install github:mohsaqr/tna-js
```

---

## Quick Start

```typescript
import { tna, centralities, prune, communities } from 'tnaj';

// Your sequential data: each row is a sequence of states
const data = [
  ['A', 'B', 'C', 'A', 'B'],
  ['B', 'C', 'A', 'C', 'B'],
  ['A', 'A', 'B', 'C', 'A'],
  ['C', 'B', 'A', 'B', 'C'],
];

// Build a TNA model (relative transition probabilities)
const model = tna(data);

console.log(model.labels);    // ['A', 'B', 'C']
console.log(model.type);      // 'relative'
console.log(model.inits);     // Float64Array [0.5, 0.25, 0.25]

// Compute centrality measures
const cent = centralities(model);
console.log(cent.measures.OutStrength);   // Float64Array [...]
console.log(cent.measures.Betweenness);   // Float64Array [...]

// Prune weak edges (remove edges below threshold)
const pruned = prune(model, 0.1);

// Detect communities
const comm = communities(model, { methods: 'louvain' });
console.log(comm.assignments.louvain);    // [0, 0, 1]
console.log(comm.counts.louvain);         // 2
```

---

## Core Concepts

**TNA** (Transition Network Analysis) models sequential data as a weighted directed graph. Each unique state in the data becomes a node, and transitions between consecutive states become weighted directed edges.

**Model types:**

| Function | Type | Description |
|----------|------|-------------|
| `tna()` | `relative` | Row-normalized transition probabilities (values sum to 1 per row) |
| `ftna()` | `frequency` | Raw transition counts |
| `ctna()` | `co-occurrence` | Bidirectional co-occurrence counts |
| `atna()` | `attention` | Exponential decay weighted transitions (recent transitions weighted more) |

**What you get:**
- A `TNA` object with a weight matrix, initial state probabilities, and state labels
- 9 centrality measures per node (OutStrength, InStrength, Betweenness, Closeness, etc.)
- Network pruning, community detection, clique detection
- Sequence clustering (PAM, hierarchical) with 4 distance metrics
- Pattern comparison across groups with permutation testing

---

## API Reference

### Model Building

#### `tna(data, options?)`

Build a relative transition probability model.

```typescript
import { tna } from 'tnaj';

const data = [
  ['read', 'write', 'discuss', 'read'],
  ['write', 'discuss', 'read', 'write'],
];

const model = tna(data);
```

**Parameters:**
- `data` — `SequenceData | TNAData | number[][]` — Input data (see [Data Format](#data-format))
- `options.scaling` — `string | string[] | null` — Scaling to apply: `'minmax'`, `'max'`, `'rank'`, or array of these
- `options.labels` — `string[]` — Override auto-detected state labels
- `options.beginState` — `string` — Add artificial begin state
- `options.endState` — `string` — Add artificial end state

**Returns:** `TNA` object

#### `ftna(data, options?)`

Build a frequency (raw count) model. Same parameters as `tna()`.

```typescript
import { ftna } from 'tnaj';

const model = ftna(data);
// model.weights contains raw transition counts
```

#### `ctna(data, options?)`

Build a co-occurrence model. Same parameters as `tna()`.

```typescript
import { ctna } from 'tnaj';

const model = ctna(data);
// model.weights contains bidirectional co-occurrence counts
```

#### `atna(data, options?)`

Build an attention-weighted model with exponential decay.

```typescript
import { atna } from 'tnaj';

const model = atna(data, { beta: 0.1 });
// Recent transitions receive higher weights
```

**Additional parameter:**
- `options.beta` — `number` — Decay parameter (default: `0.1`). Higher values = stronger recency weighting.

#### `buildModel(data, options?)`

Generic model builder. Use this for advanced model types.

```typescript
import { buildModel } from 'tnaj';

// Build from a pre-computed weight matrix
const model = buildModel([
  [0, 0.3, 0.7],
  [0.5, 0, 0.5],
  [0.6, 0.4, 0],
], {
  type: 'matrix',
  labels: ['A', 'B', 'C'],
});
```

**All model types:** `'relative'`, `'frequency'`, `'co-occurrence'`, `'reverse'`, `'n-gram'`, `'gap'`, `'window'`, `'attention'`, `'matrix'`

#### `summary(model)`

Get a summary of model properties.

```typescript
import { tna, summary } from 'tnaj';

const s = summary(tna(data));
// {
//   nStates: 3,
//   type: 'relative',
//   nEdges: 6,
//   density: 0.667,
//   meanWeight: 0.333,
//   maxWeight: 0.5,
//   hasSelfLoops: false,
// }
```

---

### Centralities

#### `centralities(model, options?)`

Compute 9 centrality measures for each state in the network.

```typescript
import { tna, centralities, AVAILABLE_MEASURES } from 'tnaj';

const model = tna(data);
const cent = centralities(model);

// Access individual measures
cent.measures.OutStrength;      // Float64Array — weighted out-degree
cent.measures.InStrength;       // Float64Array — weighted in-degree
cent.measures.Betweenness;      // Float64Array — shortest-path betweenness
cent.measures.BetweennessRSP;   // Float64Array — randomized shortest-path betweenness
cent.measures.ClosenessIn;      // Float64Array — in-closeness
cent.measures.ClosenessOut;     // Float64Array — out-closeness
cent.measures.Closeness;        // Float64Array — undirected closeness
cent.measures.Diffusion;        // Float64Array — diffusion centrality
cent.measures.Clustering;       // Float64Array — weighted clustering coefficient

// Labels are aligned with measure arrays
cent.labels; // ['A', 'B', 'C']

// List all available measures
console.log(AVAILABLE_MEASURES);
```

**Parameters:**
- `model` — `TNA | GroupTNA`
- `options.loops` — `boolean` — Include self-loops (default: `true`)
- `options.normalize` — `boolean` — Normalize measures (default: `false`)
- `options.measures` — `CentralityMeasure[]` — Subset of measures to compute

**Returns:** `CentralityResult`

#### `betweennessNetwork(model)`

Replace edge weights with edge betweenness centrality values.

```typescript
import { tna, betweennessNetwork } from 'tnaj';

const bModel = betweennessNetwork(tna(data));
// bModel.weights now contains edge betweenness values
```

---

### Pruning

#### `prune(model, threshold?)`

Remove edges below a weight threshold.

```typescript
import { tna, prune } from 'tnaj';

const model = tna(data);
const pruned = prune(model, 0.1);
// All edges with weight < 0.1 are set to 0
```

**Parameters:**
- `model` — `TNA | GroupTNA`
- `threshold` — `number` — Weight threshold (default: `0.1`)

**Returns:** `TNA` (or `Record<string, TNA>` for GroupTNA)

---

### Community Detection

#### `communities(model, options?)`

Detect communities (clusters of densely connected nodes).

```typescript
import { tna, communities, AVAILABLE_METHODS } from 'tnaj';

const model = tna(data);

// Single method
const comm = communities(model, { methods: 'louvain' });
console.log(comm.assignments.louvain);  // [0, 1, 0] — community per node
console.log(comm.counts.louvain);       // 2 — number of communities

// Multiple methods at once
const commAll = communities(model, {
  methods: ['louvain', 'fast_greedy', 'walktrap'],
});

// Available methods
console.log(AVAILABLE_METHODS);
// ['fast_greedy', 'louvain', 'label_prop', 'leading_eigen', 'edge_betweenness', 'walktrap']
```

**Parameters:**
- `model` — `TNA | GroupTNA`
- `options.methods` — `CommunityMethod | CommunityMethod[]` — Detection method(s)

**Returns:** `CommunityResult`

---

### Clique Detection

#### `cliques(model, options?)`

Find fully connected subgraphs (cliques) above a weight threshold.

```typescript
import { tna, cliques } from 'tnaj';

const model = tna(data);
const cl = cliques(model, { threshold: 0.05 });
console.log(cl.labels);   // [['A', 'B'], ['B', 'C']] — labels per clique
console.log(cl.size);     // 2 — minimum clique size
```

**Parameters:**
- `model` — `TNA | GroupTNA`
- `options.threshold` — `number` — Minimum edge weight (default: `0.05`)

**Returns:** `CliqueResult`

---

### Sequence Clustering

#### `clusterSequences(data, k, options?)`

Cluster sequences based on pairwise distances.

```typescript
import { clusterSequences } from 'tnaj';

const data = [
  ['A', 'B', 'C'],
  ['A', 'B', 'C'],
  ['C', 'B', 'A'],
  ['C', 'B', 'A'],
];

const result = clusterSequences(data, 2, {
  dissimilarity: 'hamming',
  method: 'pam',
});

console.log(result.assignments);  // [1, 1, 2, 2]
console.log(result.silhouette);   // 0.75 — clustering quality (-1 to 1)
console.log(result.sizes);        // [2, 2]
```

**Parameters:**
- `data` — `SequenceData | TNAData`
- `k` — `number` — Number of clusters (must be >= 2)
- `options.dissimilarity` — `'hamming' | 'lv' | 'osa' | 'lcs'` — Distance metric (default: `'hamming'`)
  - `'hamming'` — Positional mismatches
  - `'lv'` — Levenshtein (edit distance: insert, delete, substitute)
  - `'osa'` — Optimal string alignment (edit distance + transpositions)
  - `'lcs'` — Longest common subsequence distance
- `options.method` — `'pam' | string` — Clustering algorithm (default: `'pam'`)
  - `'pam'` — Partitioning Around Medoids (like k-medoids)
  - Any other string — Hierarchical clustering (average linkage)
- `options.weighted` — `boolean` — Position-weighted Hamming (default: `false`)
- `options.lambda` — `number` — Exponential decay for weighted Hamming (default: `1`)

**Returns:** `ClusterResult`

---

### Sequence Comparison

#### `compareSequences(data, groups, options?)`

Compare subsequence patterns between groups (e.g., high vs. low performers).

```typescript
import { compareSequences } from 'tnaj';

const data = [
  ['A', 'B', 'C'],
  ['A', 'C', 'B'],
  ['C', 'B', 'A'],
  ['C', 'A', 'B'],
];
const groups = ['high', 'high', 'low', 'low'];

const result = compareSequences(data, groups, {
  lengths: [1, 2],
  minFreq: 0,
});

// Each row: { pattern, frequencies, proportions, effectSize?, pValue? }
for (const row of result) {
  console.log(row.pattern, row.frequencies, row.proportions);
}
```

**Parameters:**
- `data` — `SequenceData`
- `groups` — `string[]` — Group label per sequence
- `options.lengths` — `number[]` — Subsequence lengths to compare (default: `[1, 2, 3]`)
- `options.minFreq` — `number` — Minimum frequency threshold (default: `0`)
- `options.test` — `boolean` — Run permutation test (default: `false`)
- `options.iter` — `number` — Number of permutation iterations (default: `999`)
- `options.adjust` — `string` — P-value adjustment method: `'bonferroni'`, `'holm'`, `'bh'` (default: `'bonferroni'`)
- `options.seed` — `number` — Random seed for reproducibility

**Returns:** `CompareRow[]`

---

### Group Models

Build and analyze TNA models for multiple groups simultaneously.

```typescript
import { groupTna, centralities, communities, prune, groupNames, groupEntries } from 'tnaj';

// Data split by group
const groups = {
  high: [
    ['A', 'B', 'C'],
    ['A', 'C', 'B'],
  ],
  low: [
    ['C', 'B', 'A'],
    ['C', 'A', 'B'],
  ],
};

// Build group models
const gModel = groupTna(groups);

// List group names
console.log(groupNames(gModel)); // ['high', 'low']

// Iterate over groups
for (const [name, model] of groupEntries(gModel)) {
  console.log(name, model.labels);
}

// All analysis functions accept GroupTNA directly
const cent = centralities(gModel);       // Combined centralities with group labels
const pruned = prune(gModel, 0.1);       // { high: TNA, low: TNA }
const comm = communities(gModel);        // { high: CommunityResult, low: CommunityResult }
```

#### Group builder functions

| Function | Description |
|----------|-------------|
| `groupTna(groups)` | Relative transition probabilities per group |
| `groupFtna(groups)` | Frequency counts per group |
| `groupCtna(groups)` | Co-occurrence per group |
| `groupAtna(groups)` | Attention-weighted per group |
| `createGroupTNA(models)` | Create from pre-built TNA objects |

---

### Data Preparation

#### `prepareData(data, options?)`

Prepare raw sequence data into a `TNAData` container with summary statistics.

```typescript
import { prepareData } from 'tnaj';

const raw = [
  ['read', 'write', 'discuss'],
  ['write', 'read', null, 'discuss'],
];

const prepared = prepareData(raw);
console.log(prepared.labels);           // ['discuss', 'read', 'write']
console.log(prepared.statistics);
// {
//   nSessions: 2,
//   nUniqueActions: 3,
//   uniqueActions: ['discuss', 'read', 'write'],
//   maxSequenceLength: 4,
//   meanSequenceLength: 3.5,
// }
```

#### `createSeqdata(data, options?)`

Lower-level function to clean sequence data and detect labels.

```typescript
import { createSeqdata } from 'tnaj';

const { data, labels } = createSeqdata(rawData, {
  beginState: 'START',
  endState: 'END',
});
```

#### `importOnehot(data, window, step?)`

Import windowed one-hot encoded data for TNA analysis.

```typescript
import { importOnehot } from 'tnaj';

const onehot = importOnehot(rawData, 5, 1);
```

---

### Utility Functions

#### Matrix operations

```typescript
import { Matrix, rowNormalize, minmaxScale, maxScale, rankScale } from 'tnaj';

// Create matrices
const m = Matrix.zeros(3, 3);
const m2 = Matrix.from2D([[1, 2], [3, 4]]);

// Access
m.get(0, 1);
m.set(0, 1, 0.5);
m.rows;  // number of rows
m.cols;  // number of columns

// Operations
const norm = rowNormalize(m2);     // Row-normalize
const mm = minmaxScale(m2);        // Min-max scale
const mx = maxScale(m2);           // Max scale
const rk = rankScale(m2);          // Rank scale
```

#### Color utilities

```typescript
import {
  colorPalette, DEFAULT_COLORS, createColorMap,
  hexToRgb, rgbToHex, lightenColor, darkenColor,
} from 'tnaj';

const colors = colorPalette(5);                       // Get 5 distinct colors
const map = createColorMap(['A', 'B', 'C']);           // { A: '#...', B: '#...', C: '#...' }
const [r, g, b] = hexToRgb('#4e79a7');                // [78, 121, 167]
const lighter = lightenColor('#4e79a7', 0.3);         // Lighten by 30%
```

#### Statistical utilities

```typescript
import { arrayMean, arrayStd, pearsonCorr, arrayQuantile } from 'tnaj';

const arr = new Float64Array([1, 2, 3, 4, 5]);
arrayMean(arr);              // 3
arrayStd(arr);               // ~1.58 (sample std, ddof=1)
arrayQuantile(arr, 0.5);     // 3 (median)

const a = new Float64Array([1, 2, 3]);
const b = new Float64Array([2, 4, 6]);
pearsonCorr(a, b);           // 1.0
```

#### Seeded RNG

```typescript
import { SeededRNG } from 'tnaj';

const rng = new SeededRNG(42);
rng.random();     // Deterministic float in [0, 1)
rng.randint(10);  // Deterministic integer in [0, 10)
```

---

## Using in a Website

### With a Bundler

The recommended approach for production websites. Works with Vite, Webpack, Rollup, esbuild, Parcel, etc.

**1. Install:**

```bash
npm install tnaj
```

**2. Import and use in your code:**

```typescript
// app.ts or app.js
import { tna, centralities, prune, communities, summary } from 'tnaj';

const data = [
  ['login', 'browse', 'search', 'purchase'],
  ['login', 'search', 'browse', 'logout'],
  ['login', 'browse', 'browse', 'purchase'],
];

const model = tna(data);
const cent = centralities(model);

// Render results in the DOM
const el = document.getElementById('results')!;
el.innerHTML = model.labels.map((label, i) => {
  return `<div>${label}: OutStrength=${cent.measures.OutStrength[i]!.toFixed(3)}</div>`;
}).join('');
```

**3. Vite example config:**

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
export default defineConfig({
  // tnaj is a standard ESM/CJS package — no special config needed
});
```

**4. Webpack example config:**

```javascript
// webpack.config.js
module.exports = {
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      { test: /\.ts$/, use: 'ts-loader', exclude: /node_modules/ },
    ],
  },
};
```

### From a CDN

For quick prototyping or simple pages without a build step. Use [esm.sh](https://esm.sh), [jsDelivr](https://www.jsdelivr.com/), or [unpkg](https://unpkg.com/).

```html
<!DOCTYPE html>
<html>
<head>
  <title>TNA Demo</title>
</head>
<body>
  <div id="output"></div>

  <script type="module">
    // Import from CDN
    import { tna, centralities, prune, summary } from 'https://esm.sh/tnaj';

    const data = [
      ['A', 'B', 'C', 'A'],
      ['B', 'C', 'A', 'B'],
      ['A', 'A', 'B', 'C'],
    ];

    const model = tna(data);
    const cent = centralities(model);
    const s = summary(model);

    document.getElementById('output').innerHTML = `
      <h2>TNA Model (${s.type})</h2>
      <p>States: ${model.labels.join(', ')}</p>
      <p>Edges: ${s.nEdges}, Density: ${Number(s.density).toFixed(3)}</p>
      <h3>OutStrength</h3>
      <ul>
        ${model.labels.map((l, i) =>
          `<li>${l}: ${cent.measures.OutStrength[i].toFixed(4)}</li>`
        ).join('')}
      </ul>
    `;
  </script>
</body>
</html>
```

### Server-Side

**Node.js (ESM):**

```typescript
import { tna, centralities } from 'tnaj';

const data = [['A', 'B', 'C'], ['B', 'C', 'A']];
const model = tna(data);
const cent = centralities(model);
console.log(cent.measures.OutStrength);
```

**Node.js (CommonJS):**

```javascript
const { tna, centralities } = require('tnaj');

const data = [['A', 'B', 'C'], ['B', 'C', 'A']];
const model = tna(data);
const cent = centralities(model);
console.log(cent.measures.OutStrength);
```

**Deno:**

```typescript
import { tna, centralities } from 'npm:tnaj';

const data = [['A', 'B', 'C'], ['B', 'C', 'A']];
const model = tna(data);
console.log(centralities(model));
```

**Bun:**

```typescript
import { tna, centralities } from 'tnaj';
// Same API as Node.js ESM
```

---

## Data Format

### Sequence Data (primary input)

A 2D array where each row is a sequence of state labels (strings). Sequences can have different lengths. Use `null` for missing values.

```typescript
const data: SequenceData = [
  ['read', 'write', 'discuss', 'read'],
  ['write', 'discuss', null, 'read'],    // null = missing
  ['read', 'read', 'write'],             // shorter sequence is fine
];
```

### Direct Weight Matrix

A square 2D number array where `matrix[i][j]` is the weight from state `i` to state `j`.

```typescript
const matrix = [
  [0.0, 0.3, 0.7],
  [0.5, 0.0, 0.5],
  [0.6, 0.4, 0.0],
];

const model = tna(matrix, { labels: ['A', 'B', 'C'] });
```

### Grouped Data

An object mapping group names to sequence arrays.

```typescript
const groups = {
  experimental: [['A', 'B', 'C'], ['A', 'C', 'B']],
  control:      [['C', 'B', 'A'], ['C', 'A', 'B']],
};

const gModel = groupTna(groups);
```

---

## Types Reference

```typescript
/** A sequence: array of string states (null = missing). */
type Sequence = (string | null)[];

/** Sequence dataset: array of sequences. */
type SequenceData = Sequence[];

/** TNA model object. */
interface TNA {
  weights: Matrix;          // n_states x n_states adjacency matrix
  inits: Float64Array;      // Initial state probabilities
  labels: string[];         // State labels
  data: SequenceData | null;
  type: ModelType;
  scaling: string[];
}

/** Group of TNA models. */
interface GroupTNA {
  models: Record<string, TNA>;
}

/** Centrality result. */
interface CentralityResult {
  labels: string[];
  measures: Record<CentralityMeasure, Float64Array>;
  groups?: string[];       // Present for GroupTNA input
}

/** Centrality measures. */
type CentralityMeasure =
  | 'OutStrength' | 'InStrength'
  | 'ClosenessIn' | 'ClosenessOut' | 'Closeness'
  | 'Betweenness' | 'BetweennessRSP'
  | 'Diffusion' | 'Clustering';

/** Community detection result. */
interface CommunityResult {
  counts: Record<string, number>;       // method → number of communities
  assignments: Record<string, number[]>; // method → community per node
  labels: string[];
}

/** Community detection methods. */
type CommunityMethod =
  | 'fast_greedy' | 'louvain' | 'label_prop'
  | 'leading_eigen' | 'edge_betweenness' | 'walktrap';

/** Clique detection result. */
interface CliqueResult {
  weights: Matrix[];    // Weight sub-matrices per clique
  indices: number[][];  // Node indices per clique
  labels: string[][];   // Node labels per clique
  size: number;
  threshold: number;
}

/** Cluster result. */
interface ClusterResult {
  data: SequenceData;
  k: number;
  assignments: number[];    // 1-indexed cluster labels
  silhouette: number;       // Silhouette score (-1 to 1)
  sizes: number[];
  method: string;
  distance: Matrix;
  dissimilarity: string;
}

/** Sequence comparison result row. */
interface CompareRow {
  pattern: string;
  frequencies: Record<string, number>;
  proportions: Record<string, number>;
  effectSize?: number;
  pValue?: number;
}
```

---

## Full Example: Analyzing Student Learning Behavior

```typescript
import {
  tna, ftna, centralities, prune, communities, clusterSequences,
  summary, AVAILABLE_MEASURES,
} from 'tnaj';

// Student learning activity sequences
const sequences = [
  ['read', 'annotate', 'discuss', 'quiz', 'read'],
  ['read', 'quiz', 'read', 'annotate', 'discuss'],
  ['discuss', 'read', 'annotate', 'read', 'quiz'],
  ['quiz', 'read', 'discuss', 'annotate', 'quiz'],
  ['read', 'read', 'annotate', 'discuss', 'quiz'],
];

// 1. Build model
const model = tna(sequences);
const s = summary(model);
console.log(`States: ${s.nStates}, Edges: ${s.nEdges}, Density: ${s.density}`);

// 2. Centralities — find most important states
const cent = centralities(model);
model.labels.forEach((label, i) => {
  console.log(`${label}: OutStrength=${cent.measures.OutStrength[i]!.toFixed(3)}, ` +
    `Betweenness=${cent.measures.Betweenness[i]!.toFixed(3)}`);
});

// 3. Prune weak connections
const pruned = prune(model, 0.15);
const ps = summary(pruned);
console.log(`After pruning: ${ps.nEdges} edges remain`);

// 4. Communities — find clusters of related activities
const comm = communities(model, { methods: 'louvain' });
model.labels.forEach((label, i) => {
  console.log(`${label} → Community ${comm.assignments.louvain![i]}`);
});

// 5. Cluster students by behavior similarity
const clusters = clusterSequences(sequences, 2, {
  dissimilarity: 'lv',
  method: 'pam',
});
console.log(`Silhouette: ${clusters.silhouette.toFixed(3)}`);
console.log(`Assignments: ${clusters.assignments}`);
```

---

## Citation

If you use TNA in your research, please cite:

> Saqr, M., Lopez-Pernas, S., Tormanen, T., Kaliisa, R., Misiejuk, K., & Tikka, S. (2025). Transition Network Analysis: A Novel Framework for Modeling, Visualizing, and Identifying the Temporal Patterns of Learners and Learning Processes. In *Proceedings of the 15th International Learning Analytics and Knowledge Conference (LAK '25)*, 351-361. https://doi.org/10.1145/3706468.3706513

```bibtex
@inproceedings{saqr2025tna,
  title     = {Transition Network Analysis: A Novel Framework for Modeling,
               Visualizing, and Identifying the Temporal Patterns of Learners
               and Learning Processes},
  author    = {Saqr, Mohammed and L{\'o}pez-Pernas, Sonsoles and
               T{\"o}rm{\"a}nen, Tiina and Kaliisa, Rogers and
               Misiejuk, Kamila and Tikka, Santtu},
  booktitle = {Proceedings of the 15th International Learning Analytics
               and Knowledge Conference (LAK '25)},
  pages     = {351--361},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3706468.3706513},
}
```

**Key references:**

- Saqr, M., Lopez-Pernas, S., Tormanen, T., Kaliisa, R., Misiejuk, K., & Tikka, S. (2025). Transition Network Analysis: A Novel Framework for Modeling, Visualizing, and Identifying the Temporal Patterns of Learners and Learning Processes. In *Proceedings of LAK '25*, 351-361. https://doi.org/10.1145/3706468.3706513

- Tikka, S., Lopez-Pernas, S., & Saqr, M. (2025). tna: An R Package for Transition Network Analysis. *Applied Psychological Measurement*. https://doi.org/10.1177/01466216251348840

- Saqr, M., Lopez-Pernas, S., & Tikka, S. (2025). Mapping relational dynamics with transition network analysis: A primer and tutorial. In *Advanced Learning Analytics Methods: AI, Precision and Complexity*. Springer.

- Saqr, M., Lopez-Pernas, S., & Tikka, S. (2025). Capturing the breadth and dynamics of the temporal process with frequency transition network analysis. In *Advanced Learning Analytics Methods: AI, Precision and Complexity*. Springer.

- Lopez-Pernas, S., Tikka, S., & Saqr, M. (2025). Mining patterns and clusters with transition network analysis: A heterogeneity approach. In *Advanced Learning Analytics Methods: AI, Precision and Complexity*. Springer.

---

## License

MIT
