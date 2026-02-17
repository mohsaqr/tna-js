/**
 * TNA.js Interactive Demo
 *
 * Builds a TNA model from sample data and renders:
 *   - Circular network graph (D3)
 *   - Horizontal centrality bar chart
 *   - Permutation test
 */
import {
  tna, ftna, ctna, atna,
  centralities, prune, communities, summary,
  AVAILABLE_MEASURES, AVAILABLE_METHODS,
} from '../src/index';
import type {
  TNA, CentralityResult, CommunityResult,
  CentralityMeasure, CommunityMethod, SequenceData,
} from '../src/index';
import * as d3 from 'd3';
import groundTruth from '../tests/fixtures/ground_truth.json';

// ═══════════════════════════════════════════════════════════
//  Sample Data
// ═══════════════════════════════════════════════════════════
const sampleData: SequenceData = groundTruth.small_data;

// ═══════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════
let modelType: 'tna' | 'ftna' | 'ctna' | 'atna' = 'tna';
let threshold = 0;
let showCommunities = false;
let communityMethod: CommunityMethod = 'louvain';
let selectedMeasure1: CentralityMeasure = 'OutStrength';
let selectedMeasure2: CentralityMeasure = 'Betweenness';

// ═══════════════════════════════════════════════════════════
//  Colors (Tableau 10)
// ═══════════════════════════════════════════════════════════
const NODE_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
  '#edc948', '#b07aa1', '#ff9da7', '#9c755f',
];
const COMMUNITY_COLORS = [
  '#4e79a7', '#e15759', '#59a14f', '#edc948',
  '#b07aa1', '#76b7b2', '#f28e2b', '#ff9da7',
];

// ═══════════════════════════════════════════════════════════
//  DOM References
// ═══════════════════════════════════════════════════════════
const modelSelect = document.getElementById('model-type') as HTMLSelectElement;
const pruneSlider = document.getElementById('prune-threshold') as HTMLInputElement;
const pruneValue = document.getElementById('prune-value') as HTMLSpanElement;
const communityCheck = document.getElementById('community-toggle') as HTMLInputElement;
const methodSelect = document.getElementById('community-method') as HTMLSelectElement;
const measureSelect1 = document.getElementById('centrality-measure-1') as HTMLSelectElement;
const measureSelect2 = document.getElementById('centrality-measure-2') as HTMLSelectElement;
const summaryEl = document.getElementById('model-summary') as HTMLDivElement;
const tooltip = document.getElementById('tooltip') as HTMLDivElement;

// Populate community method options
AVAILABLE_METHODS.forEach(m => {
  const opt = document.createElement('option');
  opt.value = m;
  opt.textContent = m.replace(/_/g, ' ');
  methodSelect.appendChild(opt);
});
methodSelect.value = 'louvain';

// Populate centrality measure options (both selects)
for (const sel of [measureSelect1, measureSelect2]) {
  AVAILABLE_MEASURES.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  });
}
measureSelect1.value = 'OutStrength';
measureSelect2.value = 'Betweenness';

// ═══════════════════════════════════════════════════════════
//  Tooltip helpers
// ═══════════════════════════════════════════════════════════
function showTooltip(event: MouseEvent, html: string) {
  tooltip.innerHTML = html;
  tooltip.style.opacity = '1';
  tooltip.style.left = event.clientX + 12 + 'px';
  tooltip.style.top = event.clientY - 10 + 'px';
}

function hideTooltip() {
  tooltip.style.opacity = '0';
}

// ═══════════════════════════════════════════════════════════
//  Model Building
// ═══════════════════════════════════════════════════════════
function buildCurrentModel(): TNA {
  const builders = { tna, ftna, ctna, atna };
  let model = builders[modelType](sampleData);
  if (threshold > 0) {
    model = prune(model, threshold) as TNA;
  }
  return model;
}

// ═══════════════════════════════════════════════════════════
//  Network Graph (Circular Layout + Manual Arrowheads)
// ═══════════════════════════════════════════════════════════
interface NodeDatum {
  id: string;
  idx: number;
  color: string;
  x: number;
  y: number;
}

interface EdgeDatum {
  fromIdx: number;
  toIdx: number;
  weight: number;
}

/** Format edge weight: integers shown without decimals, others as .XX */
function fmtWeight(w: number): string {
  if (Number.isInteger(w)) return String(w);
  return w.toFixed(2).replace(/^0\./, '.');
}

const graphContainer = document.getElementById('network-graph')!;
const graphWidth = 640;
const graphHeight = 420;
const NODE_R = 22;

const graphSvg = d3.select(graphContainer)
  .append('svg')
  .attr('viewBox', `0 0 ${graphWidth} ${graphHeight}`)
  .attr('width', '100%')
  .attr('height', '100%')
  .style('min-height', '380px');

const edgeGroup = graphSvg.append('g').attr('class', 'edges');
const arrowGroup = graphSvg.append('g').attr('class', 'arrows');
const edgeLabelGroup = graphSvg.append('g').attr('class', 'edge-labels');
const nodeGroup = graphSvg.append('g').attr('class', 'nodes');

let nodes: NodeDatum[] = [];

function initLayout(labels: string[]) {
  const cx = graphWidth / 2;
  const cy = graphHeight / 2;
  const radius = Math.min(cx, cy) - NODE_R - 30;

  nodes = labels.map((id, i) => {
    const angle = (2 * Math.PI * i) / labels.length - Math.PI / 2;
    return {
      id, idx: i,
      color: NODE_COLORS[i % NODE_COLORS.length]!,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });
}

/** Compute a quadratic bezier edge from source to target, curving left of the direction. */
function computeEdgePath(
  sx: number, sy: number, tx: number, ty: number, curvature: number,
): { path: string; tipX: number; tipY: number; tipDx: number; tipDy: number; labelX: number; labelY: number } {
  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return { path: '', tipX: tx, tipY: ty, tipDx: 0, tipDy: -1, labelX: (sx + tx) / 2, labelY: (sy + ty) / 2 };

  const ux = dx / len;
  const uy = dy / len;
  // Perpendicular (left of direction)
  const px = -uy;
  const py = ux;

  // Control point: midpoint offset perpendicular
  const mx = (sx + tx) / 2 + px * curvature;
  const my = (sy + ty) / 2 + py * curvature;

  // Start point: on source boundary toward control point
  const sdx = mx - sx;
  const sdy = my - sy;
  const slen = Math.sqrt(sdx * sdx + sdy * sdy);
  const startX = sx + (sdx / slen) * NODE_R;
  const startY = sy + (sdy / slen) * NODE_R;

  // End point direction: from control point toward target
  const edx = tx - mx;
  const edy = ty - my;
  const elen = Math.sqrt(edx * edx + edy * edy);
  const eux = edx / elen;
  const euy = edy / elen;

  // Arrow tip sits on node boundary
  const tipX = tx - eux * NODE_R;
  const tipY = ty - euy * NODE_R;
  // End the path a bit before the tip to leave room for arrowhead
  const endX = tx - eux * (NODE_R + 8);
  const endY = ty - euy * (NODE_R + 8);

  // Label at t=0.55 on the quadratic bezier (slightly toward target)
  const t = 0.55;
  const labelX = (1 - t) * (1 - t) * startX + 2 * (1 - t) * t * mx + t * t * endX;
  const labelY = (1 - t) * (1 - t) * startY + 2 * (1 - t) * t * my + t * t * endY;

  return {
    path: `M${startX},${startY} Q${mx},${my} ${endX},${endY}`,
    tipX, tipY, tipDx: eux, tipDy: euy, labelX, labelY,
  };
}

/** Build a triangle arrowhead polygon string. */
function arrowPoly(tipX: number, tipY: number, dx: number, dy: number): string {
  const len = 7;
  const halfW = 3.5;
  const baseX = tipX - dx * len;
  const baseY = tipY - dy * len;
  const lx = baseX - dy * halfW;
  const ly = baseY + dx * halfW;
  const rx = baseX + dy * halfW;
  const ry = baseY - dx * halfW;
  return `${tipX},${tipY} ${lx},${ly} ${rx},${ry}`;
}

function renderNetwork(model: TNA, comm?: CommunityResult) {
  const n = model.labels.length;
  const weights = model.weights;

  // Build edges (skip self-loops and zero-weight)
  const edges: EdgeDatum[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const w = weights.get(i, j);
      if (w > 0 && w >= 0.05) edges.push({ fromIdx: i, toIdx: j, weight: w });
    }
  }

  // Check which pairs are bidirectional (for curvature)
  const bidir = new Set<string>();
  for (const e of edges) {
    const rev = edges.find(r => r.fromIdx === e.toIdx && r.toIdx === e.fromIdx);
    if (rev) {
      bidir.add(`${e.fromIdx}-${e.toIdx}`);
    }
  }

  // Update node colors
  if (comm) {
    const methodKey = Object.keys(comm.assignments)[0]!;
    const assign = comm.assignments[methodKey]!;
    nodes.forEach((nd, i) => {
      nd.color = COMMUNITY_COLORS[assign[i]! % COMMUNITY_COLORS.length]!;
    });
  } else {
    nodes.forEach((nd, i) => {
      nd.color = NODE_COLORS[i % NODE_COLORS.length]!;
    });
  }

  const maxW = Math.max(...edges.map(e => e.weight), 1e-6);
  const widthScale = d3.scaleLinear().domain([0, maxW]).range([0.3, 4]);
  const opacityScale = d3.scaleLinear().domain([0, maxW]).range([0.7, 1.0]);

  const EDGE_COLOR = '#2B4C7E';
  const ARROW_COLOR = '#2B4C7E';

  // ── Edges ──
  edgeGroup.selectAll('*').remove();
  arrowGroup.selectAll('*').remove();
  edgeLabelGroup.selectAll('*').remove();

  for (const e of edges) {
    const src = nodes[e.fromIdx]!;
    const tgt = nodes[e.toIdx]!;
    const isBidir = bidir.has(`${e.fromIdx}-${e.toIdx}`);
    const curvature = isBidir ? 22 : 0;
    const { path, tipX, tipY, tipDx, tipDy, labelX, labelY } = computeEdgePath(
      src.x, src.y, tgt.x, tgt.y, curvature,
    );

    if (!path) continue;

    const op = opacityScale(e.weight);

    edgeGroup.append('path')
      .attr('d', path)
      .attr('fill', 'none')
      .attr('stroke', EDGE_COLOR)
      .attr('stroke-width', widthScale(e.weight))
      .attr('stroke-opacity', op)
      .attr('stroke-linecap', 'round')
      .on('mouseover', function (event: MouseEvent) {
        d3.select(this).attr('stroke', '#e15759').attr('stroke-opacity', 0.85);
        showTooltip(event, `<b>${src.id} → ${tgt.id}</b><br>Weight: ${e.weight.toFixed(4)}`);
      })
      .on('mousemove', function (event: MouseEvent) {
        tooltip.style.left = event.clientX + 12 + 'px';
        tooltip.style.top = event.clientY - 10 + 'px';
      })
      .on('mouseout', function () {
        d3.select(this).attr('stroke', EDGE_COLOR).attr('stroke-opacity', op);
        hideTooltip();
      });

    arrowGroup.append('polygon')
      .attr('points', arrowPoly(tipX, tipY, tipDx, tipDy))
      .attr('fill', ARROW_COLOR)
      .attr('opacity', op + 0.15);

    // Edge weight label
    edgeLabelGroup.append('text')
      .attr('x', labelX)
      .attr('y', labelY)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('font-size', '9px')
      .attr('fill', '#2B4C7E')
      .attr('pointer-events', 'none')
      .style('paint-order', 'stroke')
      .style('stroke', '#ffffff')
      .style('stroke-width', '3px')
      .style('stroke-linejoin', 'round')
      .text(fmtWeight(e.weight));
  }

  // ── Nodes ──
  const nodeSel = nodeGroup.selectAll<SVGGElement, NodeDatum>('g.node')
    .data(nodes, d => d.id);

  // Enter
  const nodeEnter = nodeSel.enter().append('g').attr('class', 'node');

  nodeEnter.append('circle')
    .attr('r', NODE_R)
    .attr('stroke', '#999999')
    .attr('stroke-width', 2);

  nodeEnter.append('text')
    .attr('class', 'node-label')
    .attr('dy', '0.35em');

  nodeEnter
    .on('mouseover', function (event: MouseEvent, d: NodeDatum) {
      d3.select(this).select('circle').attr('stroke', '#333').attr('stroke-width', 3);
      showTooltip(event, `<b>${d.id}</b><br>Init prob: ${model.inits[d.idx]!.toFixed(4)}`);
    })
    .on('mousemove', function (event: MouseEvent) {
      tooltip.style.left = event.clientX + 12 + 'px';
      tooltip.style.top = event.clientY - 10 + 'px';
    })
    .on('mouseout', function () {
      d3.select(this).select('circle').attr('stroke', '#999999').attr('stroke-width', 2);
      hideTooltip();
    });

  // Update
  const nodeUpdate = nodeEnter.merge(nodeSel);
  nodeUpdate
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .select('circle').attr('fill', d => d.color);
  nodeUpdate.select('text').text(d => d.id);
}

// ═══════════════════════════════════════════════════════════
//  Centrality Bar Charts (reusable for both panels)
// ═══════════════════════════════════════════════════════════
const chartContainer1 = document.getElementById('centrality-chart-1')!;
const chartContainer2 = document.getElementById('centrality-chart-2')!;

function renderCentralityChart(
  container: HTMLElement,
  cent: CentralityResult,
  measure: CentralityMeasure,
) {
  const values = cent.measures[measure];
  if (!values) return;

  const data = cent.labels.map((label, i) => ({
    label,
    value: values[i]!,
    color: NODE_COLORS[i % NODE_COLORS.length]!,
  })).sort((a, b) => b.value - a.value);

  const maxVal = Math.max(...data.map(d => d.value), 1e-6);

  const rect = container.getBoundingClientRect();
  const width = Math.max(rect.width, 300);
  const height = Math.max(rect.height, 280);
  const margin = { top: 10, right: 50, bottom: 10, left: 85 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  d3.select(container).selectAll('*').remove();

  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const y = d3.scaleBand()
    .domain(data.map(d => d.label))
    .range([0, innerH])
    .padding(0.25);

  const x = d3.scaleLinear()
    .domain([0, maxVal * 1.1])
    .range([0, innerW]);

  g.selectAll('rect')
    .data(data)
    .enter()
    .append('rect')
    .attr('y', d => y(d.label)!)
    .attr('width', d => x(d.value))
    .attr('height', y.bandwidth())
    .attr('fill', d => d.color)
    .attr('rx', 4)
    .on('mouseover', function (event: MouseEvent, d) {
      d3.select(this).attr('opacity', 0.8);
      showTooltip(event, `<b>${d.label}</b><br>${measure}: ${d.value.toFixed(4)}`);
    })
    .on('mousemove', function (event: MouseEvent) {
      tooltip.style.left = event.clientX + 12 + 'px';
      tooltip.style.top = event.clientY - 10 + 'px';
    })
    .on('mouseout', function () {
      d3.select(this).attr('opacity', 1);
      hideTooltip();
    });

  g.selectAll('.val-label')
    .data(data)
    .enter()
    .append('text')
    .attr('y', d => y(d.label)! + y.bandwidth() / 2)
    .attr('x', d => x(d.value) + 5)
    .attr('dy', '0.35em')
    .attr('font-size', '11px')
    .attr('fill', '#666')
    .text(d => d.value.toFixed(3));

  g.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(y).tickSize(0).tickPadding(8));
}

function renderCentralities(cent: CentralityResult) {
  renderCentralityChart(chartContainer1, cent, selectedMeasure1);
  renderCentralityChart(chartContainer2, cent, selectedMeasure2);
}

// ═══════════════════════════════════════════════════════════
//  State Frequency Chart
// ═══════════════════════════════════════════════════════════
const freqContainer = document.getElementById('frequency-chart')!;

function renderFrequencies(model: TNA) {
  const labels = model.labels;
  const inits = model.inits;

  const data = labels.map((label, i) => ({
    label,
    value: inits[i]!,
    color: NODE_COLORS[i % NODE_COLORS.length]!,
  })).sort((a, b) => b.value - a.value);

  const maxVal = Math.max(...data.map(d => d.value), 1e-6);

  const rect = freqContainer.getBoundingClientRect();
  const width = Math.max(rect.width, 500);
  const height = 260;
  const margin = { top: 10, right: 40, bottom: 30, left: 85 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  d3.select(freqContainer).selectAll('*').remove();

  const svg = d3.select(freqContainer)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const y = d3.scaleBand()
    .domain(data.map(d => d.label))
    .range([0, innerH])
    .padding(0.2);

  const x = d3.scaleLinear()
    .domain([0, maxVal * 1.15])
    .range([0, innerW]);

  g.selectAll('rect')
    .data(data)
    .enter()
    .append('rect')
    .attr('y', d => y(d.label)!)
    .attr('width', d => x(d.value))
    .attr('height', y.bandwidth())
    .attr('fill', d => d.color)
    .attr('rx', 4)
    .on('mouseover', function (event: MouseEvent, d) {
      d3.select(this).attr('opacity', 0.8);
      showTooltip(event, `<b>${d.label}</b><br>Frequency: ${d.value.toFixed(4)}`);
    })
    .on('mousemove', function (event: MouseEvent) {
      tooltip.style.left = event.clientX + 12 + 'px';
      tooltip.style.top = event.clientY - 10 + 'px';
    })
    .on('mouseout', function () {
      d3.select(this).attr('opacity', 1);
      hideTooltip();
    });

  g.selectAll('.val-label')
    .data(data)
    .enter()
    .append('text')
    .attr('y', d => y(d.label)! + y.bandwidth() / 2)
    .attr('x', d => x(d.value) + 5)
    .attr('dy', '0.35em')
    .attr('font-size', '11px')
    .attr('fill', '#666')
    .text(d => d.value.toFixed(3));

  g.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(y).tickSize(0).tickPadding(8));

  g.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(5).tickSize(3));
}

// ═══════════════════════════════════════════════════════════
//  Sequence Index Plot
// ═══════════════════════════════════════════════════════════
const seqContainer = document.getElementById('sequence-plot')!;

function renderSequences() {
  const data = sampleData;
  // Strip trailing nulls per sequence
  const cleaned = data.map(seq => {
    let last = seq.length - 1;
    while (last >= 0 && seq[last] === null) last--;
    return seq.slice(0, last + 1) as string[];
  });

  const maxLen = Math.max(...cleaned.map(s => s.length));

  // Build label → color map (use same order as the model labels)
  const model = buildCurrentModel();
  const labels = model.labels;
  const colorMap = new Map<string, string>();
  labels.forEach((l, i) => { colorMap.set(l, NODE_COLORS[i % NODE_COLORS.length]!); });

  const rect = seqContainer.getBoundingClientRect();
  const width = Math.max(rect.width, 500);
  const margin = { top: 10, right: 120, bottom: 30, left: 70 };
  const cellH = 18;
  const innerH = cleaned.length * cellH;
  const height = innerH + margin.top + margin.bottom;
  const innerW = width - margin.left - margin.right;
  const cellW = innerW / maxLen;

  d3.select(seqContainer).selectAll('*').remove();

  const svg = d3.select(seqContainer)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Draw colored blocks
  cleaned.forEach((seq, row) => {
    seq.forEach((state, col) => {
      g.append('rect')
        .attr('x', col * cellW)
        .attr('y', row * cellH)
        .attr('width', cellW - 0.5)
        .attr('height', cellH - 1)
        .attr('fill', colorMap.get(state) ?? '#ccc')
        .attr('rx', 1)
        .on('mouseover', function (event: MouseEvent) {
          d3.select(this).attr('stroke', '#333').attr('stroke-width', 1.5);
          showTooltip(event, `<b>${state}</b><br>Seq ${row + 1}, Step ${col + 1}`);
        })
        .on('mousemove', function (event: MouseEvent) {
          tooltip.style.left = event.clientX + 12 + 'px';
          tooltip.style.top = event.clientY - 10 + 'px';
        })
        .on('mouseout', function () {
          d3.select(this).attr('stroke', 'none');
          hideTooltip();
        });
    });
  });

  // Y axis: sequence numbers
  const yScale = d3.scaleBand()
    .domain(cleaned.map((_, i) => `${i + 1}`))
    .range([0, innerH]);

  g.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(yScale).tickSize(0).tickPadding(6))
    .selectAll('text').attr('font-size', '10px');

  // X axis: time steps
  const xScale = d3.scaleLinear().domain([0, maxLen]).range([0, innerW]);
  g.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).ticks(Math.min(maxLen, 10)).tickSize(3))
    .selectAll('text').attr('font-size', '10px');

  g.append('text')
    .attr('x', innerW / 2).attr('y', innerH + 26)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Time Step');

  g.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -innerH / 2).attr('y', -50)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Sequence');

  // Legend
  const legend = svg.append('g')
    .attr('transform', `translate(${width - margin.right + 10}, ${margin.top})`);

  labels.forEach((label, i) => {
    const ly = i * 16;
    legend.append('rect')
      .attr('x', 0).attr('y', ly)
      .attr('width', 10).attr('height', 10)
      .attr('rx', 2)
      .attr('fill', colorMap.get(label) ?? '#ccc');
    legend.append('text')
      .attr('x', 14).attr('y', ly + 9)
      .attr('font-size', '9px').attr('fill', '#555')
      .text(label);
  });
}

// ═══════════════════════════════════════════════════════════
//  Mosaic Plot (State Associations)
// ═══════════════════════════════════════════════════════════
const mosaicContainer = document.getElementById('mosaic-plot')!;

/** Compute adjusted standardized residuals (R's chisq.test()$stdres). */
function computeStdRes(tab: number[][]): number[][] {
  const n = tab.length;
  let N = 0;
  const rowSums = new Array(n).fill(0) as number[];
  const colSums = new Array(n).fill(0) as number[];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = tab[i]![j]!;
      rowSums[i] += v;
      colSums[j] += v;
      N += v;
    }
  }
  if (N === 0) return tab.map(r => r.map(() => 0));

  const res: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const expected = (rowSums[i]! * colSums[j]!) / N;
      const denom = Math.sqrt(
        expected * (1 - rowSums[i]! / N) * (1 - colSums[j]! / N),
      );
      row.push(denom > 1e-12 ? (tab[i]![j]! - expected) / denom : 0);
    }
    res.push(row);
  }
  return res;
}

function renderMosaic(model: TNA) {
  const labels = model.labels;
  const n = labels.length;
  const weights = model.weights;

  // Transpose: tab[i][j] = weights[j][i] (R convention)
  const tab: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(Math.max(0, weights.get(j, i)));
    }
    tab.push(row);
  }

  const residuals = computeStdRes(tab);

  const rect = mosaicContainer.getBoundingClientRect();
  const size = Math.min(Math.max(rect.width, 300), 450);
  const margin = { top: 10, right: 10, bottom: 50, left: 55 };
  const innerW = size - margin.left - margin.right;
  const innerH = size - margin.top - margin.bottom;

  // Row totals → tile widths
  const rowTotals = tab.map(r => r.reduce((a, b) => a + b, 0));
  const totalSum = rowTotals.reduce((a, b) => a + b, 0);
  const tileWidths = rowTotals.map(r => totalSum > 0 ? r / totalSum : 0);

  // RdBu color scale: blue = positive residual, red = negative
  const colorScale = d3.scaleDiverging(d3.interpolateRdBu).domain([4, 0, -4]);

  d3.select(mosaicContainer).selectAll('*').remove();

  const svg = d3.select(mosaicContainer)
    .append('svg')
    .attr('width', size)
    .attr('height', size);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  let xStart = 0;
  for (let i = 0; i < n; i++) {
    const w = tileWidths[i]! * innerW;
    if (w < 0.5) { xStart += w; continue; }

    const rowSum = rowTotals[i]!;
    const colProps = rowSum > 0 ? tab[i]!.map(v => v / rowSum) : new Array(n).fill(0);

    let yStart = 0;
    for (let j = 0; j < n; j++) {
      const h = (colProps[j] ?? 0) * innerH;
      if (h < 0.5) { yStart += h; continue; }

      const stdres = residuals[i]![j]!;
      const val = tab[i]![j]!;

      g.append('rect')
        .attr('x', xStart + 0.5)
        .attr('y', yStart + 0.5)
        .attr('width', Math.max(0, w - 1))
        .attr('height', Math.max(0, h - 1))
        .attr('fill', colorScale(stdres))
        .attr('stroke', '#333')
        .attr('stroke-width', 0.5)
        .on('mouseover', function (event: MouseEvent) {
          d3.select(this).attr('stroke-width', 2);
          showTooltip(event,
            `<b>${labels[i]} → ${labels[j]}</b><br>` +
            `Count: ${Number.isInteger(val) ? val : val.toFixed(2)}<br>Residual: ${stdres.toFixed(2)}`);
        })
        .on('mousemove', function (event: MouseEvent) {
          tooltip.style.left = event.clientX + 12 + 'px';
          tooltip.style.top = event.clientY - 10 + 'px';
        })
        .on('mouseout', function () {
          d3.select(this).attr('stroke-width', 0.5);
          hideTooltip();
        });

      // Value label if tile is big enough
      if (w > 25 && h > 14) {
        g.append('text')
          .attr('x', xStart + w / 2)
          .attr('y', yStart + h / 2)
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('font-size', '8px')
          .attr('fill', Math.abs(stdres) > 2.5 ? '#fff' : '#333')
          .attr('pointer-events', 'none')
          .text(Number.isInteger(val) ? val : val.toFixed(2));
      }

      yStart += h;
    }
    xStart += w;
  }

  // X-axis labels (incoming edges = to-states)
  xStart = 0;
  for (let i = 0; i < n; i++) {
    const w = tileWidths[i]! * innerW;
    if (w > 5) {
      g.append('text')
        .attr('x', xStart + w / 2)
        .attr('y', innerH + 12)
        .attr('text-anchor', 'end')
        .attr('font-size', '9px')
        .attr('fill', '#555')
        .attr('transform', `rotate(-40, ${xStart + w / 2}, ${innerH + 12})`)
        .text(labels[i]!);
    }
    xStart += w;
  }

  // Y-axis labels (outgoing edges = from-states) — use first column's proportions
  // Find the widest column for label positioning
  const firstNonZero = tileWidths.findIndex(w => w > 0.01);
  if (firstNonZero >= 0) {
    const rowSum = rowTotals[firstNonZero]!;
    const colProps = rowSum > 0 ? tab[firstNonZero]!.map(v => v / rowSum) : [];
    let yPos = 0;
    for (let j = 0; j < n; j++) {
      const h = (colProps[j] ?? 0) * innerH;
      if (h > 10) {
        g.append('text')
          .attr('x', -6)
          .attr('y', yPos + h / 2)
          .attr('text-anchor', 'end')
          .attr('dy', '0.35em')
          .attr('font-size', '9px')
          .attr('fill', '#555')
          .text(labels[j]!);
      }
      yPos += h;
    }
  }

  // Axis titles
  g.append('text')
    .attr('x', innerW / 2).attr('y', innerH + 42)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Incoming edges');

  g.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -innerH / 2).attr('y', -42)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Outgoing edges');
}

// ═══════════════════════════════════════════════════════════
//  State Distribution Plot
// ═══════════════════════════════════════════════════════════
const distContainer = document.getElementById('distribution-plot')!;

function renderDistribution() {
  const data = sampleData;
  const cleaned = data.map(seq => {
    let last = seq.length - 1;
    while (last >= 0 && seq[last] === null) last--;
    return seq.slice(0, last + 1) as string[];
  });

  const maxLen = Math.max(...cleaned.map(s => s.length));
  const model = buildCurrentModel();
  const labels = model.labels;
  const colorMap = new Map<string, string>();
  labels.forEach((l, i) => { colorMap.set(l, NODE_COLORS[i % NODE_COLORS.length]!); });

  // Compute proportions at each time step
  const proportions: { step: number; state: string; proportion: number; count: number; total: number }[] = [];
  for (let t = 0; t < maxLen; t++) {
    const counts = new Map<string, number>();
    let total = 0;
    for (const seq of cleaned) {
      if (t < seq.length) {
        const s = seq[t]!;
        counts.set(s, (counts.get(s) ?? 0) + 1);
        total++;
      }
    }
    let cumulative = 0;
    for (const label of labels) {
      const c = counts.get(label) ?? 0;
      const p = total > 0 ? c / total : 0;
      proportions.push({ step: t, state: label, proportion: p, count: c, total });
      cumulative += p;
    }
  }

  const rect = distContainer.getBoundingClientRect();
  const width = Math.max(rect.width, 500);
  const height = 260;
  const margin = { top: 10, right: 120, bottom: 35, left: 55 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  d3.select(distContainer).selectAll('*').remove();

  const svg = d3.select(distContainer)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleBand()
    .domain(Array.from({ length: maxLen }, (_, i) => `${i}` ))
    .range([0, innerW])
    .padding(0.05);

  const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  // Stacked bars per time step
  for (let t = 0; t < maxLen; t++) {
    let y0 = 0;
    for (const label of labels) {
      const entry = proportions.find(p => p.step === t && p.state === label);
      const p = entry?.proportion ?? 0;
      if (p <= 0) { y0 += p; continue; }

      g.append('rect')
        .attr('x', xScale(`${t}`)!)
        .attr('y', yScale(y0 + p))
        .attr('width', xScale.bandwidth())
        .attr('height', yScale(y0) - yScale(y0 + p))
        .attr('fill', colorMap.get(label) ?? '#ccc')
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.3)
        .on('mouseover', function (event: MouseEvent) {
          d3.select(this).attr('stroke', '#333').attr('stroke-width', 1.5);
          showTooltip(event,
            `<b>${label}</b><br>Step ${t + 1}: ${(p * 100).toFixed(1)}% (${entry?.count}/${entry?.total})`);
        })
        .on('mousemove', function (event: MouseEvent) {
          tooltip.style.left = event.clientX + 12 + 'px';
          tooltip.style.top = event.clientY - 10 + 'px';
        })
        .on('mouseout', function () {
          d3.select(this).attr('stroke', '#fff').attr('stroke-width', 0.3);
          hideTooltip();
        });

      y0 += p;
    }
  }

  // Axes
  const tickVals = Array.from({ length: maxLen }, (_, i) => `${i}`)
    .filter((_, i) => i % Math.max(1, Math.floor(maxLen / 12)) === 0);

  g.append('g')
    .attr('class', 'axis')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).tickValues(tickVals).tickSize(3))
    .selectAll('text').attr('font-size', '10px');

  g.append('g')
    .attr('class', 'axis')
    .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%')).tickSize(3))
    .selectAll('text').attr('font-size', '10px');

  g.append('text')
    .attr('x', innerW / 2).attr('y', innerH + 30)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Time Step');

  g.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -innerH / 2).attr('y', -40)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px').attr('fill', '#666')
    .text('Proportion');

  // Legend
  const legend = svg.append('g')
    .attr('transform', `translate(${width - margin.right + 10}, ${margin.top})`);

  labels.forEach((label, i) => {
    const ly = i * 16;
    legend.append('rect')
      .attr('x', 0).attr('y', ly)
      .attr('width', 10).attr('height', 10)
      .attr('rx', 2)
      .attr('fill', colorMap.get(label) ?? '#ccc');
    legend.append('text')
      .attr('x', 14).attr('y', ly + 9)
      .attr('font-size', '9px').attr('fill', '#555')
      .text(label);
  });
}

// ═══════════════════════════════════════════════════════════
//  Model Summary
// ═══════════════════════════════════════════════════════════
function renderSummary(model: TNA) {
  const s = summary(model);
  summaryEl.innerHTML = [
    row('Type', model.type),
    row('States', s.nStates),
    row('Edges', s.nEdges),
    row('Density', (s.density as number).toFixed(3)),
    row('Mean Wt', (s.meanWeight as number).toFixed(4)),
    row('Max Wt', (s.maxWeight as number).toFixed(4)),
    row('Self-loops', s.hasSelfLoops ? 'Yes' : 'No'),
  ].join('');
}

function row(label: string, value: unknown): string {
  return `<div><strong>${label}</strong><span>${value}</span></div>`;
}

// ═══════════════════════════════════════════════════════════
//  Master Update
// ═══════════════════════════════════════════════════════════
function update() {
  const model = buildCurrentModel();
  const cent = centralities(model);

  let comm: CommunityResult | undefined;
  if (showCommunities) {
    comm = communities(model, { methods: communityMethod }) as CommunityResult;
  }

  renderNetwork(model, comm);
  renderCentralities(cent);
  renderFrequencies(model);
  renderMosaic(model);
  renderSequences();
  renderDistribution();
  renderSummary(model);
}

// ═══════════════════════════════════════════════════════════
//  Event Listeners
// ═══════════════════════════════════════════════════════════
modelSelect.addEventListener('change', () => {
  modelType = modelSelect.value as typeof modelType;
  update();
});

pruneSlider.addEventListener('input', () => {
  threshold = parseFloat(pruneSlider.value);
  pruneValue.textContent = threshold.toFixed(2);
  update();
});

communityCheck.addEventListener('change', () => {
  showCommunities = communityCheck.checked;
  methodSelect.disabled = !showCommunities;
  update();
});

methodSelect.addEventListener('change', () => {
  communityMethod = methodSelect.value as CommunityMethod;
  if (showCommunities) update();
});

measureSelect1.addEventListener('change', () => {
  selectedMeasure1 = measureSelect1.value as CentralityMeasure;
  const cent = centralities(buildCurrentModel());
  renderCentralityChart(chartContainer1, cent, selectedMeasure1);
});

measureSelect2.addEventListener('change', () => {
  selectedMeasure2 = measureSelect2.value as CentralityMeasure;
  const cent = centralities(buildCurrentModel());
  renderCentralityChart(chartContainer2, cent, selectedMeasure2);
});

// ═══════════════════════════════════════════════════════════
//  Initialize
// ═══════════════════════════════════════════════════════════
const initialModel = buildCurrentModel();
initLayout(initialModel.labels);
update();
