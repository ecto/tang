// boot.js — tang landing page bootstrap
// Loads tang WASM (math) + loon WASM (rendering), bridges them.
// tang computes: curve paths, dual evaluations
// loon renders: all SVG visualizations
// boot.js: animation loop + glue (no visualization logic)

import tangInit, { dual_eval, curve_svg_path, tangent_svg, to_svg } from './pkg/tang/tang_site_wasm.js';
import loonInit, { init_dom_bridge, eval_ui } from './pkg/loon/loon_wasm.js';

// --- DOM Bridge (minimal, just what viz.loon needs) ---

const nodes = [document];
let nextHandle = 1;

function allocHandle(node) {
  const h = nextHandle++;
  nodes[h] = node;
  return h;
}

function getNode(handle) {
  return nodes[handle];
}

function domBridge(op, args) {
  switch (op) {
    case 'createElement': {
      const tag = args[0];
      const svgTags = ['svg','path','circle','rect','line','polyline','polygon','ellipse','g','defs','use','text','tspan','clipPath','mask','pattern','image','foreignObject','linearGradient','radialGradient','stop'];
      const el = svgTags.includes(tag)
        ? document.createElementNS('http://www.w3.org/2000/svg', tag)
        : document.createElement(tag);
      return allocHandle(el);
    }
    case 'createText': {
      return allocHandle(document.createTextNode(args[0]));
    }
    case 'setAttribute': {
      const node = getNode(args[0]);
      if (node) node.setAttribute(args[1], args[2]);
      return null;
    }
    case 'setStyle': {
      const node = getNode(args[0]);
      if (node) node.style.setProperty(args[1], args[2]);
      return null;
    }
    case 'appendChild': {
      const parent = getNode(args[0]);
      const child = getNode(args[1]);
      if (parent && child) parent.appendChild(child);
      return null;
    }
    case 'setText': {
      const node = getNode(args[0]);
      if (node) node.textContent = args[1];
      return null;
    }
    case 'setInnerHTML': {
      const node = getNode(args[0]);
      if (node) node.innerHTML = args[1];
      return null;
    }
    case 'querySelector': {
      const el = document.querySelector(args[0]);
      if (!el) return null;
      return allocHandle(el);
    }
    default:
      console.warn('Unknown bridge op:', op);
      return null;
  }
}

// --- Boot ---

async function boot() {
  // Load both WASM modules in parallel
  await Promise.all([tangInit(), loonInit()]);

  // Initialize loon DOM bridge
  init_dom_bridge(domBridge);

  // Compute curve path using tang WASM (Dual<f64> under the hood)
  const xMin = -8, xMax = 8, yMin = -6, yMax = 8;
  const W = 800, H = 320;
  const curvePath = curve_svg_path(xMin, xMax, yMin, yMax, W, H, 400);

  // Load and execute loon visualization source
  const loonSource = await fetch('./viz.loon').then(r => r.text());

  // Inject curve path data and call main
  const fullSource = loonSource + `\n[main "${curvePath}"]`;

  try {
    eval_ui(fullSource);
  } catch (e) {
    console.error('Loon eval error:', e);
  }

  // --- Animation loop (tang WASM computes, updates SVG attributes) ---
  const tangentLine = document.getElementById('tangent-line');
  const dualPoint = document.getElementById('dual-point');
  const slopeLabel = document.getElementById('slope-label');
  const roX = document.getElementById('ro-x');
  const roVal = document.getElementById('ro-val');
  const roDual = document.getElementById('ro-dual');

  let t = 0;
  const speed = 0.4;

  function animate() {
    t += 0.016;
    const cx = xMin + 1.5 + ((t * speed) % (xMax - xMin - 3));

    // tang WASM: compute f(x) and f'(x) via Dual<f64>
    const [val, dual] = dual_eval(cx);

    // tang WASM: compute tangent line endpoints in SVG coords
    const [tx1, ty1, tx2, ty2] = tangent_svg(cx, xMin, xMax, yMin, yMax, W, H, 2.5);

    // tang WASM: compute point position in SVG coords
    const [px, py] = to_svg(cx, val, xMin, xMax, yMin, yMax, W, H);

    // Update SVG elements (structure built by loon, values from tang)
    if (tangentLine) {
      tangentLine.setAttribute('x1', tx1.toFixed(1));
      tangentLine.setAttribute('y1', ty1.toFixed(1));
      tangentLine.setAttribute('x2', tx2.toFixed(1));
      tangentLine.setAttribute('y2', ty2.toFixed(1));
    }
    if (dualPoint) {
      dualPoint.setAttribute('cx', px.toFixed(1));
      dualPoint.setAttribute('cy', py.toFixed(1));
    }
    if (slopeLabel) {
      slopeLabel.setAttribute('x', (px + 12).toFixed(1));
      slopeLabel.setAttribute('y', (py - 12).toFixed(1));
      slopeLabel.textContent = `slope = ${dual.toFixed(2)}`;
    }

    // Update readout bar
    if (roX) roX.textContent = cx.toFixed(2);
    if (roVal) roVal.textContent = val.toFixed(2);
    if (roDual) roDual.textContent = dual.toFixed(2);

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

boot();
