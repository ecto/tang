// boot.js — tang landing page bootstrap
// Loads tang WASM (math) + loon WASM (rendering), bridges them.
// tang computes: curve paths, dual evaluations
// loon renders: all SVG visualizations
// boot.js: animation loop + glue (no visualization logic)

import tangInit, { dual_eval, curve_svg_path, tangent_svg, to_svg, poet_init, poet_train_epoch, poet_generate, expr_kernel, orbit_eval, arm_ik_step, graph_layout, graph_values } from './pkg/tang/tang_site_wasm.js?v=code';
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

// --- Visibility: only run what's on screen ---

const Vis = (() => {
  const state = new WeakMap(), subs = new WeakMap();
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    state.set(e.target, e.isIntersecting);
    e.target.classList.toggle('offscreen', !e.isIntersecting);
    if (e.target instanceof SVGSVGElement) e.isIntersecting ? e.target.unpauseAnimations() : e.target.pauseAnimations();
    (subs.get(e.target) || []).forEach(f => f(e.isIntersecting));
  }), { rootMargin: '120px' });
  return {
    watch(el, cb) {
      if (!el) return () => false;
      io.observe(el);
      if (cb) subs.set(el, [...(subs.get(el) || []), cb]);
      return () => state.get(el) === true;
    },
  };
})();

// Run fn every frame while el is on screen; stop the frame loop entirely otherwise.
function visibleLoop(el, fn) {
  let id = 0;
  const run = now => { fn(now); id = requestAnimationFrame(run); };
  Vis.watch(el, on => { cancelAnimationFrame(id); id = on ? requestAnimationFrame(run) : 0; });
}

// --- Boot ---

async function boot() {
  // Load both WASM modules in parallel
  await Promise.all([tangInit({ module_or_path: './pkg/tang/tang_site_wasm_bg.wasm?v=code' }), loonInit()]);
  setupKernelDemo();
  const agentToggle = document.getElementById('agent-toggle'), agentPanel = document.getElementById('agent-panel');
  if (agentToggle) agentToggle.onclick = () => { const open = agentPanel.hidden; agentPanel.hidden = !open; agentToggle.setAttribute('aria-expanded', open); };
  const agentCopy = document.getElementById('agent-copy');
  if (agentCopy) {
    const body = document.getElementById('agent-prompt'), more = document.getElementById('agent-more');
    more.onclick = () => { const open = body.parentElement.classList.toggle('open'); more.textContent = open ? 'Hide the full prompt' : 'Show the full prompt'; };
    agentCopy.onclick = () => { navigator.clipboard.writeText(body.querySelector('.agent-first').textContent + '\n\n' + body.querySelector('.agent-rest').textContent); agentCopy.textContent = 'Copied'; setTimeout(() => (agentCopy.textContent = 'Copy prompt'), 1400); };
  }
  const cargo = document.getElementById('cargo-copy');
  if (cargo) cargo.onclick = () => { navigator.clipboard.writeText('cargo add tang'); const h = document.getElementById('cargo-hint'); h.textContent = 'Copied'; setTimeout(() => (h.textContent = 'Copy'), 1400); };
  setupMech();
  setupOrbit();
  setupArm();
  document.querySelectorAll('.deeper-art svg, .mech-fig svg').forEach(el => Vis.watch(el));
  const illo = document.getElementById('build-illo');
  Vis.watch(illo, on => illo.querySelector('svg')?.[on ? 'unpauseAnimations' : 'pauseAnimations']());
  setupGraphPanels();

  // Initialize loon DOM bridge
  init_dom_bridge(domBridge);

  // Compute curve path using tang WASM (Dual<f64> under the hood)
  const xMin = -8, xMax = 8, yMin = -6, yMax = 8;
  const W = 800, H = 320;
  const curvePath = curve_svg_path(xMin, xMax, yMin, yMax, W, H, 400);

  // Load and execute loon visualization source
  const loonSource = await fetch('./viz.loon').then(r => r.text());

  try {
    eval_ui(loonSource + '\n[main]');
  } catch (e) {
    console.error('Loon eval error:', e);
  }

  // Inject curve path via JS (too long for loon string literals)
  const curveEl = document.getElementById('curve-path');
  if (curveEl) curveEl.setAttribute('d', curvePath);

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
    window.__dualX = cx;

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
  }

  visibleLoop(document.getElementById('dual-viz'), animate);

  // --- Quantum Poet interactive demo ---

  const trainBtn = document.getElementById('poet-train-btn');
  const terminal = document.getElementById('poet-terminal');
  const progressWrap = document.getElementById('poet-progress-wrap');
  const epochLabel = document.getElementById('poet-epoch-label');
  const lossLabel = document.getElementById('poet-loss-label');
  const progressBar = document.getElementById('poet-progress-bar');
  const sparkline = document.getElementById('poet-sparkline-line');
  const generateWrap = document.getElementById('poet-generate-wrap');
  const generateBtn = document.getElementById('poet-generate-btn');
  const seedInput = document.getElementById('poet-seed');
  const tempSlider = document.getElementById('poet-temp');
  const tempLabel = document.getElementById('poet-temp-label');
  const poetOutput = document.getElementById('poet-output');

  const TOTAL_EPOCHS = 200;
  const EPOCHS_PER_FRAME = 5;
  let poetLosses = [];

  if (tempSlider) {
    tempSlider.addEventListener('input', () => {
      tempLabel.textContent = parseFloat(tempSlider.value).toFixed(1);
    });
  }

  function updateSparkline() {
    if (!sparkline || poetLosses.length < 2) return;
    const maxLoss = poetLosses[0];
    const w = 200;
    const h = 48;
    const points = poetLosses.map((loss, i) => {
      const x = (i / (TOTAL_EPOCHS - 1)) * w;
      const y = (loss / maxLoss) * (h - 4) + 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    sparkline.setAttribute('points', points);
  }

  if (trainBtn) {
    trainBtn.addEventListener('click', () => {
      trainBtn.disabled = true;
      trainBtn.textContent = 'Initializing...';
      trainBtn.classList.add('opacity-50', 'cursor-wait');

      // Initialize on next frame so the button text updates first
      requestAnimationFrame(() => {
        poet_init();
        trainBtn.textContent = 'Training...';
        terminal.innerHTML = '';
        progressWrap.classList.remove('hidden');

        // pop the tangerine out
        const mascot = document.getElementById('poet-mascot');
        if (mascot) {
          setTimeout(() => {
            mascot.style.opacity = '1';
            mascot.style.transform = 'translateY(0)';
          }, 800);
        }

        let epoch = 0;

        function trainFrame() {
          const batchEnd = Math.min(epoch + EPOCHS_PER_FRAME, TOTAL_EPOCHS);
          let loss = 0;
          for (; epoch < batchEnd; epoch++) {
            loss = poet_train_epoch();
            window.__poet = { loss, epoch: (window.__poet?.epoch || 0) + 1 };
            poetLosses.push(loss);
          }

          // Update UI
          epochLabel.textContent = `epoch ${epoch}/${TOTAL_EPOCHS}`;
          lossLabel.textContent = `loss = ${loss.toFixed(4)}`;
          progressBar.style.width = `${(epoch / TOTAL_EPOCHS) * 100}%`;
          terminal.innerHTML = `<span class="text-muted">$</span> training... epoch ${epoch}/${TOTAL_EPOCHS}  loss = ${loss.toFixed(4)}`;
          updateSparkline();

          if (epoch < TOTAL_EPOCHS) {
            requestAnimationFrame(trainFrame);
          } else {
            // Training complete
            trainBtn.textContent = 'Trained \u2713';
            trainBtn.classList.remove('cursor-wait');
            trainBtn.classList.add('text-green-400');
            terminal.innerHTML = `<span class="text-muted">$</span> training complete. final loss = ${loss.toFixed(4)}`;
            generateWrap.classList.remove('hidden');
          }
        }

        requestAnimationFrame(trainFrame);
      });
    });
  }

  if (generateBtn) {
    generateBtn.addEventListener('click', () => {
      const seed = seedInput.value;
      const temp = parseFloat(tempSlider.value);
      generateBtn.disabled = true;
      generateBtn.textContent = 'Generating...';
      requestAnimationFrame(() => {
        const result = poet_generate(seed, temp);
        poetOutput.textContent = result;
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate';

        // swap mascot speech bubble
        const mascot = document.getElementById('poet-mascot');
        if (mascot) {
          const bubble = mascot.querySelector('div');
          if (bubble) bubble.textContent = 'Congrats on training and running your first model from the browser!';
        }
      });
    });
  }
}

// --- Kernel demo: expression in, kernel source out (tang-expr in WASM) ---

function setupKernelDemo() {
  const input = document.getElementById('kd-input');
  if (!input) return;
  const out = document.getElementById('kd-out');
  const grad = document.getElementById('kd-grad');
  const tabs = [...document.querySelectorAll('[data-kd-dialect]')];
  let dialect = 'cuda';
  const render = () => {
    const src = expr_kernel(input.value, dialect, grad.checked);
    const err = src.startsWith('error:');
    if (err) out.textContent = src; else out.innerHTML = window.hl(src);
    out.classList.toggle('text-red-400', err);
    out.classList.toggle('text-green-400', !err);
    tabs.forEach(t => t.classList.toggle('text-accent', t.dataset.kdDialect === dialect));
  };
  tabs.forEach(t => t.onclick = () => { dialect = t.dataset.kdDialect; render(); });
  document.querySelectorAll('[data-kd-example]').forEach(b => b.onclick = () => { input.value = b.dataset.kdExample; render(); });
  input.addEventListener('input', render);
  grad.addEventListener('change', render);
  render();
}

// --- Orbit game: three manual shots, then gradient descent takes its turn ---

function setupOrbit() {
  const svg = document.getElementById('orbit-viz');
  if (!svg) return;
  const W = 800, H = 420, S = 52, CX = W / 2, CY = H / 2 + 20, PAD_X = CX, PAD_Y = CY - S;
  const px = x => CX + x * S, py = y => CY - y * S;
  const stars = Array.from({ length: 70 }, (_, i) => [((i * 7919) % W), ((i * 104729) % H), (i % 3) * 0.4 + 0.4]);
  const $ = id => document.getElementById(id);
  let target, shots, best, tangBest, ghosts, current, drag = null, solving = false, timer = 0, paused = false;
  let demo = true, demoTimer = 0;
  const visible = Vis.watch(svg, on => { if (on && paused) { paused = false; descend(); } });
  const toD = arr => { let d = ''; for (let i = 0; i < arr.length; i += 2) d += (i ? 'L' : 'M') + px(arr[i]).toFixed(1) + ',' + py(arr[i + 1]).toFixed(1); return d; };
  const newTarget = () => {
    const a = 0.3 + Math.random() * 2.6 - (Math.random() < 0.5 ? 0 : Math.PI), r = 2.2 + Math.random() * 1.6;
    target = [r * Math.cos(a), r * Math.sin(a) * 0.85];
    shots = 3; best = Infinity; tangBest = null; ghosts = []; current = null; solving = false; clearTimeout(timer); clearTimeout(demoTimer);
    status(); draw();
  };
  const status = () => {
    $('ob-shots').textContent = demo ? (solving ? 'Demo: tang is solving' : 'Demo: a player aims') : solving ? 'tang is solving' : shots ? `${shots} shot${shots > 1 ? 's' : ''} left` : 'Your turn is over';
    $('ob-you').textContent = isFinite(best) ? best.toFixed(3) + ' R' : '–';
    $('ob-tang').textContent = tangBest === null ? '–' : `${tangBest.miss.toFixed(3)} R in ${tangBest.steps} steps`;
    $('ob-solve').disabled = solving;
  };
  const draw = () => {
    let html = stars.map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="var(--m-muted)" opacity=".5"/>`).join('')
      + ghosts.map(g => `<path d="${toD(g.path)}" fill="none" stroke="${g.tang ? 'var(--m-cyan)' : 'var(--m-amber)'}" stroke-opacity="${g.tang ? 0.3 : 0.45}" stroke-width="1"/>`).join('')
      + (current ? `<path d="${toD(current.path)}" fill="none" stroke="${current.tang ? 'var(--m-cyan)' : 'var(--m-orange)'}" stroke-width="2.5"/>` : '')
      + `<circle cx="${px(0)}" cy="${py(0)}" r="${S}" fill="var(--m-surface)" stroke="var(--m-border)"/>`
      + `<circle cx="${PAD_X}" cy="${PAD_Y}" r="5" fill="var(--m-fg)"/>`
      + `<g transform="translate(${px(target[0])},${py(target[1])})"><circle r="16" fill="var(--m-orange)" fill-opacity=".12"/><circle r="5" fill="var(--m-orange)"/></g>`;
    if (drag) {
      const [dx, dy] = drag;
      html += `<path d="M${PAD_X},${PAD_Y} L${PAD_X + dx},${PAD_Y + dy}" stroke="var(--m-fg)" stroke-width="2" stroke-dasharray="4 4"/><circle cx="${PAD_X + dx}" cy="${PAD_Y + dy}" r="6" fill="none" stroke="var(--m-fg)"/>`;
    }
    if (demo) html += `<text x="${W / 2}" y="${H - 16}" text-anchor="middle" font-size="12" fill="var(--m-muted)">Demo. Drag from the launch pad to play.</text>`;
    else if (!drag && shots === 3 && !current) html += `<text x="${PAD_X + 14}" y="${PAD_Y - 14}" font-size="12" fill="var(--m-muted)">drag from the launch pad to aim</text>`;
    svg.innerHTML = html;
  };
  const launch = (speed, heading, tang) => {
    const r = orbit_eval(speed, heading, target[0], target[1]);
    const path = Array.from(r.slice(3)), miss = Math.sqrt(r[0]);
    window.__orbitPath = path;
    if (current) ghosts.push(current);
    if (ghosts.length > 16) ghosts.shift();
    current = { path, tang };
    return { miss, g: [r[1], r[2]] };
  };
  // drag-to-aim slingshot: vector from pad to pointer is the launch velocity
  const local = e => { const b = svg.getBoundingClientRect(); return [(e.clientX - b.left) / b.width * W, (e.clientY - b.top) / b.height * H]; };
  svg.addEventListener('pointerdown', e => { if (demo) { demo = false; newTarget(); } if (!shots || solving) return; const [x, y] = local(e); if (Math.hypot(x - PAD_X, y - PAD_Y) < 40) { drag = [0, 0]; svg.setPointerCapture(e.pointerId); } });
  svg.addEventListener('pointermove', e => { if (!drag) return; const [x, y] = local(e); drag = [x - PAD_X, y - PAD_Y]; draw(); });
  svg.addEventListener('pointerup', () => {
    if (!drag) return;
    const [dx, dy] = drag; drag = null;
    const speed = Math.min(1.6, Math.hypot(dx, dy) / 110), heading = Math.atan2(-dy, dx);
    if (speed < 0.2 || heading <= 0.02 || heading >= Math.PI - 0.02) { draw(); return; }
    const { miss } = launch(speed, heading, false);
    shots--; best = Math.min(best, miss); status(); draw();
    if (miss < 0.05) $('ob-shots').textContent = 'Direct hit!';
  });
  // gradient descent: multi-start, then Adam on (speed, heading) with exact dual gradients
  let p, m, v, it;
  const descend = () => {
    if (!visible()) { paused = true; return; }
    const { miss, g } = launch(p[0], p[1], true);
    tangBest = { miss: Math.min(miss, tangBest?.miss ?? Infinity), steps: it };
    status(); draw();
    if (miss < 0.01 || it > 160) { solving = false; status(); if (demo) demoTimer = setTimeout(demoRound, 1800); return; }
    it++;
    for (let k = 0; k < 2; k++) {
      m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k];
      p[k] -= 0.02 * (m[k] / (1 - 0.9 ** it)) / (Math.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8);
    }
    p[0] = Math.min(1.6, Math.max(0.3, p[0])); p[1] = Math.min(Math.PI - 0.05, Math.max(0.05, p[1]));
    timer = setTimeout(descend, 40);
  };
  const solve = () => {
    if (solving) return;
    solving = true; m = [0, 0]; v = [0, 0]; it = 0; tangBest = null;
    let bestMiss = Infinity;
    for (const sp of [0.8, 0.95, 1.1, 1.25, 1.4]) for (let h = 0.15; h < Math.PI; h += 0.3) {
      const miss = orbit_eval(sp, h, target[0], target[1])[0];
      if (miss < bestMiss) { bestMiss = miss; p = [sp, h]; }
    }
    descend();
  };
  $('ob-solve').onclick = () => { demo = false; clearTimeout(demoTimer); solve(); };
  $('ob-new').onclick = () => { demo = false; newTarget(); };
  // attract loop: a pretend player takes two rough shots, then tang solves; repeat until someone plays
  const demoRound = () => {
    if (!demo) return;
    if (!visible()) { demoTimer = setTimeout(demoRound, 500); return; }
    newTarget();
    const aim = Math.atan2(target[1] - 1, target[0]), h = Math.min(Math.PI - 0.1, Math.max(0.1, aim + 0.35));
    const tries = [[1.05 + Math.random() * 0.2, h], [0.95 + Math.random() * 0.25, Math.min(Math.PI - 0.1, Math.max(0.1, h - 0.4))]];
    tries.forEach(([sp, hd], i) => setTimeout(() => {
      if (!demo) return;
      const { miss } = launch(sp, hd, false); shots--; best = Math.min(best, miss); status(); draw();
    }, 700 + i * 1100));
    demoTimer = setTimeout(() => demo && solve(), 700 + tries.length * 1100 + 400);
  };
  newTarget();
  demoRound();
}

// --- Robot arm: pick-and-place loop until you take over, then catch the balls ---

function setupArm() {
  const svg = document.getElementById('arm-viz');
  if (!svg) return;
  const W = 800, H = 420, S = 640, SX = 330, SY = 140, L1 = 0.227, L2 = 0.236;
  const px = x => SX + x * S, py = y => SY - y * S;
  const $ = id => document.getElementById(id);
  const ease = t => t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  let q = [0.2, 0.4], target = [0.2, -0.3], last = performance.now(), mode = 'demo', idleSince = 0;
  // demo: three balls on a shelf, a bin on the other side
  const SHELF = [[0.23, -0.30], [0.275, -0.30], [0.32, -0.30]], BIN = [0.12, 0.12];
  let shelf = [], held = null, binned = [], plan = [], seg = null;
  const resetDemo = () => { shelf = SHELF.map(p => [...p]); held = null; binned = []; plan = []; seg = null; };
  const queue = () => {
    if (!shelf.length) { plan.push({ to: [0.22, -0.18], dur: 900, then: resetDemo }); return; }
    const b = shelf[shelf.length - 1];
    plan.push({ to: [b[0], b[1] + 0.07], dur: 900 }, { to: [b[0], b[1] + 0.012], dur: 450, then: () => { held = shelf.pop(); } },
      { to: [b[0] - 0.02, b[1] + 0.14], dur: 500 }, { to: [BIN[0], BIN[1] + 0.06], dur: 1100 }, { to: [BIN[0], BIN[1] + 0.03], dur: 350, then: () => { binned.push([BIN[0] + (binned.length - 1) * 0.022, BIN[1] - 0.005]); held = null; } },
      { to: [BIN[0] + 0.06, BIN[1] + 0.1], dur: 500 });
  };
  // game
  let balls = [], caught = 0, dropped = 0, spawnIn = 0.4, flashes = [];
  resetDemo();
  const frame = now => {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (mode === 'game' && now - idleSince > 4000 && idleSince) { mode = 'demo'; resetDemo(); balls = []; }
    if (mode === 'demo') {
      if (!seg) { if (!plan.length) queue(); const n = plan.shift(); seg = { ...n, from: [...target], t0: now }; }
      const k = Math.min(1, (now - seg.t0) / seg.dur), e = ease(k);
      target = [seg.from[0] + (seg.to[0] - seg.from[0]) * e, seg.from[1] + (seg.to[1] - seg.from[1]) * e];
      if (k >= 1) { seg.then?.(); seg = null; }
    }
    let r;
    for (let k = 0; k < 3; k++) { r = arm_ik_step(q[0], q[1], target[0], target[1]); q = [r[0], r[1]]; }
    window.__armQ = q;
    const [, , err, ex, ey, tx, ty] = r;
    if (held) held = [tx, ty - 0.022];
    if (mode === 'game') {
      spawnIn -= dt;
      if (spawnIn <= 0) { balls.push({ x: 0.08 + Math.random() * 0.34, y: 0.2, vy: 0 }); spawnIn = 1.0 + Math.random() * 0.9; }
      balls.forEach(b => { b.vy -= 0.22 * dt; b.y += b.vy * dt; });
      balls = balls.filter(b => {
        if (Math.hypot(b.x - tx, b.y - ty) < 0.035) { caught++; flashes.push({ x: b.x, y: b.y, t: now }); return false; }
        if (b.y < -0.56) { dropped++; return false; }
        return true;
      });
    }
    flashes = flashes.filter(f => now - f.t < 500);
    // rendering: grid, reach, links, joint frames and angle arcs, gripper
    const a1 = q[0], a2 = q[0] + q[1];
    const frameAxes = (x, y, ang, len = 26) => {
      const c = Math.cos(ang), s = Math.sin(ang);
      // link frame: x along the link (coral), y normal to it (green)
      return `<path d="M${px(x)},${py(y)} l${s * len},${c * len}" stroke="var(--m-coral)" stroke-width="1.5"/>`
        + `<path d="M${px(x)},${py(y)} l${c * len},${-s * len}" stroke="#39ff14" stroke-width="1.5" opacity=".8"/>`;
    };
    const arc = (x, y, from, to, r, label) => {
      const p0 = [px(x) + r * Math.sin(from), py(y) + r * Math.cos(from)], p1 = [px(x) + r * Math.sin(to), py(y) + r * Math.cos(to)];
      const large = Math.abs(to - from) > Math.PI ? 1 : 0, sweep = to < from ? 1 : 0;
      const mid = (from + to) / 2;
      return `<path d="M${p0[0]},${p0[1]} A${r},${r} 0 ${large} ${sweep} ${p1[0]},${p1[1]}" fill="none" stroke="var(--m-amber)" stroke-width="1.25"/>`
        + `<text x="${px(x) + (r + 14) * Math.sin(mid)}" y="${py(y) + (r + 14) * Math.cos(mid) + 4}" text-anchor="middle" font-size="10.5" fill="var(--m-amber)">${label}</text>`;
    };
    let grid = '';
    for (let gx = -0.4; gx <= 0.8; gx += 0.1) grid += `<path d="M${px(gx)},0 V${H}" stroke="var(--m-border-muted)" stroke-width=".5"/>`;
    for (let gy = -0.6; gy <= 0.3; gy += 0.1) grid += `<path d="M0,${py(gy)} H${W}" stroke="var(--m-border-muted)" stroke-width=".5"/>`;
    const deg = r => Math.round(r * 180 / Math.PI) + '°';
    svg.innerHTML = grid
      + `<rect x="${px(-0.09)}" y="${py(0.08)}" width="${0.07 * S}" height="${0.7 * S}" fill="var(--m-surface)" stroke="var(--m-border)"/>`
      + `<circle cx="${px(0)}" cy="${py(0)}" r="${(L1 + L2) * S}" fill="none" stroke="var(--m-border)" stroke-dasharray="2 6"/>`
      + `<path d="M${px(0)},${py(0)} V${py(-0.12)}" stroke="var(--m-muted)" stroke-dasharray="3 4"/>`
      + (mode === 'demo'
        ? `<path d="M${px(0.20)},${py(-0.31)} H${px(0.36)}" stroke="var(--m-muted)" stroke-width="3"/>`
          + `<path d="M${px(BIN[0] - 0.045)},${py(BIN[1] + 0.03)} V${py(BIN[1] - 0.02)} H${px(BIN[0] + 0.045)} V${py(BIN[1] + 0.03)}" fill="none" stroke="var(--m-muted)" stroke-width="2.5"/>`
          + [...shelf, ...binned, ...(held ? [held] : [])].map(([x, y]) => `<circle cx="${px(x)}" cy="${py(y + 0.012)}" r="8" fill="var(--m-cyan)"/>`).join('')
        : balls.map(b => `<circle cx="${px(b.x)}" cy="${py(b.y)}" r="9" fill="var(--m-cyan)"/>`).join(''))
      + flashes.map(f => { const k = (now - f.t) / 500; return `<circle cx="${px(f.x)}" cy="${py(f.y)}" r="${10 + 30 * k}" fill="none" stroke="#39ff14" stroke-opacity="${1 - k}" stroke-width="2"/>`; }).join('')
      + `<g transform="translate(${px(target[0])},${py(target[1])})" stroke="var(--m-muted)" stroke-width="1"><path d="M-7 0H7M0 -7V7"/></g>`
      + `<path d="M${px(0)},${py(0)} L${px(ex)},${py(ey)}" stroke="var(--m-orange)" stroke-width="18" stroke-linecap="round"/>`
      + `<path d="M${px(ex)},${py(ey)} L${px(tx)},${py(ty)}" stroke="var(--m-orange)" stroke-width="14" stroke-linecap="round" opacity=".85"/>`
      + arc(0, 0, 0, a1, 34, deg(q[0])) + arc(ex, ey, a1, a2, 28, deg(q[1]))
      + `<path d="M${px(ex)},${py(ey)} l${Math.sin(a1) * 40},${Math.cos(a1) * 40}" stroke="var(--m-muted)" stroke-dasharray="3 4"/>`
      + frameAxes(0, 0, a1) + frameAxes(ex, ey, a2) + frameAxes(tx, ty, a2, 18)
      + [[0, 0, 12], [ex, ey, 10]].map(([x, y, rr]) => `<circle cx="${px(x)}" cy="${py(y)}" r="${rr}" fill="var(--m-bg)" stroke="var(--m-fg)" stroke-width="2"/><circle cx="${px(x)}" cy="${py(y)}" r="2.5" fill="var(--m-fg)"/>`).join('')
      + `<g transform="translate(${px(tx)},${py(ty)}) rotate(${(-a2 * 180 / Math.PI).toFixed(1)})"><path d="M-12,4 V-2 M12,4 V-2 M-12,-2 H12" stroke="var(--m-fg)" stroke-width="3" fill="none" stroke-linecap="round" transform="rotate(180)"/></g>`
      + `<text x="${px(0) - 18}" y="${py(0) - 16}" text-anchor="end" font-size="11" fill="var(--m-muted)">shoulder pitch</text>`
      + `<text x="${px(ex) - 16}" y="${py(ey) + 24}" text-anchor="end" font-size="11" fill="var(--m-muted)">elbow</text>`
      + (mode === 'demo' ? `<text x="${W / 2}" y="${H - 14}" text-anchor="middle" font-size="12" fill="var(--m-muted)">Demo: pick and place. Move the pointer here to play catch.</text>` : '');
    $('arm-score').textContent = mode === 'demo' ? 'Demo' : `${caught} caught · ${dropped} dropped`;
    $('arm-q1').textContent = `${deg(q[0])} / ${deg(q[1])}`;
    $('arm-err').textContent = (err * 1000).toFixed(1) + ' mm';
  };
  svg.addEventListener('pointermove', e => {
    const b = svg.getBoundingClientRect();
    target = [((e.clientX - b.left) / b.width * W - SX) / S, (SY - (e.clientY - b.top) / b.height * H) / S];
    if (mode === 'demo') { mode = 'game'; caught = 0; dropped = 0; balls = []; spawnIn = 0.3; }
    idleSince = 0;
  });
  svg.addEventListener('pointerleave', e => { idleSince = performance.now(); });
  visibleLoop(svg, frame);
}

// --- Live tang graphs: each demo's own Scalar code, traced with ExprId ---

const Tip = (() => {
  const el = document.createElement('div'); el.className = 'tip'; document.body.appendChild(el);
  let anchor = null;
  return {
    // Anchored to the hovered element, not the cursor: centred below it, or above when there's no room.
    show(e, html) {
      const a = e.target.closest('[data-tip], [data-tip-node], .ka-k') || e.target;
      if (a === anchor && el.style.opacity === '1') { el.innerHTML = html; return; }
      anchor = a; el.innerHTML = html; el.style.opacity = 1;
      const r = a.getBoundingClientRect(), t = el.getBoundingClientRect(), GAP = 10;
      const x = Math.max(8, Math.min(innerWidth - t.width - 8, r.left + r.width / 2 - t.width / 2));
      const y = r.bottom + GAP + t.height > innerHeight ? r.top - GAP - t.height : r.bottom + GAP;
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    },
    hide() { el.style.opacity = 0; anchor = null; },
  };
})();
addEventListener('scroll', () => Tip.hide(), { passive: true });

document.addEventListener('pointermove', e => {
  const t = e.target.closest?.('[data-tip]');
  if (t) Tip.show(e, t.dataset.tip); else if (!e.target.closest?.('[data-tip-node], .ka-k')) Tip.hide();
});

const OP_TIPS = {
  var: 'An input. Its value comes from the demo each frame.',
  lit: 'A constant baked into the graph.',
  '+': 'Addition. Subtraction is addition of a negation.',
  '×': 'Multiplication.',
  '−': 'Negation.',
  '1/': 'Reciprocal. Division is multiplication by a reciprocal.',
  '√': 'Square root.',
  sin: 'Sine, the only trig primitive. cos x is traced as sin(x + π/2).',
  atan2: 'Two-argument arctangent.',
  '2^': 'Base-2 exponential. exp(x) is traced as 2^(x·log₂e).',
  'log₂': 'Base-2 logarithm. ln x is traced as log₂(x)·ln 2.',
  sel: 'Branch-free select: picks one of two values without a jump.',
};
const OUT_NAMES = {
  dual: ['f(x) = x·sin x', 'f′(x), by symbolic differentiation'],
  orbit: ['acceleration x', 'acceleration y'],
  arm: ['tip x', 'tip y'],
};
const KERNEL_TIPS = {
  'flash attention': 'Attention as one online-softmax reduction. The compiler derived it from max and two sums; the score matrix never reaches memory.',
  matmul: 'A contraction: a sum over a shared index of a product. Projections and the feed-forward layers.',
  reduce: 'A reduction with a custom combine, such as a running max with its sum.',
  max: 'A max reduction, as in the softmax stabilizer.',
  sum: 'A sum reduction, as in RMSNorm\'s mean of squares.',
  map: 'Elementwise work with no reduction: activations, RoPE, residual adds, scaling.',
};

function GraphPanel(el) {
  const W = 720, NODE_W = 78, NODE_H = 38;
  let kind = null, nodes = [], outs = [], valEls = [], edgeEls = [], last = [];
  const fmt = v => !isFinite(v) ? '∞' : Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(Math.abs(v) >= 100 ? 0 : 2);
  const head = document.createElement('div'); head.className = 'gp-head';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('class', 'gp-svg');
  el.append(head, svg);
  const layout = k => {
    kind = k;
    const r = JSON.parse(graph_layout(k));
    if (r.error) { head.textContent = r.error; svg.innerHTML = ''; nodes = []; return; }
    nodes = r.nodes; outs = r.outputs;
    const depth = nodes.map(() => 0);
    nodes.forEach((n, i) => { depth[i] = n.args.length ? 1 + Math.max(...n.args.map(a => depth[a])) : 0; });
    const cols = Math.max(...depth) + 1, byCol = Array.from({ length: cols }, () => []);
    nodes.forEach((n, i) => byCol[depth[i]].push(i));
    const rows = Math.max(...byCol.map(c => c.length));
    const H = Math.max(90, rows * (NODE_H + 12) + 20);
    const colW = (W - NODE_W - 20) / Math.max(1, cols - 1);
    const pos = nodes.map(() => [0, 0]);
    byCol.forEach((c, ci) => c.forEach((i, ri) => {
      pos[i] = [10 + ci * colW, (H - c.length * (NODE_H + 12)) / 2 + ri * (NODE_H + 12) + 6];
    }));
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    let html = '';
    nodes.forEach((n, i) => n.args.forEach(a => {
      const [x1, y1] = pos[a], [x2, y2] = pos[i];
      const sx = x1 + NODE_W, sy = y1 + NODE_H / 2, ex = x2, ey = y2 + NODE_H / 2, mx = (sx + ex) / 2;
      html += `<path class="gp-edge" data-to="${i}" d="M${sx},${sy} C${mx},${sy} ${mx},${ey} ${ex},${ey}"/>`;
    }));
    nodes.forEach((n, i) => {
      const oi = outs.indexOf(i), cls = `gp-node gp-${n.kind}${oi === 0 ? ' gp-out' : oi > 0 ? ' gp-out2' : ''}`;
      const [x, y] = pos[i];
      html += `<g class="${cls}" data-tip-node="${i}" data-id="${n.id}" transform="translate(${x},${y})"><rect width="${NODE_W}" height="${NODE_H}"/>`
        + `<text x="${NODE_W / 2}" y="15" class="gp-op">${n.op}</text>`
        + (n.kind === 'lit' ? '' : `<text x="${NODE_W / 2}" y="30" class="gp-val" data-i="${i}"></text>`) + '</g>';
    });
    svg.innerHTML = html;
    valEls = [...svg.querySelectorAll('.gp-val')];
    edgeEls = [...svg.querySelectorAll('.gp-edge')];
    const ops = nodes.filter(n => n.kind === 'op').length;
    head.innerHTML = `<span>tang graph</span><span>${nodes.length} nodes · ${ops} ops · evaluated live</span>`;
    last = [];
  };
  const update = inputs => {
    if (!nodes.length) return;
    const v = graph_values(kind, new Float64Array(inputs));
    valEls.forEach(t => { const i = +t.dataset.i; t.textContent = fmt(v[i]); });
    const changed = new Set(); for (let i = 0; i < v.length; i++) if (last[i] !== v[i]) changed.add(i);
    edgeEls.forEach(e => e.classList.toggle('gp-live', changed.has(+e.dataset.to)));
    last = Array.from(v);
  };
  svg.addEventListener('pointermove', e => {
    const g = e.target.closest('[data-tip-node]');
    if (!g) return Tip.hide();
    const i = +g.dataset.tipNode, n = nodes[i];
    el.__onNode?.(n);
    const oi = outs.indexOf(i), role = OUT_NAMES[kind]?.[oi] ?? (oi === 0 ? 'output' : oi > 0 ? `∂ output / ∂ input ${oi}` : '');
    const argv = n.args.map(a => `${nodes[a].op}`).join(', ');
    Tip.show(e, `<b>${n.op}</b>${role ? ` <span class="tip-role">${role}</span>` : ''}<br>${OP_TIPS[n.kind === 'op' ? n.op : n.kind] || ''}`
      + (n.args.length ? `<br><span class="tip-dim">reads ${argv}</span>` : '')
      + (last[i] !== undefined && n.kind !== 'lit' ? `<br><span class="tip-dim">value now ${fmt(last[i])}</span>` : ''));
  });
  svg.addEventListener('pointerleave', () => { Tip.hide(); el.__onNode?.(null); });
  const highlight = ids => svg.querySelectorAll('[data-id]').forEach(g => g.classList.toggle('hit', ids.has(+g.dataset.id)));
  return { layout, update, highlight, get nodes() { return nodes; }, get kind() { return kind; } };
}

function setupGraphPanels() {
  const panels = {};
  const seen = {};
  document.querySelectorAll('[data-graph]:not([data-graph=lower])').forEach(el => { panels[el.dataset.graph] = GraphPanel(el); seen[el.dataset.graph] = Vis.watch(el.closest('figure, .fig, section') || el); });
  if (panels.dual) panels.dual.layout('dual');
  if (panels.orbit) panels.orbit.layout('orbit');
  if (panels.arm) panels.arm.layout('arm');
  const input = document.getElementById('kd-input');
  if (panels.kernel && input) {
    const relayout = () => panels.kernel.layout('expr:' + input.value);
    relayout(); input.addEventListener('input', relayout);
    document.querySelectorAll('[data-kd-example]').forEach(b => b.addEventListener('click', () => setTimeout(relayout)));
  }
  let k = 0, lastT = 0, poet = null;
  const poetEl = document.querySelector('[data-graph-module=poet]');
  const poetSeen = Vis.watch(poetEl);
  if (poetEl) {
    const layers = [['input', '240'], ['Linear', '240→64 · 15,424'], ['tanh', ''], ['Linear', '64→64 · 4,160'], ['tanh', ''], ['Linear', '64→30 · 1,950'], ['cross-entropy', 'loss']];
    const tips = [
      'Eight previous characters, one-hot: 8 × 30 = 240 numbers.',
      'y = Wx + b with 64 outputs. 15,360 weights and 64 biases, updated by Adam each step.',
      'Squashes each value into (−1, 1). Its derivative, 1 − tanh², flows back in the backward pass.',
      'A second hidden layer, 64 → 64.',
      'The same nonlinearity again.',
      'Scores for each of the 30 characters that could come next.',
      'Softmax plus negative log-likelihood of the true next character. Its gradient starts the backward pass.',
    ];
    const W = 720, w = 86, gap = (W - w * layers.length) / (layers.length - 1);
    poetEl.innerHTML = `<div class="gp-head"><span>tang-train module graph</span><span id="pg-stat">21,534 parameters · idle</span></div>`
      + `<svg class="gp-svg" viewBox="0 0 ${W} 64">` + layers.map(([op, sub], i) => {
        const x = i * (w + gap);
        return (i ? `<path class="gp-edge pg-edge" d="M${x - gap},32 H${x}"/>` : '')
          + `<g class="gp-node${i === layers.length - 1 ? ' gp-out' : ''}" data-tip="<b>${op}</b><br>${tips[i]}" transform="translate(${x},12)"><rect width="${w}" height="40"/><text x="${w / 2}" y="17" class="gp-op">${op}</text><text x="${w / 2}" y="31" class="gp-val"${i === layers.length - 1 ? ' id="pg-loss"' : ''}>${sub}</text></g>`;
      }).join('') + '</svg>';
    let seen = 0;
    poet = () => {
      const p = window.__poet; if (!p) return;
      const live = p.epoch !== seen; seen = p.epoch;
      poetEl.querySelectorAll('.pg-edge').forEach(e => e.classList.toggle('gp-live', live));
      document.getElementById('pg-loss').textContent = 'loss ' + p.loss.toFixed(3);
      document.getElementById('pg-stat').textContent = `21,534 parameters · epoch ${p.epoch} · ${live ? 'training' : 'trained'}`;
    };
  }
  const tick = now => {
    requestAnimationFrame(tick);
    if (document.hidden) return;
    if (now - lastT < 66) return; lastT = now; const t = now / 1000;
    if (panels.kernel && seen.kernel()) panels.kernel.update([1.5 * Math.sin(t * 0.8), 1.2 * Math.cos(t * 0.6), 0.8 * Math.sin(t * 1.3)]);
    if (panels.dual && seen.dual() && window.__dualX !== undefined) panels.dual.update([window.__dualX]);
    if (panels.arm && seen.arm() && window.__armQ) panels.arm.update(window.__armQ);
    const path = window.__orbitPath;
    if (panels.orbit && seen.orbit() && path && path.length > 4) {
      k = (k + 2) % (path.length / 2);
      const x = path[2 * Math.floor(k)], y = path[2 * Math.floor(k) + 1];
      panels.orbit.update([x, y]);
      const probe = document.getElementById('ob-probe');
      if (probe) probe.setAttribute('transform', `translate(${400 + x * 52},${230 - y * 52})`);
    }
    if (poet && poetSeen()) poet(now);
  };
  requestAnimationFrame(tick);
  setupKernelArcs();
}

// --- Kernel arc diagrams: the real block graph, exported from the IR ---

async function setupKernelArcs() {
  const els = [...document.querySelectorAll('[data-arcs]')];
  if (!els.length) return;
  const graphs = await fetch('./graphs.json').then(r => r.json()).catch(() => null);
  if (!graphs) return;
  const COLOR = { 'flash attention': 'var(--m-orange)', matmul: 'var(--m-cyan)', reduce: 'var(--m-amber)', max: 'var(--m-amber)', sum: 'var(--m-amber)', map: 'var(--m-muted)' };
  for (const el of els) {
    const keys = el.dataset.arcs.split(','), layers = +(el.dataset.layers || 1);
    const W = 720, H = 150, BASE = 104;
    let key = keys[keys.length - 1], timer = 0;
    const arcSeen = Vis.watch(el);
    const draw = () => {
      const g = graphs[key];
      const kernels = g.nodes.map((n, i) => ({ ...n, i })).filter(n => n.kind === 'kernel');
      const idx = new Map(kernels.map((k, j) => [k.i, j]));
      const step = (W - 20) / kernels.length, size = Math.min(26, step - 3);
      const x = j => 10 + j * step + (step - size) / 2;
      let arcs = '', boxes = '';
      kernels.forEach((k, j) => (k.args || []).forEach(a => {
        if (!idx.has(a)) return;
        const s = idx.get(a), x1 = x(s) + size / 2, x2 = x(j) + size / 2, h = Math.min(90, 10 + (x2 - x1) * 0.35);
        arcs += `<path class="ka-arc" data-to="${j}" d="M${x1},${BASE} C${x1},${BASE - h} ${x2},${BASE - h} ${x2},${BASE}"/>`;
      }));
      kernels.forEach((k, j) => {
        const c = COLOR[k.op] || COLOR.map;
        boxes += `<g class="ka-k" data-j="${j}" transform="translate(${x(j)},${BASE})"><rect width="${size}" height="${size}" fill="${c}" fill-opacity=".22" stroke="${c}"/>`
          + (k.fused ? `<text x="${size / 2}" y="${size / 2 + 4}" class="ka-f">+${k.fused}</text>` : '')
          + `</g>`;
      });
      const flash = kernels.findIndex(k => k.op === 'flash attention');
      const note = flash >= 0 ? `<text x="${x(flash) + size / 2}" y="${BASE + size + 16}" class="ka-note" text-anchor="middle">flash attention, derived</text>` : '';
      el.querySelector('svg').innerHTML = arcs + boxes + note;
      el.querySelector('svg').onpointermove = e => {
        const gk = e.target.closest('.ka-k'); if (!gk) return Tip.hide();
        const k = kernels[+gk.dataset.j], reads = (k.args || []).map(a => g.nodes[a]);
        const ins = reads.filter(n => n.kind === 'input').map(n => n.label), kin = reads.filter(n => n.kind === 'kernel').length;
        Tip.show(e, `<b>Kernel ${+gk.dataset.j + 1}: ${k.op}</b><br>${KERNEL_TIPS[k.op] || ''}`
          + `<br><span class="tip-dim">writes ${k.shape}${k.fused ? ` · ${k.fused} op${k.fused > 1 ? 's' : ''} fused in` : ''}</span>`
          + `<br><span class="tip-dim">reads ${[...ins, kin ? `${kin} earlier kernel${kin > 1 ? 's' : ''}` : ''].filter(Boolean).join(', ') || 'nothing'}</span>`);
      };
      el.querySelector('svg').onpointerleave = () => Tip.hide();
      el.querySelector('.ka-stat').textContent = `${g.kernels} kernels${layers > 1 ? ' per layer' : ''} · ${g.nodes.length - kernels.length} inputs`+ (el.dataset.note ? ` · ${el.dataset.note}` : '');
      const ks = [...el.querySelectorAll('.ka-k')], as = [...el.querySelectorAll('.ka-arc')];
      let j = 0; clearInterval(timer);
      timer = setInterval(() => {
        if (!arcSeen()) return;
        ks.forEach((e, n) => e.classList.toggle('ka-on', n === j));
        as.forEach(e => e.classList.toggle('ka-on', +e.dataset.to === j));
        j = (j + 1) % kernels.length;
      }, Math.max(70, 1400 / kernels.length));
    };
    el.innerHTML = `<div class="gp-head"><span>tang kernel graph</span><span class="ka-stat"></span></div>`
      + (keys.length > 1 ? `<div class="ka-tabs">${keys.map(k => `<button data-k="${k}">${k === 'block_written' ? 'As written' : 'Compiled'}</button>`).join('')}</div>` : '')
      + `<svg class="gp-svg" viewBox="0 0 ${W} ${H}"></svg>`
      + `<div class="ka-key">${['flash attention', 'matmul', 'reduce', 'map'].map(o => `<span><i style="background:${COLOR[o]}"></i>${o}</span>`).join('')}<span>+n ops fused in</span></div>`;
    el.querySelectorAll('[data-k]').forEach(b => b.onclick = () => { key = b.dataset.k; el.querySelectorAll('[data-k]').forEach(c => c.classList.toggle('on', c === b)); draw(); });
    el.querySelector(`[data-k="${key}"]`)?.classList.add('on');
    draw();
  }
}

// --- How tang works: four interactive figures ---

function setupMech() {
  // 1. dual numbers, row by row
  const dualFig = document.getElementById('m-dual');
  if (dualFig) {
    const slider = document.getElementById('md-a');
    const cells = [...dualFig.querySelectorAll('.md-c')];
    cells.forEach(c => c.innerHTML = '<i></i><b></b>');
    const set = (r, p, v) => {
      const c = cells.find(c => c.dataset.r == r && c.dataset.p === p), w = Math.min(50, Math.abs(v) / 7 * 50);
      c.firstChild.style.left = (v < 0 ? 50 - w : 50) + '%'; c.firstChild.style.width = w + '%';
      c.lastChild.textContent = v.toFixed(3);
    };
    const render = () => {
      const a = +slider.value, [fv, fe] = dual_eval(a);
      document.getElementById('md-a-v').textContent = a.toFixed(2);
      set(0, 'v', a); set(0, 'e', 1);
      set(1, 'v', Math.sin(a)); set(1, 'e', Math.cos(a));
      set(2, 'v', fv); set(2, 'e', fe);
    };
    slider.addEventListener('input', render); render();
  }

  // 2. views as index maps
  const views = document.getElementById('m-views');
  if (views) {
    const C = ['#ff5f00', '#00f0ff', '#ffd400', '#ff2052', '#39ff14', '#a08d70'];
    const cell = (x, y, r, c, label, src, kind) => `<g class="mv-c" data-src="${src}" data-kind="${kind}"><rect x="${x}" y="${y}" width="30" height="30" fill="${C[src]}" fill-opacity=".22" stroke="${C[src]}"/><text x="${x + 15}" y="${y + 19}" text-anchor="middle" fill="var(--m-fg)">${label}</text></g>`;
    let html = `<text x="20" y="16" fill="var(--m-muted)">source x (2 × 3)</text><text x="200" y="16" fill="var(--m-muted)">transpose, view (i, j) ↦ (j, i)</text><text x="380" y="16" fill="var(--m-muted)">broadcast row, stride 0</text>`;
    const pos = {};
    for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) { const x = 20 + c * 34, y = 26 + r * 34; pos[r * 3 + c] = [x + 15, y + 15]; html += cell(x, y, r, c, `${r},${c}`, r * 3 + c, 'src'); }
    for (let r = 0; r < 3; r++) for (let c = 0; c < 2; c++) html += cell(200 + c * 34, 26 + r * 34, r, c, `${r},${c}`, c * 3 + r, 'out');
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) html += cell(380 + c * 34, 26 + r * 34, r, c, `${r},${c}`, c, 'out');
    views.innerHTML = html + '<path id="mv-link" fill="none" stroke="var(--m-fg)" stroke-width="1.5" stroke-dasharray="3 3" opacity="0"/>'
      + `<text id="mv-note" x="20" y="160" fill="var(--m-muted)">hover an output cell</text>`;
    const link = views.querySelector('#mv-link'), note = views.querySelector('#mv-note');
    views.addEventListener('pointerover', e => {
      const g = e.target.closest('.mv-c[data-kind=out]');
      views.querySelectorAll('.mv-c').forEach(c => c.style.opacity = g ? (c === g || (c.dataset.kind === 'src' && c.dataset.src === g.dataset.src) ? 1 : 0.25) : 1);
      if (!g) { link.setAttribute('opacity', 0); note.textContent = 'hover an output cell'; return; }
      const r = g.querySelector('rect'), [sx, sy] = pos[+g.dataset.src], ox = +r.getAttribute('x') + 15, oy = +r.getAttribute('y') + 15;
      link.setAttribute('d', `M${ox},${oy} C${ox - 60},${oy + 60} ${sx + 60},${sy + 60} ${sx},${sy}`); link.setAttribute('opacity', 0.9);
      const src = +g.dataset.src;
      note.textContent = `output (${g.querySelector('text').textContent}) reads x[${Math.floor(src / 3)}, ${src % 3}], the same memory, no copy`;
    });
  }

  // 3. attention as written vs derived online softmax
  const flash = document.getElementById('m-flash');
  if (flash) {
    const N = 8, scores = Array.from({ length: N }, (_, j) => Math.sin(j * 1.7 + 0.4) * 2 + Math.cos(j * 0.9) * 1.2);
    const T = 900; // ms per key
    visibleLoop(flash, now => {
      const t = (now % (T * (N + 3))) / T, k = Math.min(N, Math.floor(t)), f = Math.min(1, t - Math.floor(t));
      let html = `<text x="20" y="16" fill="var(--m-fg)" font-weight="600">As written</text><text x="330" y="16" fill="var(--m-fg)" font-weight="600">Derived</text>`;
      // left: stored scores for N queries × N keys (row by row)
      const stored = Math.min(N * N, Math.floor(t / (N + 1) * N * N * 1.15));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const n = i * N + j, on = n < stored;
        html += `<rect x="${20 + j * 19}" y="${28 + i * 19}" width="16" height="16" fill="var(--m-coral)" fill-opacity="${on ? 0.25 + 0.6 * (scores[j] + 3.2) / 6.4 : 0.05}" stroke="var(--m-border)" stroke-width=".5"/>`;
      }
      html += `<text x="20" y="200" fill="var(--m-muted)">scores stored</text><rect x="20" y="208" width="${(stored / (N * N)) * 260}" height="8" fill="var(--m-coral)"/><text x="${24 + (stored / (N * N)) * 260}" y="216" fill="var(--m-muted)">${stored} floats</text>`;
      // right: keys stream, state (m, l) updates
      let m = -Infinity, l = 0;
      for (let j = 0; j < k; j++) { const m2 = Math.max(m, scores[j]); l = l * Math.exp(m - m2) + Math.exp(scores[j] - m2); m = m2; }
      for (let j = 0; j < N; j++) {
        const x = 330 + j * 32, done = j < k, cur = j === k && k < N;
        html += `<rect x="${x}" y="${cur ? 34 - 8 * (1 - f) : 34}" width="26" height="26" fill="var(--m-cyan)" fill-opacity="${done ? 0.15 : cur ? 0.8 : 0.4}" stroke="var(--m-cyan)" stroke-opacity="${done ? 0.3 : 1}"/><text x="${x + 13}" y="${cur ? 51 - 8 * (1 - f) : 51}" text-anchor="middle" font-size="9" fill="var(--m-fg)">k${j}</text>`;
      }
      if (k < N) html += `<path d="M${330 + k * 32 + 13},64 V98" stroke="var(--m-cyan)" stroke-dasharray="3 3"/>`;
      const box = (x, label, v) => `<rect x="${x}" y="100" width="76" height="46" fill="var(--m-bg)" stroke="var(--m-orange)"/><text x="${x + 38}" y="118" text-anchor="middle" fill="var(--m-muted)">${label}</text><text x="${x + 38}" y="137" text-anchor="middle" fill="var(--m-fg)" font-size="13">${v}</text>`;
      html += box(330, 'running max m', isFinite(m) ? m.toFixed(2) : '−∞') + box(416, 'running sum ℓ', l.toFixed(2)) + box(502, 'output o', k ? `${k}/${N} keys` : '0');
      html += `<text x="330" y="200" fill="var(--m-muted)">state kept</text><rect x="330" y="208" width="${(3 / (N * N)) * 260}" height="8" fill="var(--m-orange)"/><text x="${334 + (3 / (N * N)) * 260}" y="216" fill="var(--m-muted)">3 values</text>`;
      flash.innerHTML = html;
    });
  }

  // 4. graph and code, linked
  const lowerEl = document.querySelector('[data-graph=lower]');
  if (lowerEl) {
    const panel = GraphPanel(lowerEl), code = document.getElementById('ml-code');
    panel.layout('expr:x * sin(x)');
    let dialect = 'cuda';
    const renderCode = () => {
      const src = expr_kernel('x * sin(x)', dialect, true);
      code.innerHTML = src.split('\n').map(line => {
        const ids = [...line.matchAll(/\bt(\d+)\b/g)].map(m => +m[1]);
        if (/\bx0\b/.test(line)) ids.push(...panel.nodes.filter(n => n.kind === 'var').map(n => n.id));
        return `<span class="ln" data-ids="${ids.join(',')}">${window.hl(line) || ' '}</span>`;
      }).join('');
    };
    renderCode();
    document.querySelectorAll('[data-ml]').forEach(btn => btn.onclick = () => {
      dialect = btn.dataset.ml; document.querySelectorAll('[data-ml]').forEach(x => x.classList.toggle('on', x === btn)); renderCode();
    });
    // a line defines the first id it mentions on its left-hand side
    const definesId = line => { const m = line.textContent.match(/\b(?:float|let|var|double|const)\s+t(\d+)\b/); return m ? +m[1] : null; };
    lowerEl.__onNode = n => code.querySelectorAll('.ln').forEach(l => l.classList.toggle('hit', !!n && (definesId(l) === n.id || (n.kind === 'var' && /\bx0\s*=/.test(l.textContent)))));
    code.addEventListener('pointerover', e => {
      const l = e.target.closest('.ln');
      code.querySelectorAll('.ln').forEach(x => x.classList.toggle('hit', x === l));
      panel.highlight(new Set(l ? (l.dataset.ids || '').split(',').filter(Boolean).map(Number) : []));
    });
    code.addEventListener('pointerleave', () => { panel.highlight(new Set()); code.querySelectorAll('.ln').forEach(x => x.classList.remove('hit')); });
    visibleLoop(lowerEl, now => { if (now % 4 < 1 || !lowerEl.__t || now - lowerEl.__t > 120) { lowerEl.__t = now; panel.update([1.5 * Math.sin(now / 1400)]); } });
  }
}

boot();
