import './styles.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

type ViewMode = 'surface' | 'density' | 'viscosity' | 'stiffness' | 'lattice';

type ParamKey = keyof Params;

interface Params {
  weight: number;
  length: number;
  width: number;
  thickness: number;
  taper: number;
  asymmetry: number;
  curvature: number;
  irregularity: number;
  density: number;
  heterogeneity: number;
  viscosity: number;
  stiffness: number;
  recovery: number;
  friction: number;
  targetWeight: number;
  tolerance: number;
  cutAngle: number;
  knifeSpeed: number;
  bladeDepth: number;
  fingerRadius: number;
  fingerForce: number;
}

interface Poke {
  center: THREE.Vector2;
  radius: number;
  depth: number;
  push: THREE.Vector2;
  amplitude: number;
  held: boolean;
}

interface Candidate {
  angle: number;
  offset: number;
  targetSide: 'low' | 'high';
  portionA: number;
  portionB: number;
  error: number;
  pathLength: number;
  score: number;
  confidence: number;
  points: THREE.Vector3[];
}

interface LatticeData {
  rest: Float32Array;
  current: Float32Array;
  color: Float32Array;
  density: Float32Array;
  viscosity: Float32Array;
  stiffness: Float32Array;
  massRaw: Float32Array;
  mass: Float32Array;
  count: number;
}

interface SurfaceData {
  rest: Float32Array;
  current: Float32Array;
  color: Float32Array;
  density: Float32Array;
  viscosity: Float32Array;
  stiffness: Float32Array;
  count: number;
}

interface BackendPokePayload {
  center: [number, number];
  radius: number;
  depth: number;
  push: [number, number];
  amplitude: number;
  held: boolean;
}

interface BackendCandidatePayload {
  angle: number;
  offset: number;
  targetSide: 'low' | 'high';
  portionA: number;
  portionB: number;
  error: number;
  pathLength: number;
  score: number;
  confidence: number;
  points: number[][];
}

interface BackendSnapshot {
  params: Params;
  viewMode: ViewMode;
  seed: number;
  approved: boolean;
  cutDone: boolean;
  selectedCandidate: number;
  surface: {
    rest: number[][];
    current: number[][];
    density: number[];
    viscosity: number[];
    stiffness: number[];
    count: number;
  };
  lattice: {
    rest: number[][];
    current: number[][];
    density: number[];
    viscosity: number[];
    stiffness: number[];
    massRaw: number[];
    mass: number[];
    count: number;
  };
  candidates: BackendCandidatePayload[];
  summary: Record<string, unknown>;
}

const defaults: Params = {
  weight: 285,
  length: 182,
  width: 96,
  thickness: 31,
  taper: 0.58,
  asymmetry: 0.05,
  curvature: 0.035,
  irregularity: 0.035,
  density: 1055,
  heterogeneity: 0.18,
  viscosity: 0.58,
  stiffness: 31,
  recovery: 0.42,
  friction: 0.48,
  targetWeight: 160,
  tolerance: 5,
  cutAngle: 0,
  knifeSpeed: 48,
  bladeDepth: 36,
  fingerRadius: 24,
  fingerForce: 0.62,
};

const params: Params = { ...defaults };
const BOARD_TOP = 4;
const SURFACE_BASE = BOARD_TOP + 1.2;
const RING_COUNT = 70;
const RING_SEGMENTS = 38;

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App container was not found.');

app.innerHTML = `
<div class="app-shell">
  <header class="topbar">
    <div class="brand">
      <div class="logo-mark">FT</div>
      <div><h1>FilletTwin Apprentice</h1><p>Interactive protein digital twin • simulation prediction only</p></div>
    </div>
    <div class="top-actions">
      <button class="btn desktop-only" id="toggle-left">Parameters</button>
      <button class="btn desktop-only" id="toggle-right">Planner</button>
      <button class="btn secondary" id="camera-reset">Reset camera</button>
      <button class="btn primary" id="randomize-top">Randomize fillet</button>
    </div>
  </header>
  <main class="main-grid">
    <aside class="panel side-panel left-panel" id="left-panel">
      <div class="panel-header"><h2>Protein parameters</h2><span class="badge live">Interactive</span></div>
      <div class="panel-body">
        <div class="section">
          <div class="section-title"><strong>Shape & mass</strong><span>Deboned breast fillet</span></div>
          <div id="shape-controls"></div>
          <div class="btn-row"><button class="btn primary" id="randomize">Randomize</button><button class="btn" id="reset-shape">Defaults</button></div>
        </div>
        <div class="section">
          <div class="section-title"><strong>Material model</strong><span>Viscoelastic proxy</span></div>
          <div id="material-controls"></div>
        </div>
        <div class="section">
          <div class="section-title"><strong>Finger interaction</strong><span>Click + drag fillet</span></div>
          <div id="finger-controls"></div>
          <div class="btn-row"><button class="btn active" id="finger-mode">Finger tool: ON</button><button class="btn" id="reset-deform">Reset deformation</button></div>
        </div>
        <p class="notice">“Viscosity” is shown as a user-friendly viscoelastic damping proxy. This demo is a qualitative planning model, not a calibrated food-science instrument.</p>
      </div>
    </aside>

    <section class="panel viewport-panel">
      <div id="viewport"></div>
      <div class="viewport-hud">
        <div class="viewport-top">
          <div class="badge-row"><span class="badge live">Digital twin active</span><span class="badge" id="seed-badge">Seed 001</span><span class="badge" id="mode-badge">Surface</span></div>
          <div class="view-switcher" id="view-switcher">
            <button class="btn active" data-view="surface">Surface</button>
            <button class="btn" data-view="density">Density</button>
            <button class="btn" data-view="viscosity">Viscosity</button>
            <button class="btn" data-view="stiffness">Stiffness</button>
            <button class="btn" data-view="lattice">Lattice</button>
          </div>
        </div>
        <div class="legend-card" id="legend-card">
          <div class="legend-title"><span id="legend-name">Surface appearance</span><span id="legend-unit">qualitative</span></div>
          <div class="legend-bar"></div><div class="legend-scale"><span id="legend-min">low</span><span id="legend-mid">nominal</span><span id="legend-max">high</span></div>
        </div>
        <div class="bottom-console">
          <div class="metric"><label>Total mass</label><strong id="metric-mass">285 g</strong><small>particle-integrated</small></div>
          <div class="metric"><label>Portion A</label><strong id="metric-a">160.0 g</strong><small id="metric-a-sub">target 160 ± 5</small></div>
          <div class="metric"><label>Portion B</label><strong id="metric-b">125.0 g</strong><small>predicted remainder</small></div>
          <div class="metric"><label>Cut path</label><strong id="metric-path">94 mm</strong><small id="metric-path-sub">adaptive plane</small></div>
          <div class="metric"><label>Deformation</label><strong id="metric-deform">0.0 mm</strong><small id="metric-deform-sub">stable</small></div>
        </div>
      </div>
    </section>

    <aside class="panel side-panel right-panel" id="right-panel">
      <div class="panel-header"><h2>Cut planner</h2><span class="badge" id="plan-state">Ready</span></div>
      <div class="panel-body">
        <div class="section">
          <div class="section-title"><strong>Commercial target</strong><span>Mass-based partition</span></div>
          <div id="planner-controls"></div>
          <div class="btn-row"><button class="btn primary" id="optimize">Optimize path</button><button class="btn" id="approve">Approve selected</button></div>
        </div>
        <div class="section">
          <div class="section-title"><strong>Candidate paths</strong><span>Best three</span></div>
          <div id="candidate-list"></div>
        </div>
        <div class="section">
          <div class="section-title"><strong>Knife execution</strong><span>Visual simulation</span></div>
          <div id="knife-controls"></div>
          <div class="btn-row"><button class="btn warning" id="run-cut">Run knife</button><button class="btn" id="reset-cut">Reset cut</button></div>
          <button class="btn secondary" id="export-path" style="width:100%;margin-top:7px">Export approved path JSON</button>
        </div>
        <div class="section">
          <div class="status-box"><strong id="status-title">Apprentice proposal ready</strong><p id="status-copy">Drag the virtual finger across the fillet. The digital twin will deform, recalculate mass distribution, and update the cut proposal.</p></div>
        </div>
        <p class="notice">Approved paths are visualization outputs. Physical robot execution would require camera calibration, force limits, collision validation, and food-process testing.</p>
      </div>
    </aside>
  </main>
</div>
<div class="tooltip" id="tooltip"></div>
`;

const CONTROL_DEFS: Array<{ container: string; key: ParamKey; label: string; min: number; max: number; step: number; unit: string; rebuild?: boolean; replan?: boolean }> = [
  { container: 'shape-controls', key: 'weight', label: 'Total weight', min: 160, max: 420, step: 1, unit: 'g', rebuild: true },
  { container: 'shape-controls', key: 'length', label: 'Length', min: 135, max: 225, step: 1, unit: 'mm', rebuild: true },
  { container: 'shape-controls', key: 'width', label: 'Maximum width', min: 62, max: 125, step: 1, unit: 'mm', rebuild: true },
  { container: 'shape-controls', key: 'thickness', label: 'Maximum thickness', min: 17, max: 48, step: 1, unit: 'mm', rebuild: true },
  { container: 'shape-controls', key: 'taper', label: 'Tip taper', min: 0.25, max: 0.88, step: 0.01, unit: '', rebuild: true },
  { container: 'shape-controls', key: 'asymmetry', label: 'Left/right asymmetry', min: -0.18, max: 0.18, step: 0.01, unit: '', rebuild: true },
  { container: 'shape-controls', key: 'curvature', label: 'Centerline curvature', min: -0.12, max: 0.12, step: 0.005, unit: '', rebuild: true },
  { container: 'shape-controls', key: 'irregularity', label: 'Surface irregularity', min: 0, max: 0.10, step: 0.005, unit: '', rebuild: true },

  { container: 'material-controls', key: 'density', label: 'Mean density', min: 1015, max: 1090, step: 1, unit: 'kg/m³', rebuild: true },
  { container: 'material-controls', key: 'heterogeneity', label: 'Property variation', min: 0, max: 0.36, step: 0.01, unit: '', rebuild: true },
  { container: 'material-controls', key: 'viscosity', label: 'Viscoelastic damping', min: 0.12, max: 0.92, step: 0.01, unit: '', rebuild: true },
  { container: 'material-controls', key: 'stiffness', label: 'Relative stiffness', min: 12, max: 65, step: 1, unit: 'kPa', rebuild: true },
  { container: 'material-controls', key: 'recovery', label: 'Shape recovery', min: 0.08, max: 0.95, step: 0.01, unit: '', replan: true },
  { container: 'material-controls', key: 'friction', label: 'Board friction', min: 0.12, max: 0.86, step: 0.01, unit: '', replan: true },

  { container: 'finger-controls', key: 'fingerRadius', label: 'Finger contact radius', min: 12, max: 42, step: 1, unit: 'mm' },
  { container: 'finger-controls', key: 'fingerForce', label: 'Finger pressure', min: 0.15, max: 1, step: 0.01, unit: '' },

  { container: 'planner-controls', key: 'targetWeight', label: 'Target portion A', min: 70, max: 280, step: 1, unit: 'g', replan: true },
  { container: 'planner-controls', key: 'tolerance', label: 'Weight tolerance', min: 1, max: 20, step: 1, unit: 'g', replan: true },
  { container: 'planner-controls', key: 'cutAngle', label: 'Preferred cut angle', min: -30, max: 30, step: 1, unit: '°', replan: true },

  { container: 'knife-controls', key: 'knifeSpeed', label: 'Knife velocity', min: 15, max: 120, step: 1, unit: 'mm/s' },
  { container: 'knife-controls', key: 'bladeDepth', label: 'Blade penetration', min: 18, max: 55, step: 1, unit: 'mm' },
];

function formatValue(key: ParamKey, value: number, unit: string): string {
  const decimals = ['taper', 'asymmetry', 'curvature', 'irregularity', 'heterogeneity', 'viscosity', 'recovery', 'friction', 'fingerForce'].includes(key) ? 2 : 0;
  return `${value.toFixed(decimals)}${unit ? ` ${unit}` : ''}`;
}

for (const def of CONTROL_DEFS) {
  const container = document.getElementById(def.container);
  if (!container) continue;
  const wrapper = document.createElement('div');
  wrapper.className = 'control';
  wrapper.innerHTML = `
    <div class="control-top"><label for="param-${def.key}">${def.label}</label><output id="output-${def.key}">${formatValue(def.key, params[def.key], def.unit)}</output></div>
    <input id="param-${def.key}" type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${params[def.key]}" />
  `;
  container.appendChild(wrapper);
}

const viewport = document.querySelector<HTMLDivElement>('#viewport')!;
if (!viewport) throw new Error('Viewport container missing.');
const tooltip = document.querySelector<HTMLDivElement>('#tooltip')!;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111a18);
scene.fog = new THREE.FogExp2(0x111a18, 0.00175);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1300);
camera.position.set(185, -218, 172);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.localClippingEnabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 21);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 120;
controls.maxDistance = 520;
controls.maxPolarAngle = Math.PI * 0.49;

scene.add(new THREE.HemisphereLight(0xd9fff0, 0x1c2824, 1.15));
const keyLight = new THREE.DirectionalLight(0xfff4df, 3.5);
keyLight.position.set(-120, -100, 240);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.left = -220;
keyLight.shadow.camera.right = 220;
keyLight.shadow.camera.top = 180;
keyLight.shadow.camera.bottom = -180;
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x7fe4c7, 1.7);
fillLight.position.set(160, 80, 120);
scene.add(fillLight);
const rimLight = new THREE.PointLight(0xffba8a, 1.6, 440);
rimLight.position.set(-140, 130, 110);
scene.add(rimLight);

const table = new THREE.Mesh(
  new THREE.BoxGeometry(420, 300, 16),
  new THREE.MeshStandardMaterial({ color: 0x343c3a, roughness: 0.36, metalness: 0.42 })
);
table.position.z = -8;
table.receiveShadow = true;
scene.add(table);

const board = new THREE.Mesh(
  new THREE.BoxGeometry(300, 205, 8, 2, 2, 1),
  new THREE.MeshStandardMaterial({ color: 0xd8c6a6, roughness: 0.68, metalness: 0.02 })
);
board.position.z = 0;
board.receiveShadow = true;
board.castShadow = true;
scene.add(board);

const boardEdge = new THREE.LineSegments(
  new THREE.EdgesGeometry(board.geometry),
  new THREE.LineBasicMaterial({ color: 0x796b56, transparent: true, opacity: 0.55 })
);
boardEdge.position.copy(board.position);
scene.add(boardEdge);

const grid = new THREE.GridHelper(300, 20, 0x668077, 0x4a5b56);
grid.rotation.x = Math.PI / 2;
grid.position.z = BOARD_TOP + 0.15;
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.16;
scene.add(grid);

const filletGroup = new THREE.Group();
scene.add(filletGroup);
let filletMesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial>;
let filletWire: THREE.Mesh;
let latticePoints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
let splitLow: THREE.Mesh | null = null;
let splitHigh: THREE.Mesh | null = null;
let cutFace: THREE.Mesh | null = null;
let surfaceData: SurfaceData;
let latticeData: LatticeData;
let materialSeed = 1;
let viewMode: ViewMode = 'surface';
let fingerMode = true;
let pokes: Poke[] = [];
let activePoke: Poke | null = null;
let dragStart = new THREE.Vector2();
let currentCandidates: Candidate[] = [];
let selectedCandidate = 0;
let approved = false;
let cutDone = false;
let replanTimer: number | null = null;
let lastPlanTime = 0;

const surfaceMaterial = new THREE.MeshPhysicalMaterial({
  color: 0xeab4a2,
  vertexColors: true,
  roughness: 0.62,
  metalness: 0,
  transmission: 0.05,
  thickness: 1.4,
  clearcoat: 0.08,
  clearcoatRoughness: 0.7,
  side: THREE.DoubleSide,
});

const cutLineMaterial = new THREE.LineBasicMaterial({ color: 0xff664f, transparent: true, opacity: 0.98 });
let cutLine = new THREE.Line(new THREE.BufferGeometry(), cutLineMaterial);
cutLine.renderOrder = 8;
scene.add(cutLine);

const shadowLineMaterial = new THREE.LineBasicMaterial({ color: 0x240b07, transparent: true, opacity: 0.7 });
let cutShadow = new THREE.Line(new THREE.BufferGeometry(), shadowLineMaterial);
cutShadow.renderOrder = 7;
scene.add(cutShadow);

const knifeGroup = new THREE.Group();
const blade = new THREE.Mesh(
  new THREE.BoxGeometry(64, 2.6, 20),
  new THREE.MeshStandardMaterial({ color: 0xe3ebea, roughness: 0.2, metalness: 0.86 })
);
blade.position.x = -1;
blade.castShadow = true;
knifeGroup.add(blade);
const bladeEdge = new THREE.Mesh(
  new THREE.BoxGeometry(64, 1.2, 2.2),
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.12, metalness: 0.92 })
);
bladeEdge.position.set(-1, 0, -10.5);
knifeGroup.add(bladeEdge);
const handle = new THREE.Mesh(
  new THREE.CapsuleGeometry(6.5, 42, 6, 12),
  new THREE.MeshStandardMaterial({ color: 0x17201d, roughness: 0.62, metalness: 0.08 })
);
handle.rotation.z = Math.PI / 2;
handle.position.x = 50;
handle.castShadow = true;
knifeGroup.add(handle);
knifeGroup.visible = true;
scene.add(knifeGroup);

let knifeAnimation: { start: number; duration: number; curve: THREE.CatmullRomCurve3 } | null = null;

const fingerGroup = new THREE.Group();
const fingertip = new THREE.Mesh(
  new THREE.SphereGeometry(12, 28, 20),
  new THREE.MeshPhysicalMaterial({ color: 0xf0b994, roughness: 0.58, clearcoat: 0.05, transparent: true, opacity: 0.86 })
);
fingertip.scale.set(0.82, 0.82, 1.22);
fingertip.castShadow = true;
fingerGroup.add(fingertip);
const fingerShaft = new THREE.Mesh(
  new THREE.CapsuleGeometry(10, 34, 6, 14),
  new THREE.MeshPhysicalMaterial({ color: 0xeeb38e, roughness: 0.62, transparent: true, opacity: 0.82 })
);
fingerShaft.position.z = 27;
fingerGroup.add(fingerShaft);
fingerGroup.visible = false;
scene.add(fingerGroup);

const raycaster = new THREE.Raycaster();
const mouseNdc = new THREE.Vector2();
const boardPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -BOARD_TOP);

function seededNoise(x: number, y: number, z: number, seed = materialSeed): number {
  const s1 = Math.sin(x * 0.071 + y * 0.109 + z * 0.047 + seed * 1.713);
  const s2 = Math.sin(x * 0.023 - y * 0.057 + z * 0.083 + seed * 4.311);
  const s3 = Math.cos(x * 0.137 + y * 0.031 - z * 0.059 + seed * 0.997);
  return (s1 * 0.5 + s2 * 0.3 + s3 * 0.2);
}

function clamp01(v: number): number { return Math.max(0, Math.min(1, v)); }
function lerp(a: number, b: number, t: number): number { return a + (b - a) * t; }

function profileAt(x: number): { halfWidth: number; thickness: number; centerY: number; u: number } {
  const u = clamp01((x + params.length / 2) / params.length);
  const endEnvelope = Math.pow(Math.max(0, Math.sin(Math.PI * (0.035 + 0.93 * u))), 0.62);
  const tipGrowth = lerp(0.46, 1.0, Math.pow(u, Math.max(0.35, params.taper)));
  const shoulder = 1 + 0.09 * Math.exp(-Math.pow((u - 0.72) / 0.22, 2));
  const wave = 1 + params.irregularity * (0.55 * Math.sin(u * 17.3 + materialSeed) + 0.25 * Math.sin(u * 31.7 + 0.8));
  const halfWidth = Math.max(0.8, params.width * 0.5 * endEnvelope * tipGrowth * shoulder * wave);
  const thickEnvelope = Math.pow(Math.max(0, Math.sin(Math.PI * (0.015 + 0.95 * u))), 0.78);
  const thickGrowth = lerp(0.55, 1.0, Math.pow(u, 0.62));
  const thickness = Math.max(1.2, params.thickness * thickEnvelope * thickGrowth * (1 + params.irregularity * 0.18 * Math.sin(u * 21 + 1.7)));
  const centerY = params.curvature * params.width * Math.sin((u - 0.1) * Math.PI) + params.asymmetry * params.width * (u - 0.5) * 0.52;
  return { halfWidth, thickness, centerY, u };
}

function topHeightAt(x: number, y: number): number {
  const p = profileAt(x);
  const q = Math.abs(y - p.centerY) / Math.max(0.1, p.halfWidth);
  if (q >= 1) return SURFACE_BASE;
  const dome = Math.pow(Math.max(0, 1 - q * q), 0.66);
  const sideBias = 1 + params.asymmetry * 0.18 * ((y - p.centerY) / Math.max(p.halfWidth, 1));
  return SURFACE_BASE + p.thickness * dome * sideBias;
}

function localMaterial(x: number, y: number, z: number): { density: number; viscosity: number; stiffness: number } {
  const p = profileAt(x);
  const thickNorm = clamp01((z - SURFACE_BASE) / Math.max(1, p.thickness));
  const longitudinal = p.u;
  const n1 = seededNoise(x, y, z, materialSeed);
  const n2 = seededNoise(x * 1.4 + 13, y * 0.8 - 9, z * 1.1, materialSeed + 7);
  const n3 = seededNoise(x * 0.65 - 21, y * 1.3 + 4, z * 1.6, materialSeed + 13);
  const density = params.density * (1 + params.heterogeneity * 0.11 * n1 + 0.018 * thickNorm + 0.014 * longitudinal);
  const viscosity = clamp01(params.viscosity + params.heterogeneity * 0.32 * n2 + 0.06 * (1 - thickNorm));
  const stiffness = Math.max(4, params.stiffness * (1 + params.heterogeneity * 0.48 * n3 + 0.12 * longitudinal));
  return { density, viscosity, stiffness };
}

function applyPokes(restX: number, restY: number, restZ: number, out: THREE.Vector3): THREE.Vector3 {
  out.set(restX, restY, restZ);
  const p = profileAt(restX);
  const top = Math.max(SURFACE_BASE + 1, topHeightAt(restX, restY));
  const zNorm = clamp01((restZ - SURFACE_BASE) / Math.max(1, top - SURFACE_BASE));
  for (const poke of pokes) {
    if (poke.amplitude < 0.001) continue;
    const dx = restX - poke.center.x;
    const dy = restY - poke.center.y;
    const r2 = dx * dx + dy * dy;
    const sigma2 = poke.radius * poke.radius;
    const w = Math.exp(-r2 / (2 * sigma2));
    const r = Math.sqrt(r2);
    const ring = Math.exp(-Math.pow((r - poke.radius * 0.72) / Math.max(2, poke.radius * 0.34), 2));
    const body = 0.28 + 0.72 * zNorm;
    const slip = 0.22 + 0.78 * (1 - params.friction);
    out.x += poke.push.x * w * body * poke.amplitude * slip;
    out.y += poke.push.y * w * body * poke.amplitude * slip;
    out.z -= poke.depth * w * body * poke.amplitude;
    out.z += poke.depth * 0.19 * ring * body * poke.amplitude;
    out.y += params.asymmetry * poke.depth * 0.06 * w * poke.amplitude;
  }
  out.z = Math.max(SURFACE_BASE - 0.2, out.z);
  return out;
}

function createSurfaceGeometry(): { geometry: THREE.BufferGeometry; data: SurfaceData } {
  const positions: number[] = [];
  const colors: number[] = [];
  const densities: number[] = [];
  const viscosities: number[] = [];
  const stiffnesses: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < RING_COUNT; i++) {
    const u = i / (RING_COUNT - 1);
    const x = -params.length / 2 + u * params.length;
    const p = profileAt(x);
    for (let j = 0; j < RING_SEGMENTS; j++) {
      const theta = (j / RING_SEGMENTS) * Math.PI * 2;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      const lateralAsym = 1 + params.asymmetry * 0.14 * Math.sign(sinT) * (0.3 + 0.7 * p.u);
      const y = p.centerY + p.halfWidth * sinT * lateralAsym;
      const vertical01 = Math.pow((cosT + 1) * 0.5, 0.70);
      const topVariation = 1 + params.irregularity * 0.08 * seededNoise(x, y, vertical01 * p.thickness, materialSeed + 22);
      const z = SURFACE_BASE + p.thickness * vertical01 * topVariation;
      positions.push(x, y, z);
      const mat = localMaterial(x, y, z);
      densities.push(mat.density);
      viscosities.push(mat.viscosity);
      stiffnesses.push(mat.stiffness);
      colors.push(1, 1, 1);
    }
  }

  for (let i = 0; i < RING_COUNT - 1; i++) {
    for (let j = 0; j < RING_SEGMENTS; j++) {
      const nextJ = (j + 1) % RING_SEGMENTS;
      const a = i * RING_SEGMENTS + j;
      const b = (i + 1) * RING_SEGMENTS + j;
      const c = (i + 1) * RING_SEGMENTS + nextJ;
      const d = i * RING_SEGMENTS + nextJ;
      indices.push(a, b, d, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  const rest = new Float32Array(positions);
  const current = new Float32Array(positions);
  const color = new Float32Array(colors);
  geometry.setAttribute('position', new THREE.BufferAttribute(current, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(color, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  return {
    geometry,
    data: {
      rest,
      current,
      color,
      density: new Float32Array(densities),
      viscosity: new Float32Array(viscosities),
      stiffness: new Float32Array(stiffnesses),
      count: positions.length / 3,
    },
  };
}

function createLatticeGeometry(): { geometry: THREE.BufferGeometry; data: LatticeData } {
  const positions: number[] = [];
  const colors: number[] = [];
  const densities: number[] = [];
  const viscosities: number[] = [];
  const stiffnesses: number[] = [];
  const rawMass: number[] = [];
  const nx = 27;
  const ny = 15;
  const nz = 9;
  const dx = params.length / (nx - 1);

  for (let ix = 0; ix < nx; ix++) {
    const u = ix / (nx - 1);
    const x = -params.length / 2 + u * params.length;
    const p = profileAt(x);
    const dy = (p.halfWidth * 2) / Math.max(1, ny - 1);
    for (let iy = 0; iy < ny; iy++) {
      const yNorm = -1 + (iy / (ny - 1)) * 2;
      const y = p.centerY + yNorm * p.halfWidth;
      const top = topHeightAt(x, y);
      const localH = Math.max(0, top - SURFACE_BASE);
      if (localH < 0.6) continue;
      const dz = localH / Math.max(1, nz - 1);
      for (let iz = 0; iz < nz; iz++) {
        const z = SURFACE_BASE + (iz / (nz - 1)) * localH;
        const edgeSkip = Math.abs(yNorm) > 0.96 && iz > nz * 0.5;
        if (edgeSkip) continue;
        positions.push(x, y, z);
        const mat = localMaterial(x, y, z);
        densities.push(mat.density);
        viscosities.push(mat.viscosity);
        stiffnesses.push(mat.stiffness);
        const volumeMm3 = Math.max(0.1, dx * Math.max(0.1, dy) * Math.max(0.1, dz));
        rawMass.push(mat.density * volumeMm3);
        colors.push(1, 1, 1);
      }
    }
  }

  const rest = new Float32Array(positions);
  const current = new Float32Array(positions);
  const color = new Float32Array(colors);
  const massRaw = new Float32Array(rawMass);
  const mass = new Float32Array(rawMass.length);
  const rawTotal = massRaw.reduce((a, b) => a + b, 0);
  const scale = params.weight / Math.max(1e-9, rawTotal);
  for (let i = 0; i < mass.length; i++) mass[i] = massRaw[i] * scale;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(current, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(color, 3));

  return {
    geometry,
    data: {
      rest,
      current,
      color,
      density: new Float32Array(densities),
      viscosity: new Float32Array(viscosities),
      stiffness: new Float32Array(stiffnesses),
      massRaw,
      mass,
      count: positions.length / 3,
    },
  };
}

const colorLow = new THREE.Color(0x2d7fff);
const colorMidLow = new THREE.Color(0x53d6d0);
const colorMid = new THREE.Color(0xb9eb70);
const colorHigh = new THREE.Color(0xffd65a);
const colorMax = new THREE.Color(0xff674d);
const colorSurface = new THREE.Color(0xe9ad9b);
const colorSurfaceDark = new THREE.Color(0xc77d72);
const tmpColor = new THREE.Color();
const tmpVec = new THREE.Vector3();
// The production UI is served by the Python app, so API calls stay on the
// same origin. Vite proxies this path during frontend development.
const pythonBackendUrl = '';
let backendSyncTimer: number | null = null;
let backendSyncToken = 0;

function heatColor(t: number, out: THREE.Color): THREE.Color {
  const v = clamp01(t);
  if (v < 0.25) return out.copy(colorLow).lerp(colorMidLow, v / 0.25);
  if (v < 0.5) return out.copy(colorMidLow).lerp(colorMid, (v - 0.25) / 0.25);
  if (v < 0.75) return out.copy(colorMid).lerp(colorHigh, (v - 0.5) / 0.25);
  return out.copy(colorHigh).lerp(colorMax, (v - 0.75) / 0.25);
}

function getFieldRange(data: Float32Array): { min: number; max: number; mean: number } {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of data) { min = Math.min(min, v); max = Math.max(max, v); sum += v; }
  return { min, max, mean: sum / Math.max(1, data.length) };
}

function updateColors(): void {
  const setColors = (data: SurfaceData | LatticeData) => {
    let field: Float32Array | null = null;
    if (viewMode === 'density') field = data.density;
    if (viewMode === 'viscosity') field = data.viscosity;
    if (viewMode === 'stiffness') field = data.stiffness;
    const range = field ? getFieldRange(field) : null;
    for (let i = 0; i < data.count; i++) {
      if (field && range) {
        const t = (field[i] - range.min) / Math.max(1e-6, range.max - range.min);
        heatColor(t, tmpColor);
      } else {
        const n = seededNoise(data.rest[i * 3], data.rest[i * 3 + 1], data.rest[i * 3 + 2], materialSeed + 31);
        tmpColor.copy(colorSurfaceDark).lerp(colorSurface, 0.54 + 0.28 * n);
      }
      data.color[i * 3] = tmpColor.r;
      data.color[i * 3 + 1] = tmpColor.g;
      data.color[i * 3 + 2] = tmpColor.b;
    }
  };
  setColors(surfaceData);
  setColors(latticeData);
  (filletMesh.geometry.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
  (latticePoints.geometry.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;

  const legend = {
    surface: ['Surface appearance', 'qualitative', 'soft pink', 'natural', 'highlight'],
    density: ['Density field', 'kg/m³', '', '', ''],
    viscosity: ['Viscoelastic damping', '0–1 proxy', '', '', ''],
    stiffness: ['Relative stiffness', 'kPa proxy', '', '', ''],
    lattice: ['Particle lattice', 'mass nodes', 'sparse', 'volume', 'dense'],
  }[viewMode];
  document.querySelector('#legend-name')!.textContent = legend[0];
  document.querySelector('#legend-unit')!.textContent = legend[1];
  let field: Float32Array | null = null;
  if (viewMode === 'density') field = latticeData.density;
  if (viewMode === 'viscosity') field = latticeData.viscosity;
  if (viewMode === 'stiffness') field = latticeData.stiffness;
  if (field) {
    const range = getFieldRange(field);
    const unit = viewMode === 'density' ? '' : '';
    document.querySelector('#legend-min')!.textContent = `${range.min.toFixed(viewMode === 'density' ? 0 : 2)}${unit}`;
    document.querySelector('#legend-mid')!.textContent = `${range.mean.toFixed(viewMode === 'density' ? 0 : 2)}${unit}`;
    document.querySelector('#legend-max')!.textContent = `${range.max.toFixed(viewMode === 'density' ? 0 : 2)}${unit}`;
  } else {
    document.querySelector('#legend-min')!.textContent = legend[2];
    document.querySelector('#legend-mid')!.textContent = legend[3];
    document.querySelector('#legend-max')!.textContent = legend[4];
  }
  filletMesh.material.opacity = viewMode === 'lattice' ? 0.18 : viewMode === 'surface' ? 1 : 0.52;
  filletMesh.material.transparent = viewMode === 'lattice' || viewMode !== 'surface';
  latticePoints.visible = viewMode === 'lattice' || viewMode === 'density' || viewMode === 'viscosity' || viewMode === 'stiffness';
  filletWire.visible = viewMode === 'lattice';
}

function clearFilletObjects(): void {
  while (filletGroup.children.length) {
    const child = filletGroup.children.pop();
    if (!child) continue;
    if ('geometry' in child && child.geometry instanceof THREE.BufferGeometry) child.geometry.dispose();
    if ('material' in child) {
      const material = child.material as THREE.Material | THREE.Material[];
      if (Array.isArray(material)) material.forEach((m) => m.dispose());
      else if (material !== surfaceMaterial) material.dispose();
    }
  }
}

function rebuildFillet(): void {
  stopKnife();
  cutDone = false;
  approved = false;
  clearSplitPreview();
  pokes = [];
  activePoke = null;
  clearFilletObjects();
  const surface = createSurfaceGeometry();
  surfaceData = surface.data;
  filletMesh = new THREE.Mesh(surface.geometry, surfaceMaterial);
  filletMesh.castShadow = true;
  filletMesh.receiveShadow = true;
  filletMesh.userData.kind = 'fillet';
  filletGroup.add(filletMesh);

  filletWire = new THREE.Mesh(
    surface.geometry,
    new THREE.MeshBasicMaterial({ color: 0xf2ffe8, wireframe: true, transparent: true, opacity: 0.11, depthWrite: false })
  );
  filletWire.renderOrder = 4;
  filletGroup.add(filletWire);

  const lattice = createLatticeGeometry();
  latticeData = lattice.data;
  latticePoints = new THREE.Points(
    lattice.geometry,
    new THREE.PointsMaterial({ size: 2.1, vertexColors: true, transparent: true, opacity: 0.86, sizeAttenuation: true, depthWrite: false })
  );
  latticePoints.renderOrder = 5;
  filletGroup.add(latticePoints);

  updateGeometryPositions(true);
  updateColors();
  optimizePaths(false);
  updateMetrics();
  scheduleBackendSync('rebuild', 80);
}

function updateGeometryPositions(recomputeNormals = false): void {
  for (let i = 0; i < surfaceData.count; i++) {
    const ix = i * 3;
    applyPokes(surfaceData.rest[ix], surfaceData.rest[ix + 1], surfaceData.rest[ix + 2], tmpVec);
    surfaceData.current[ix] = tmpVec.x;
    surfaceData.current[ix + 1] = tmpVec.y;
    surfaceData.current[ix + 2] = tmpVec.z;
  }
  const surfacePosition = filletMesh.geometry.getAttribute('position') as THREE.BufferAttribute;
  surfacePosition.needsUpdate = true;
  if (recomputeNormals) {
    filletMesh.geometry.computeVertexNormals();
    filletMesh.geometry.computeBoundingSphere();
  }

  for (let i = 0; i < latticeData.count; i++) {
    const ix = i * 3;
    applyPokes(latticeData.rest[ix], latticeData.rest[ix + 1], latticeData.rest[ix + 2], tmpVec);
    latticeData.current[ix] = tmpVec.x;
    latticeData.current[ix + 1] = tmpVec.y;
    latticeData.current[ix + 2] = tmpVec.z;
  }
  (latticePoints.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
}

function computePathPoints(angle: number, offset: number): THREE.Vector3[] {
  const normal = new THREE.Vector2(Math.cos(angle), -Math.sin(angle));
  const tangent = new THREE.Vector2(Math.sin(angle), Math.cos(angle));
  let minT = Infinity;
  let maxT = -Infinity;
  const band = 5.5;
  for (let i = 0; i < latticeData.count; i++) {
    const x = latticeData.current[i * 3];
    const y = latticeData.current[i * 3 + 1];
    const p = x * normal.x + y * normal.y;
    if (Math.abs(p - offset) < band) {
      const t = x * tangent.x + y * tangent.y;
      minT = Math.min(minT, t);
      maxT = Math.max(maxT, t);
    }
  }
  if (!Number.isFinite(minT) || !Number.isFinite(maxT)) {
    minT = -params.width * 0.5;
    maxT = params.width * 0.5;
  }
  minT -= 2;
  maxT += 2;
  const points: THREE.Vector3[] = [];
  const samples = 34;
  for (let i = 0; i < samples; i++) {
    const t = lerp(minT, maxT, i / (samples - 1));
    const x = normal.x * offset + tangent.x * t;
    const y = normal.y * offset + tangent.y * t;
    let bestZ = SURFACE_BASE + 2;
    let bestD2 = Infinity;
    for (let v = 0; v < surfaceData.count; v++) {
      const sx = surfaceData.current[v * 3];
      const sy = surfaceData.current[v * 3 + 1];
      const dx = sx - x;
      const dy = sy - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestZ = surfaceData.current[v * 3 + 2];
      }
    }
    points.push(new THREE.Vector3(x, y, bestZ + 1.9));
  }
  return points;
}

function pathLength(points: THREE.Vector3[]): number {
  let length = 0;
  for (let i = 1; i < points.length; i++) length += points[i].distanceTo(points[i - 1]);
  return length;
}

function flattenPoints(points: number[][]): Float32Array {
  const flat = new Float32Array(points.length * 3);
  points.forEach((point, index) => {
    flat[index * 3] = point[0];
    flat[index * 3 + 1] = point[1];
    flat[index * 3 + 2] = point[2];
  });
  return flat;
}

function buildGeometryFromSnapshot(current: number[][], colors: Float32Array, surfaceTopology = false): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(flattenPoints(current), 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  if (surfaceTopology) {
    const indices: number[] = [];
    for (let ring = 0; ring < RING_COUNT - 1; ring++) {
      for (let segment = 0; segment < RING_SEGMENTS; segment++) {
        const nextSegment = (segment + 1) % RING_SEGMENTS;
        const a = ring * RING_SEGMENTS + segment;
        const b = (ring + 1) * RING_SEGMENTS + segment;
        const c = (ring + 1) * RING_SEGMENTS + nextSegment;
        const d = ring * RING_SEGMENTS + nextSegment;
        indices.push(a, b, d, b, c, d);
      }
    }
    geometry.setIndex(indices);
  }
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function snapshotCandidateToLocal(candidate: BackendCandidatePayload): Candidate {
  return {
    angle: candidate.angle,
    offset: candidate.offset,
    targetSide: candidate.targetSide,
    portionA: candidate.portionA,
    portionB: candidate.portionB,
    error: candidate.error,
    pathLength: candidate.pathLength,
    score: candidate.score,
    confidence: candidate.confidence,
    points: candidate.points.map((point) => new THREE.Vector3(point[0], point[1], point[2])),
  };
}

function applyBackendSnapshot(snapshot: BackendSnapshot): void {
  Object.assign(params, snapshot.params);
  materialSeed = snapshot.seed;
  approved = snapshot.approved;
  cutDone = snapshot.cutDone;
  selectedCandidate = Math.min(snapshot.selectedCandidate, Math.max(0, snapshot.candidates.length - 1));
  currentCandidates = snapshot.candidates.map(snapshotCandidateToLocal);

  surfaceData = {
    rest: flattenPoints(snapshot.surface.rest),
    current: flattenPoints(snapshot.surface.current),
    color: new Float32Array(snapshot.surface.count * 3),
    density: new Float32Array(snapshot.surface.density),
    viscosity: new Float32Array(snapshot.surface.viscosity),
    stiffness: new Float32Array(snapshot.surface.stiffness),
    count: snapshot.surface.count,
  };
  latticeData = {
    rest: flattenPoints(snapshot.lattice.rest),
    current: flattenPoints(snapshot.lattice.current),
    color: new Float32Array(snapshot.lattice.count * 3),
    density: new Float32Array(snapshot.lattice.density),
    viscosity: new Float32Array(snapshot.lattice.viscosity),
    stiffness: new Float32Array(snapshot.lattice.stiffness),
    massRaw: new Float32Array(snapshot.lattice.massRaw),
    mass: new Float32Array(snapshot.lattice.mass),
    count: snapshot.lattice.count,
  };

  updateColors();
  const surfaceGeometry = buildGeometryFromSnapshot(snapshot.surface.current, surfaceData.color, true);
  if (filletMesh) {
    filletMesh.geometry.dispose();
    filletMesh.geometry = surfaceGeometry;
  }
  if (filletWire) {
    filletWire.geometry.dispose();
    filletWire.geometry = surfaceGeometry;
  }

  const latticeGeometry = buildGeometryFromSnapshot(snapshot.lattice.current, latticeData.color);
  if (latticePoints) {
    latticePoints.geometry.dispose();
    latticePoints.geometry = latticeGeometry;
  }

  syncControls();
  document.querySelector('#seed-badge')!.textContent = `Seed ${String(materialSeed).padStart(3, '0')}`;
  document.querySelector('#plan-state')!.textContent = cutDone ? 'Cut complete' : approved ? 'Human approved' : 'Ready';
  updateColors();
  updateCandidateList();
  updateCutLine();
  updateMetrics();
}

function buildBackendPayload(): { viewMode: ViewMode; seed: number; approved: boolean; cutDone: boolean; selectedCandidate: number; device: string; params: Params; pokes: BackendPokePayload[] } {
  return {
    viewMode,
    seed: materialSeed,
    approved,
    cutDone,
    selectedCandidate,
    device: 'cpu',
    params: { ...params },
    pokes: pokes.map((poke) => ({
      center: [poke.center.x, poke.center.y],
      radius: poke.radius,
      depth: poke.depth,
      push: [poke.push.x, poke.push.y],
      amplitude: poke.amplitude,
      held: poke.held,
    })),
  };
}

async function syncWithPythonBackend(reason: string): Promise<void> {
  const token = ++backendSyncToken;
  try {
    const response = await fetch(`${pythonBackendUrl}/api/snapshot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildBackendPayload()),
    });
    if (!response.ok) return;
    const snapshot = (await response.json()) as BackendSnapshot;
    if (token !== backendSyncToken) return;
    applyBackendSnapshot(snapshot);
    setStatus('Python backend synchronized', `Simulation state refreshed from PyTorch backend after ${reason}.`);
  } catch {
    return;
  }
}

function scheduleBackendSync(reason: string, delay = 120): void {
  if (backendSyncTimer !== null) window.clearTimeout(backendSyncTimer);
  backendSyncTimer = window.setTimeout(() => {
    backendSyncTimer = null;
    void syncWithPythonBackend(reason);
  }, delay);
}

function optimizePaths(updateStatus = true): void {
  if (!latticeData || latticeData.count === 0) return;
  const total = latticeData.mass.reduce((a, b) => a + b, 0);
  const target = Math.min(params.targetWeight, total - 20);
  const candidates: Candidate[] = [];
  const preferred = THREE.MathUtils.degToRad(params.cutAngle);

  for (let degDelta = -24; degDelta <= 24; degDelta += 4) {
    const angle = preferred + THREE.MathUtils.degToRad(degDelta);
    const normal = new THREE.Vector2(Math.cos(angle), -Math.sin(angle));
    const projected: Array<{ p: number; m: number }> = [];
    for (let i = 0; i < latticeData.count; i++) {
      projected.push({
        p: latticeData.current[i * 3] * normal.x + latticeData.current[i * 3 + 1] * normal.y,
        m: latticeData.mass[i],
      });
    }
    projected.sort((a, b) => a.p - b.p);
    let cumulative = 0;
    let bestLow = { error: Infinity, offset: 0, mass: 0 };
    let bestHigh = { error: Infinity, offset: 0, mass: 0 };
    for (let i = 0; i < projected.length; i++) {
      cumulative += projected[i].m;
      const lowErr = Math.abs(cumulative - target);
      if (lowErr < bestLow.error) bestLow = { error: lowErr, offset: projected[i].p, mass: cumulative };
      const highMass = total - cumulative;
      const highErr = Math.abs(highMass - target);
      if (highErr < bestHigh.error) bestHigh = { error: highErr, offset: projected[i].p, mass: highMass };
    }

    for (const sideCandidate of [
      { side: 'low' as const, value: bestLow },
      { side: 'high' as const, value: bestHigh },
    ]) {
      const points = computePathPoints(angle, sideCandidate.value.offset);
      const length = pathLength(points);
      const anglePenalty = Math.abs(degDelta) * 0.022;
      const lengthPenalty = Math.max(0, length - params.width * 0.75) * 0.006;
      const uncertainty = params.heterogeneity * 2.6 + Math.min(0.8, pokes.length * 0.13);
      const score = sideCandidate.value.error + anglePenalty + lengthPenalty + uncertainty;
      const confidence = clamp01(0.94 - params.heterogeneity * 0.42 - sideCandidate.value.error / Math.max(30, target) - pokes.length * 0.018);
      candidates.push({
        angle,
        offset: sideCandidate.value.offset,
        targetSide: sideCandidate.side,
        portionA: sideCandidate.value.mass,
        portionB: total - sideCandidate.value.mass,
        error: sideCandidate.value.error,
        pathLength: length,
        score,
        confidence,
        points,
      });
    }
  }

  candidates.sort((a, b) => a.score - b.score);
  const diverse: Candidate[] = [];
  for (const candidate of candidates) {
    if (diverse.every((d) => Math.abs(d.angle - candidate.angle) > THREE.MathUtils.degToRad(3) || d.targetSide !== candidate.targetSide)) {
      diverse.push(candidate);
    }
    if (diverse.length >= 3) break;
  }
  currentCandidates = diverse.length ? diverse : candidates.slice(0, 3);
  selectedCandidate = Math.min(selectedCandidate, currentCandidates.length - 1);
  approved = false;
  updateCandidateList();
  updateCutLine();
  updateMetrics();
  if (updateStatus) {
    const elapsed = performance.now() - lastPlanTime;
    document.querySelector('#plan-state')!.textContent = elapsed < 700 ? 'Replanned live' : 'Optimized';
    setStatus('Path recalculated', `The apprentice repartitioned ${latticeData.count.toLocaleString()} material nodes after the shape or target changed.`);
  }
  lastPlanTime = performance.now();
  scheduleBackendSync('optimize', 120);
}

function updateCandidateList(): void {
  const list = document.querySelector<HTMLDivElement>('#candidate-list')!;
  list.innerHTML = '';
  currentCandidates.forEach((candidate, index) => {
    const card = document.createElement('div');
    card.className = `candidate ${index === selectedCandidate ? 'selected' : ''}`;
    card.innerHTML = `
      <div class="candidate-head"><strong>Candidate ${String.fromCharCode(65 + index)}</strong><span class="candidate-score">${(candidate.confidence * 100).toFixed(0)}% confidence</span></div>
      <div class="candidate-grid">
        <div>Portion A <b>${candidate.portionA.toFixed(1)} g</b></div>
        <div>Error <b>${candidate.error.toFixed(1)} g</b></div>
        <div>Remainder <b>${candidate.portionB.toFixed(1)} g</b></div>
        <div>Cut length <b>${candidate.pathLength.toFixed(0)} mm</b></div>
      </div>
    `;
    card.addEventListener('click', () => {
      selectedCandidate = index;
      approved = false;
      updateCandidateList();
      updateCutLine();
      updateMetrics();
      setStatus(`Candidate ${String.fromCharCode(65 + index)} selected`, candidate.error <= params.tolerance ? 'This proposal is inside the target tolerance.' : 'This is the closest feasible partition under the current shape and cut-angle constraints.');
    });
    list.appendChild(card);
  });
}

function updateCutLine(): void {
  const candidate = currentCandidates[selectedCandidate];
  if (!candidate) return;
  candidate.points = computePathPoints(candidate.angle, candidate.offset);
  candidate.pathLength = pathLength(candidate.points);
  cutLine.geometry.dispose();
  cutShadow.geometry.dispose();
  cutLine.geometry = new THREE.BufferGeometry().setFromPoints(candidate.points);
  cutShadow.geometry = new THREE.BufferGeometry().setFromPoints(candidate.points.map((p) => new THREE.Vector3(p.x, p.y, BOARD_TOP + 0.5)));
  cutLineMaterial.color.set(approved ? 0xbdf47c : 0xff664f);
  positionKnifeIdle();
}

function positionKnifeIdle(): void {
  const candidate = currentCandidates[selectedCandidate];
  if (!candidate || knifeAnimation) return;
  const curve = new THREE.CatmullRomCurve3(candidate.points);
  const point = curve.getPoint(0.08);
  const tangent = curve.getTangent(0.08).normalize();
  knifeGroup.position.copy(point).add(new THREE.Vector3(0, 0, 52));
  knifeGroup.rotation.set(0, 0, Math.atan2(tangent.y, tangent.x));
}

function updateMetrics(): void {
  const candidate = currentCandidates[selectedCandidate];
  const deformation = pokes.reduce((sum, p) => sum + p.depth * p.amplitude, 0);
  document.querySelector('#metric-mass')!.textContent = `${params.weight.toFixed(0)} g`;
  document.querySelector('#metric-a')!.textContent = `${candidate ? candidate.portionA.toFixed(1) : '--'} g`;
  document.querySelector('#metric-b')!.textContent = `${candidate ? candidate.portionB.toFixed(1) : '--'} g`;
  document.querySelector('#metric-path')!.textContent = `${candidate ? candidate.pathLength.toFixed(0) : '--'} mm`;
  document.querySelector('#metric-a-sub')!.textContent = `target ${params.targetWeight.toFixed(0)} ± ${params.tolerance.toFixed(0)}`;
  document.querySelector('#metric-path-sub')!.textContent = approved ? 'human approved' : 'adaptive proposal';
  document.querySelector('#metric-deform')!.textContent = `${deformation.toFixed(1)} mm`;
  document.querySelector('#metric-deform-sub')!.textContent = activePoke ? 'finger engaged' : deformation > 0.2 ? 'recovering' : 'stable';
}

function setStatus(title: string, copy: string): void {
  document.querySelector('#status-title')!.textContent = title;
  document.querySelector('#status-copy')!.textContent = copy;
}

function scheduleReplan(delay = 90): void {
  if (replanTimer !== null) window.clearTimeout(replanTimer);
  replanTimer = window.setTimeout(() => {
    replanTimer = null;
    optimizePaths(true);
  }, delay);
}

function clearSplitPreview(): void {
  if (splitLow) scene.remove(splitLow);
  if (splitHigh) scene.remove(splitHigh);
  if (cutFace) scene.remove(cutFace);
  splitLow = null;
  splitHigh = null;
  cutFace = null;
  if (filletMesh) filletMesh.visible = true;
}

function createSplitPreview(separation = 8): void {
  clearSplitPreview();
  const candidate = currentCandidates[selectedCandidate];
  if (!candidate) return;
  filletMesh.visible = false;
  const normal3 = new THREE.Vector3(Math.cos(candidate.angle), -Math.sin(candidate.angle), 0).normalize();
  const planeLow = new THREE.Plane(normal3.clone(), -candidate.offset);
  const planeHigh = new THREE.Plane(normal3.clone().negate(), candidate.offset);
  const lowMat = surfaceMaterial.clone();
  const highMat = surfaceMaterial.clone();
  lowMat.clippingPlanes = [planeLow];
  highMat.clippingPlanes = [planeHigh];
  splitLow = new THREE.Mesh(filletMesh.geometry, lowMat);
  splitHigh = new THREE.Mesh(filletMesh.geometry, highMat);
  splitLow.castShadow = splitHigh.castShadow = true;
  splitLow.position.addScaledVector(normal3, -separation);
  splitHigh.position.addScaledVector(normal3, separation);
  scene.add(splitLow, splitHigh);

  const candidatePoints = candidate.points;
  const center = candidatePoints[Math.floor(candidatePoints.length / 2)] ?? new THREE.Vector3();
  const faceLength = Math.max(20, candidate.pathLength);
  cutFace = new THREE.Mesh(
    new THREE.PlaneGeometry(faceLength, Math.min(params.bladeDepth, params.thickness + 8)),
    new THREE.MeshStandardMaterial({ color: 0x9e4037, roughness: 0.78, metalness: 0, side: THREE.DoubleSide, transparent: true, opacity: 0.72 })
  );
  cutFace.position.set(center.x, center.y, SURFACE_BASE + params.thickness * 0.44);
  cutFace.rotation.set(Math.PI / 2, candidate.angle, 0);
  scene.add(cutFace);
}

function setSplitSeparation(separation: number): void {
  const candidate = currentCandidates[selectedCandidate];
  if (!candidate || !splitLow || !splitHigh) return;
  const normal = new THREE.Vector3(Math.cos(candidate.angle), -Math.sin(candidate.angle), 0).normalize();
  splitLow.position.copy(normal).multiplyScalar(-separation);
  splitHigh.position.copy(normal).multiplyScalar(separation);
  if (cutFace) (cutFace.material as THREE.MeshStandardMaterial).opacity = 0.18 + Math.min(1, separation / 8) * 0.54;
}

function startKnife(): void {
  if (!currentCandidates[selectedCandidate] || knifeAnimation) return;
  clearSplitPreview();
  cutDone = false;
  const candidate = currentCandidates[selectedCandidate];
  const curve = new THREE.CatmullRomCurve3(candidate.points);
  const travelSeconds = candidate.pathLength / Math.max(8, params.knifeSpeed);
  knifeAnimation = { start: performance.now(), duration: (1.15 + travelSeconds) * 1000, curve };
  controls.enabled = false;
  document.querySelector('#plan-state')!.textContent = 'Executing';
  setStatus('Knife simulation running', `Following the ${candidate.pathLength.toFixed(0)} mm approved path at ${params.knifeSpeed.toFixed(0)} mm/s.`);
}

function stopKnife(): void {
  knifeAnimation = null;
  controls.enabled = true;
  positionKnifeIdle();
}

function updateKnife(now: number): void {
  if (!knifeAnimation) return;
  const raw = (now - knifeAnimation.start) / knifeAnimation.duration;
  const t = clamp01(raw);
  let curveT = 0;
  let zLift = 0;
  if (t < 0.18) {
    curveT = 0;
    zLift = lerp(52, 4 - params.bladeDepth * 0.22, t / 0.18);
  } else if (t < 0.86) {
    curveT = (t - 0.18) / 0.68;
    zLift = 4 - params.bladeDepth * 0.22;
  } else {
    curveT = 1;
    zLift = lerp(4 - params.bladeDepth * 0.22, 52, (t - 0.86) / 0.14);
  }
  const point = knifeAnimation.curve.getPoint(curveT);
  const tangent = knifeAnimation.curve.getTangent(curveT).normalize();
  knifeGroup.position.copy(point).add(new THREE.Vector3(0, 0, zLift));
  knifeGroup.rotation.set(0, 0, Math.atan2(tangent.y, tangent.x));
  if (t >= 0.18) {
    if (!splitLow || !splitHigh) createSplitPreview(0);
    setSplitSeparation(curveT * 8);
  }
  if (t >= 1) {
    knifeAnimation = null;
    controls.enabled = true;
    cutDone = true;
    setSplitSeparation(8);
    document.querySelector('#plan-state')!.textContent = 'Cut complete';
    const c = currentCandidates[selectedCandidate];
    setStatus('Simulated cut complete', `Predicted portions: ${c.portionA.toFixed(1)} g and ${c.portionB.toFixed(1)} g. Mass is conserved in the particle model.`);
  }
}

function updatePokeRecovery(dt: number): boolean {
  let changed = false;
  for (const poke of pokes) {
    if (!poke.held) {
      const rate = params.recovery * (1.2 - params.viscosity * 0.72);
      const next = poke.amplitude * Math.exp(-dt * Math.max(0.02, rate));
      if (Math.abs(next - poke.amplitude) > 0.00005) changed = true;
      poke.amplitude = next;
    }
  }
  const before = pokes.length;
  pokes = pokes.filter((p) => p.amplitude > 0.012 || p.held);
  return changed || before !== pokes.length;
}

function setMouseFromEvent(event: PointerEvent): void {
  const rect = renderer.domElement.getBoundingClientRect();
  mouseNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouseNdc, camera);
}

function getFilletIntersection(event: PointerEvent): THREE.Intersection | null {
  setMouseFromEvent(event);
  const hits = raycaster.intersectObject(filletMesh, false);
  return hits[0] ?? null;
}

function updateFingerHover(event: PointerEvent): void {
  if (!fingerMode || cutDone) {
    fingerGroup.visible = false;
    return;
  }
  const hit = getFilletIntersection(event);
  if (hit) {
    fingerGroup.visible = true;
    const radius = params.fingerRadius * 0.48;
    fingertip.scale.set(0.82 * radius / 12, 0.82 * radius / 12, 1.22 * radius / 12);
    fingerGroup.position.set(hit.point.x, hit.point.y, hit.point.z + radius * 1.4);
    tooltip.classList.add('visible');
    tooltip.style.left = `${event.clientX}px`;
    tooltip.style.top = `${event.clientY}px`;
    const mat = localMaterial(hit.point.x, hit.point.y, hit.point.z);
    tooltip.innerHTML = `Density ${mat.density.toFixed(0)} kg/m³<br>Damping ${mat.viscosity.toFixed(2)} • Stiffness ${mat.stiffness.toFixed(0)} kPa`;
  } else if (!activePoke) {
    fingerGroup.visible = false;
    tooltip.classList.remove('visible');
  }
}

renderer.domElement.addEventListener('pointermove', (event) => {
  updateFingerHover(event);
  if (!activePoke) return;
  setMouseFromEvent(event);
  const point = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(boardPlane, point)) return;
  const dx = point.x - dragStart.x;
  const dy = point.y - dragStart.y;
  activePoke.center.set(point.x, point.y);
  activePoke.push.set(dx * 0.45, dy * 0.45);
  activePoke.depth = params.fingerForce * (params.thickness * 0.53) * (38 / Math.max(10, params.stiffness));
  activePoke.depth = Math.min(params.thickness * 0.52, Math.max(2, activePoke.depth));
  activePoke.radius = params.fingerRadius;
  activePoke.amplitude = 1;
  updateGeometryPositions(false);
  updateCutLine();
  updateMetrics();
  scheduleReplan(75);
  scheduleBackendSync('pointermove', 140);
});

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (!fingerMode || cutDone || event.button !== 0) return;
  const hit = getFilletIntersection(event);
  if (!hit) return;
  event.preventDefault();
  renderer.domElement.setPointerCapture(event.pointerId);
  controls.enabled = false;
  dragStart.set(hit.point.x, hit.point.y);
  const depth = Math.min(params.thickness * 0.52, Math.max(2, params.fingerForce * params.thickness * 0.53 * (38 / Math.max(10, params.stiffness))));
  activePoke = {
    center: new THREE.Vector2(hit.point.x, hit.point.y),
    radius: params.fingerRadius,
    depth,
    push: new THREE.Vector2(),
    amplitude: 1,
    held: true,
  };
  pokes.push(activePoke);
  if (pokes.length > 5) pokes.shift();
  setStatus('Finger contact detected', 'The material nodes are deforming according to stiffness, damping, recovery, pressure, and board friction settings.');
});

renderer.domElement.addEventListener('pointerup', (event) => {
  if (!activePoke) return;
  activePoke.held = false;
  activePoke = null;
  controls.enabled = true;
  try { renderer.domElement.releasePointerCapture(event.pointerId); } catch { /* no-op */ }
  optimizePaths(true);
  scheduleBackendSync('pointerup', 90);
});

renderer.domElement.addEventListener('pointerleave', () => {
  if (!activePoke) {
    fingerGroup.visible = false;
    tooltip.classList.remove('visible');
  }
});

function setViewMode(next: ViewMode): void {
  viewMode = next;
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((button) => button.classList.toggle('active', button.dataset.view === next));
  document.querySelector('#mode-badge')!.textContent = next.charAt(0).toUpperCase() + next.slice(1);
  updateColors();
}

function randomBetween(min: number, max: number, decimals = 2): number {
  const factor = Math.pow(10, decimals);
  return Math.round((min + Math.random() * (max - min)) * factor) / factor;
}

function syncControls(): void {
  for (const def of CONTROL_DEFS) {
    const input = document.querySelector<HTMLInputElement>(`#param-${def.key}`);
    const output = document.querySelector<HTMLOutputElement>(`#output-${def.key}`);
    if (input) input.value = String(params[def.key]);
    if (output) output.textContent = formatValue(def.key, params[def.key], def.unit);
  }
}

function randomizeFillet(): void {
  materialSeed = Math.floor(Math.random() * 9999) + 1;
  params.weight = randomBetween(190, 375, 0);
  params.length = randomBetween(148, 211, 0);
  params.width = randomBetween(74, 116, 0);
  params.thickness = randomBetween(22, 42, 0);
  params.taper = randomBetween(0.38, 0.78, 2);
  params.asymmetry = randomBetween(-0.12, 0.13, 2);
  params.curvature = randomBetween(-0.08, 0.08, 3);
  params.irregularity = randomBetween(0.018, 0.075, 3);
  params.density = randomBetween(1032, 1078, 0);
  params.heterogeneity = randomBetween(0.09, 0.29, 2);
  params.viscosity = randomBetween(0.34, 0.79, 2);
  params.stiffness = randomBetween(19, 52, 0);
  params.recovery = randomBetween(0.24, 0.72, 2);
  params.friction = randomBetween(0.30, 0.70, 2);
  params.targetWeight = Math.round(Math.min(params.weight * randomBetween(0.40, 0.62), params.weight - 45));
  params.cutAngle = randomBetween(-14, 14, 0);
  params.knifeSpeed = randomBetween(30, 78, 0);
  syncControls();
  document.querySelector('#seed-badge')!.textContent = `Seed ${String(materialSeed).padStart(3, '0')}`;
  rebuildFillet();
  setStatus('New fillet generated', `A plausible ${params.weight.toFixed(0)} g fillet was generated with spatially varying density, damping, and stiffness fields.`);
  scheduleBackendSync('randomize', 80);
}

for (const def of CONTROL_DEFS) {
  const input = document.querySelector<HTMLInputElement>(`#param-${def.key}`);
  const output = document.querySelector<HTMLOutputElement>(`#output-${def.key}`);
  if (!input || !output) continue;
  input.addEventListener('input', () => {
    params[def.key] = Number(input.value);
    output.textContent = formatValue(def.key, params[def.key], def.unit);
    if ((def.key === 'targetWeight' || def.key === 'weight') && params.targetWeight >= params.weight - 20) {
      params.targetWeight = Math.max(70, params.weight - 20);
      const targetInput = document.querySelector<HTMLInputElement>('#param-targetWeight');
      const targetOutput = document.querySelector<HTMLOutputElement>('#output-targetWeight');
      if (targetInput) targetInput.value = String(params.targetWeight);
      if (targetOutput) targetOutput.textContent = formatValue('targetWeight', params.targetWeight, 'g');
      if (def.key === 'targetWeight') {
        input.value = String(params.targetWeight);
        output.textContent = formatValue(def.key, params[def.key], def.unit);
      }
    }
    if (def.rebuild) {
      materialSeed += 1;
      rebuildFillet();
    } else if (def.replan) {
      scheduleReplan(60);
      updateMetrics();
      scheduleBackendSync('parameter-change', 120);
    } else {
      updateMetrics();
      scheduleBackendSync('parameter-change', 120);
    }
  });
}

document.querySelector('#view-switcher')!.addEventListener('click', (event) => {
  const target = event.target as HTMLElement;
  const view = target.dataset.view as ViewMode | undefined;
  if (view) setViewMode(view);
});

document.querySelector('#randomize')!.addEventListener('click', randomizeFillet);
document.querySelector('#randomize-top')!.addEventListener('click', randomizeFillet);
document.querySelector('#reset-shape')!.addEventListener('click', () => {
  Object.assign(params, defaults);
  materialSeed = 1;
  syncControls();
  document.querySelector('#seed-badge')!.textContent = 'Seed 001';
  rebuildFillet();
  setStatus('Default model restored', 'The original fillet shape and material preset have been restored.');
  scheduleBackendSync('reset-shape', 80);
});
document.querySelector('#reset-deform')!.addEventListener('click', () => {
  pokes = [];
  activePoke = null;
  updateGeometryPositions(true);
  optimizePaths(true);
  setStatus('Deformation cleared', 'The digital twin has returned to its generated rest state.');
  scheduleBackendSync('reset-deform', 80);
});
document.querySelector('#finger-mode')!.addEventListener('click', (event) => {
  fingerMode = !fingerMode;
  const button = event.currentTarget as HTMLButtonElement;
  button.textContent = `Finger tool: ${fingerMode ? 'ON' : 'OFF'}`;
  button.classList.toggle('active', fingerMode);
  if (!fingerMode) fingerGroup.visible = false;
});
document.querySelector('#optimize')!.addEventListener('click', () => optimizePaths(true));
document.querySelector('#approve')!.addEventListener('click', () => {
  if (!currentCandidates[selectedCandidate]) return;
  approved = true;
  updateCutLine();
  updateMetrics();
  document.querySelector('#plan-state')!.textContent = 'Human approved';
  setStatus('Cut path approved', `Candidate ${String.fromCharCode(65 + selectedCandidate)} is locked for the current digital-twin state. Any new deformation will require re-approval.`);
  scheduleBackendSync('approve', 80);
});
document.querySelector('#run-cut')!.addEventListener('click', () => {
  if (!approved) {
    approved = true;
    updateCutLine();
  }
  startKnife();
  scheduleBackendSync('run-cut', 80);
});
document.querySelector('#reset-cut')!.addEventListener('click', () => {
  stopKnife();
  cutDone = false;
  clearSplitPreview();
  updateCutLine();
  document.querySelector('#plan-state')!.textContent = approved ? 'Human approved' : 'Ready';
  setStatus('Cut preview reset', 'The fillet remains editable and the proposed path can be recalculated.');
});

document.querySelector('#export-path')!.addEventListener('click', () => {
  const candidate = currentCandidates[selectedCandidate];
  if (!candidate) return;
  const payload = {
    schema: 'fillettwin.approved-path.v1',
    generatedAt: new Date().toISOString(),
    status: approved ? 'human_approved' : 'proposal_only',
    simulationOnly: true,
    fillet: {
      seed: materialSeed,
      weight_g: params.weight,
      length_mm: params.length,
      width_mm: params.width,
      thickness_mm: params.thickness,
      density_kg_m3: params.density,
      viscoelastic_damping: params.viscosity,
      stiffness_kpa_proxy: params.stiffness,
    },
    target: { weight_g: params.targetWeight, tolerance_g: params.tolerance },
    path: {
      angle_deg: THREE.MathUtils.radToDeg(candidate.angle),
      plane_offset_mm: candidate.offset,
      knife_velocity_mm_s: params.knifeSpeed,
      blade_penetration_mm: params.bladeDepth,
      waypoints_mm: candidate.points.map((p) => [Number(p.x.toFixed(3)), Number(p.y.toFixed(3)), Number(p.z.toFixed(3))]),
    },
    prediction: {
      portion_a_g: Number(candidate.portionA.toFixed(3)),
      portion_b_g: Number(candidate.portionB.toFixed(3)),
      target_error_g: Number(candidate.error.toFixed(3)),
      path_length_mm: Number(candidate.pathLength.toFixed(3)),
      confidence: Number(candidate.confidence.toFixed(3)),
    },
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `fillettwin-path-${materialSeed}.json`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus('Path file exported', approved ? 'The human-approved robot-neutral path JSON has been downloaded.' : 'A proposal-only JSON was downloaded. Approve the path before treating it as a reviewed output.');
});

document.querySelector('#camera-reset')!.addEventListener('click', () => {
  camera.position.set(185, -218, 172);
  controls.target.set(0, 0, 21);
  controls.update();
});
document.querySelector('#toggle-left')!.addEventListener('click', () => document.querySelector('#left-panel')!.classList.toggle('open'));
document.querySelector('#toggle-right')!.addEventListener('click', () => document.querySelector('#right-panel')!.classList.toggle('open'));

function onResize(): void {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}
window.addEventListener('resize', onResize);

let lastFrame = performance.now();
let normalFrameCounter = 0;
function animate(now: number): void {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const recovering = updatePokeRecovery(dt);
  if (recovering && !activePoke && !cutDone) {
    normalFrameCounter++;
    updateGeometryPositions(normalFrameCounter % 8 === 0);
    updateCutLine();
    updateMetrics();
    if (normalFrameCounter % 18 === 0) scheduleReplan(120);
  }
  updateKnife(now);
  controls.update();
  renderer.render(scene, camera);
}

rebuildFillet();
onResize();
requestAnimationFrame(animate);
