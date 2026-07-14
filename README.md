# FilletTwin Apprentice — Web Demo

A self-contained, browser-based prototype for visualizing a parametric chicken-breast digital twin, spatial material-property fields, mass-based cutting-path planning, finger deformation, live replanning, and a visual knife pass.

<p align="center">
  <img src="assets/UI.gif" alt="FilletTwin UI showing a planned knife cut" width="1080">
</p>

## What is implemented

- Procedurally generated 3D chicken fillet with bounded “common-sense” shape randomization
- Adjustable weight, length, width, thickness, taper, asymmetry, curvature, and irregularity
- Spatially varying density, viscoelastic damping, and stiffness fields
- Smooth surface, heat-map, wireframe, and volumetric particle-lattice views
- Draggable virtual finger that depresses and pushes the fillet
- Shape recovery controlled by damping, recovery, stiffness, pressure, and board friction
- Mass-integrating cut planner that proposes three candidate cutting planes
- Target portion weight, tolerance, preferred angle, cut-length and confidence metrics
- Human approval state
- Knife velocity and penetration controls
- Animated knife traversal and visual split preview
- Responsive desktop/tablet layout
- Single-file production build (`dist/index.html`) that does not require a server

## Important limitations

This is a qualitative product prototype. It is not a calibrated food-material simulator and does not yet model fracture mechanics, force feedback, anisotropic muscle fibers, food safety, camera calibration, or robot collision constraints. “Viscosity” is represented as a user-friendly viscoelastic damping proxy.

## Run locally

# Conda installation
```
conda create -n df python=3.11
conda activate df
pip install -r requirements.txt
conda install -c conda-forge nodejs
```

For the browser UI, run the Vite app:

```bash
npm ci
npm run dev
```

`npm ci` installs the exact versions recorded in `package-lock.json`. If it
appears to stall, confirm that npm can reach the public registry with
`npm ping --registry=https://registry.npmjs.org/`.

Open the URL shown by Vite, normally `http://localhost:5173`.

To run the full application with the browser UI and Python simulation backbone,
build the UI once and start the Python server:

```bash
npm run build
python python/fillet_sim.py
```

Open `http://127.0.0.1:8000`. The Python process serves the exact built UI from
`dist/index.html` and provides `/api/snapshot` for the PyTorch simulation. The
browser has a local visual fallback only if that API becomes unavailable.

For UI development, run `npm run dev` and keep the Python server running in a
second terminal; Vite proxies `/api` to port 8000.

## Build the standalone demo

```bash
npm run build
```

The resulting `dist/index.html` is a single bundled HTML file. It can be opened directly in a modern browser or hosted on any static web host such as Netlify, Vercel, Cloudflare Pages, GitHub Pages, or S3.

## Interaction

1. Click **Randomize fillet**.
2. Switch between Surface, Density, Viscosity, Stiffness, and Lattice views.
3. Click and drag directly on the chicken fillet with the finger tool enabled.
4. Adjust the target portion weight and click **Optimize path**.
5. Select a candidate and click **Approve selected**.
6. Adjust knife velocity/depth and click **Run knife**.

## Suggested next engineering steps

1. Replace the qualitative deformation kernel with Taichi MPM or SOFA.
2. Add explicit fiber-direction and damage/fracture fields.
3. Import calibrated RGB-D geometry and tracked knife poses.
4. Export approved paths in a robot-neutral JSON schema.
5. Connect to Isaac Sim/ROS 2 for hardware-stage validation.
