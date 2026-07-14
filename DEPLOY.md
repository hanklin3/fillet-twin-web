# Deploy in under five minutes

## Netlify drag-and-drop

1. Run `npm install` and `npm run build`, or use the already-generated `dist` folder.
2. Open Netlify Drop in your browser.
3. Drag the `dist` folder onto the page.
4. Netlify will issue a public HTTPS URL.

## Netlify from Git

1. Push this folder to GitHub.
2. Create a new Netlify site from the repository.
3. Build command: `npm run build`
4. Publish directory: `dist`
5. Node version: 22

The included `netlify.toml` already contains these settings.

## Any static host

Upload `dist/index.html`. It is a single self-contained file with Three.js, application code, and CSS inlined. No API server or database is required.
