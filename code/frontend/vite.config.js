import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies the API to the FastAPI process rather than serving it,
// so the two run independently: the pipeline keeps its own process and its own
// restart cycle, and a front-end reload never interrupts a job that is running.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  // Built into the backend's own static directory so that a single `uvicorn`
  // serves the finished application. Without this the deployed system would
  // need two processes and a proxy in front of them, which is a lot of moving
  // parts for a tool one person runs on a laptop.
  build: {
    outDir: '../backend/static/app',
    emptyOutDir: true,
  },
  base: '/app/',
  // jsdom rather than a real browser: these test the components' logic and the
  // decisions encoded in them, not rendering. Anything that needs a real browser
  // is verified by driving one against the running server instead.
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.js'],
  },
})
