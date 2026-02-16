import { defineConfig } from 'vite';

export default defineConfig({
  root: __dirname,
  base: '/tna-js/',
  server: {
    open: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
