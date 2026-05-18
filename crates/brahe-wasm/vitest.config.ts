import { defineConfig } from 'vitest/config';
import wasmPlugin from 'vite-plugin-wasm';

export default defineConfig({
  plugins: [wasmPlugin()],
  test: {
    include: ['tests/**/*.test.ts'],
    // Vitest is Vite-based; it resolves the `#wasm` import via the
    // `imports` field in package.json, picking the `default` (bundler)
    // condition. The node-target WASM build is exercised by consumers
    // but not by these tests. See spec for the acknowledged gap.
  },
  resolve: {
    // tsconfig.json `paths` is for `tsc` type-checking; Vitest's resolver
    // needs an explicit alias to find `#wasm` at test runtime.
    alias: {
      '#wasm': new URL('./pkg-bundler/brahe_wasm.js', import.meta.url).pathname,
    },
  },
});
