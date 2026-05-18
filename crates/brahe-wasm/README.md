# brahe-wasm

WebAssembly bindings for [brahe](https://github.com/duncaneddy/brahe), a modern astrodynamics library for research and engineering applications.

This package re-exposes brahe's Rust API to JavaScript and TypeScript via WebAssembly. It currently includes:

- **Constants** — mathematical, physical, time-system, and planetary constants

More submodules (time conversions, frames, orbits, propagation) are coming.

## Installation

```bash
npm install brahe-wasm
```

## Quick Start

```typescript
import { DEG2RAD, GM_EARTH, R_EARTH } from 'brahe-wasm';

const radians = 90 * DEG2RAD;
console.log(`Earth GM: ${GM_EARTH} m³/s²`);
console.log(`Earth radius: ${R_EARTH / 1000} km`);
```

## Documentation

- [Main brahe documentation](https://duncaneddy.github.io/brahe/latest/)
- [WASM API Reference](https://duncaneddy.github.io/brahe/latest/library_api/wasm/)
- [Python API Reference](https://duncaneddy.github.io/brahe/latest/library_api/python/) (sister package)

## Source

The Rust source for these bindings lives at [`crates/brahe-wasm/`](https://github.com/duncaneddy/brahe/tree/main/crates/brahe-wasm) in the brahe monorepo. Issues and contributions: [github.com/duncaneddy/brahe](https://github.com/duncaneddy/brahe).

## License

MIT
