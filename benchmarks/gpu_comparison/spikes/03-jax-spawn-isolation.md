# Spike 3: JAX-on-CPU child + JAX-on-GPU parent process isolation

## Verdict

**both-orderings-work**

Both `child-cpu-then-parent-gpu` and `parent-gpu-then-child-cpu` orderings
succeed cleanly. The `multiprocessing.spawn` child gets a fresh interpreter,
so setting `JAX_PLATFORMS=cpu` before the child's `import jax` correctly
limits it to CPU; the parent's GPU JAX state is unaffected either way, and
the parent still sees both A100s after the child terminates.

## Required process ordering

**No ordering constraint required.** The runner is free to interleave
spawned CPU children and in-process GPU work in either direction. This
gives Task 11 flexibility — `astrojax-cpu` (spawned) and
`astrojax-gpu` / `astrojax-multigpu` (in-process) can run in any order
within a single runner invocation.

Recommended best practice anyway: the runner should set
`JAX_PLATFORMS=cpu` (and optionally `CUDA_VISIBLE_DEVICES=""`) inside the
spawned child's `target` function *before* importing JAX, exactly as the
smoke script does. Inheriting the parent's `CUDA_VISIBLE_DEVICES` is fine
on a spawn context because each child is a fresh Python process, but the
env-var override at the top of the child target is the belt-and-suspenders
guarantee.

## Repro commands and output

Host: 2x NVIDIA A100 80GB PCIe, driver 595.71.05, CUDA Version 13.2
(reported by `nvidia-smi`).

### Forward ordering — child CPU first, then parent GPU

```
$ uv run --no-project --with 'astrojax[cuda13]' \
    python benchmarks/gpu_comparison/spikes/spawn_smoke.py
('cpu_child_devices', ['cpu:0'])
PARENT GPU devices: ['cuda:0', 'cuda:1']
```

### Reverse ordering — parent GPU first, then child CPU

```
$ uv run --no-project --with 'astrojax[cuda13]' \
    python benchmarks/gpu_comparison/spikes/spawn_smoke_reverse.py
PARENT GPU devices: ['cuda:0', 'cuda:1']
('cpu_child_devices', ['cpu:0'])
PARENT GPU devices after child: ['cuda:0', 'cuda:1']
```

In the reverse case, the parent's GPU device list is identical before and
after the spawned CPU child, confirming no fork-style poisoning of the
parent CUDA context.

## Plan B (not needed)

Both orderings work, so the contingency design — running BOTH
`astrojax-gpu` AND `astrojax-cpu` in spawned children with no in-process
JAX in the runner — is **not required**. Recording it here for the plan's
benefit in case a future driver/CUDA/JAX combination breaks the
assumption: the fallback is to wrap every astrojax config in a spawned
child and pipe results back via a `multiprocessing.Queue` or a JSON file,
keeping the runner process JAX-free.

## CUDA toolchain note

- **Working extra:** `astrojax[cuda13]`
- **Why:** the host reports CUDA 13.2; `jax-cuda13-pjrt` /
  `jax-cuda13-plugin` and the matching `nvidia-*` CUDA 13 wheels resolved
  and ran cleanly on first invocation. `cuda12` was not exercised because
  `cuda13` worked on the first try.
- **Wheel size note:** first invocation downloaded the full CUDA 13 wheel
  set (~2 GB across `nvidia-cudnn-cu13`, `nvidia-cusolver`, `nvidia-cufft`,
  `nvidia-cublas`, `jax-cuda13-pjrt`, etc.). Subsequent invocations are
  cached and immediate.
- **Downstream install recipes** (`pyproject.toml` extras / `justfile`
  recipes / docs) should pin `astrojax[cuda13]` for this machine class.
  If we later need to support hosts with older CUDA 12.x drivers, add a
  parallel `cuda12` recipe; the same spike script can be re-run there to
  confirm.

### uv invocation gotcha

Running with the project's environment active produced a resolver error
because `brahe[gpu-comparison]` pins `astrojax>=0.8.0` while only
`astrojax<=0.7.3` is currently published on PyPI:

```
× No solution found when resolving dependencies for split ...
  Because only astrojax<=0.7.3 is available and brahe[gpu-comparison]
  depends on astrojax>=0.8.0, we can conclude that brahe[gpu-comparison]'s
  requirements are unsatisfiable.
```

Workaround: use `uv run --no-project --with 'astrojax[cuda13]' ...` so the
spike runs in an ephemeral env and ignores the project's
`gpu-comparison` extra. Plan Task 11 (and the eventual
`pyproject.toml` updates) should reconcile the astrojax version pin with
what's published, or switch to a Git/local source for astrojax.

## Files

- Forward script: `benchmarks/gpu_comparison/spikes/spawn_smoke.py`
- Reverse script: `benchmarks/gpu_comparison/spikes/spawn_smoke_reverse.py`
