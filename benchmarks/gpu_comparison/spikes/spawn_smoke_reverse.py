"""Reverse-ordering variant of spawn_smoke.py.

(a) Init JAX-on-GPU in the parent first.
(b) THEN spawn the CPU child and ensure it can init JAX-on-CPU cleanly.

Run as:
    uv run --with 'astrojax[cuda13]' python \
        benchmarks/gpu_comparison/spikes/spawn_smoke_reverse.py
"""
import multiprocessing as mp
import os


def _cpu_child(out_q):
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        import jax
        devs = [str(d) for d in jax.devices()]
        out_q.put(("cpu_child_devices", devs))
    except Exception as e:  # pragma: no cover - we want raw failure info
        out_q.put(("cpu_child_error", repr(e)))


def main():
    # Init JAX-on-GPU in the parent FIRST
    import jax
    try:
        gpus = jax.devices("gpu")
    except RuntimeError as e:
        print("PARENT GPU init failed:", e)
        return
    print("PARENT GPU devices:", [str(d) for d in gpus])

    # Now spawn the CPU child
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cpu_child, args=(q,))
    p.start()
    p.join()
    print(q.get())

    # Re-check parent GPU still works (sanity)
    gpus_after = jax.devices("gpu")
    print("PARENT GPU devices after child:", [str(d) for d in gpus_after])


if __name__ == "__main__":
    main()
