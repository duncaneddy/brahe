"""Verify that running JAX-on-CPU in a spawned child process does not poison
the parent's JAX-on-GPU init.

Run as:
    uv run --with astrojax python benchmarks/gpu_comparison/spikes/spawn_smoke.py
"""
import multiprocessing as mp
import os


def _cpu_child(out_q):
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    devs = [str(d) for d in jax.devices()]
    out_q.put(("cpu_child_devices", devs))


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cpu_child, args=(q,))
    p.start()
    p.join()
    print(q.get())

    # Now init JAX-on-GPU in the parent
    import jax
    try:
        gpus = jax.devices("gpu")
    except RuntimeError as e:
        print("PARENT GPU init failed:", e)
        return
    print("PARENT GPU devices:", [str(d) for d in gpus])


if __name__ == "__main__":
    main()
