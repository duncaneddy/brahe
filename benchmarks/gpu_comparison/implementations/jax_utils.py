"""Small shared helpers used by the astrojax kernel builders.

Centralises the multi-device sharding pattern so the per-task builders don't
each re-implement it. Drop-in replacement for the deprecated
``jax.device_put_sharded`` call.
"""

from __future__ import annotations

from typing import Sequence


def shard_across_devices(array, devices: Sequence):
    """Shard the leading axis of ``array`` across ``devices`` using the modern
    ``jax.device_put`` + ``NamedSharding`` API.

    ``array.shape[0]`` must equal ``len(devices)``. The leading axis is
    partitioned across the device mesh; every other axis is replicated.
    Returns a sharded JAX array ready to feed into ``jax.pmap`` or
    ``jax.jit`` with a matching shape.
    """
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    n_dev = len(devices)
    if array.shape[0] != n_dev:
        raise ValueError(
            f"leading axis ({array.shape[0]}) must equal number of devices ({n_dev})"
        )
    mesh = Mesh(list(devices), ("dev",))
    spec = PartitionSpec("dev", *([None] * (array.ndim - 1)))
    return jax.device_put(array, NamedSharding(mesh, spec))
