import equinox as eqx
from jax import Array


class ModelOutput(eqx.Module):
    """Bundles a model prediction with auxiliary side-channel tensors.

    `aux` is a mapping of named arrays produced inside a wrapper that
    downstream losses may want to consume directly (e.g. predicted
    magnitude/phase from spectral processing) instead of re-deriving them
    from `prediction`. Wrappers without side channels should pass `aux={}`.

    The keys must be stable across calls so JAX traces a consistent PyTree
    structure under `jit`/`vmap`.
    """

    prediction: Array
    aux: dict[str, Array]
