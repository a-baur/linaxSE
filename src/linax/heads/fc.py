"""Regression head."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from linax.heads.base import Head, HeadConfig


@dataclass(frozen=True)
class FCHeadConfig(HeadConfig):
    """Configuration for the regression head."""

    out_features: int

    def build(self, in_features: int, key: PRNGKeyArray) -> FCHead:
        """Build head from config."""
        return FCHead(in_features=in_features, cfg=self, key=key)


class FCHead[ConfigType: FCHeadConfig](Head):
    """Fully connected layer with activation."""

    linear: eqx.nn.Linear
    in_features: int
    out_features: int

    def __init__(self, in_features: int, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the fc head."""
        self.linear = eqx.nn.Linear(
            in_features=in_features, out_features=cfg.out_features, key=key
        )
        self.in_features, self.out_features = in_features, cfg.out_features

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the fc head."""
        x = jax.vmap(self.linear)(x)
        x = jax.nn.relu(x)
        return x, state
