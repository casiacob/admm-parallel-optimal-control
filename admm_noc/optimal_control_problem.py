from typing import NamedTuple, Callable
import jax.numpy as jnp

class ADMM_OCP(NamedTuple):
    dynamics: Callable
    projection: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable
    penalty_parameter: float


class Derivatives(NamedTuple):
    cx: jnp.ndarray
    cu: jnp.ndarray
    cxx: jnp.ndarray
    cuu: jnp.ndarray
    cxu: jnp.ndarray
    fx: jnp.ndarray
    fu: jnp.ndarray
    fxx: jnp.ndarray
    fuu: jnp.ndarray
    fxu: jnp.ndarray


class LinearizedOCP(NamedTuple):
    r: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    M: jnp.ndarray


class ADMM_LIN_OCP(NamedTuple):
    Ad: jnp.array             # transition matrices
    Bd: jnp.ndarray           # control matrices
    P: jnp.ndarray            # final cost state penalty
    Q: jnp.ndarray            # stage cost state penalty
    R: jnp.ndarray            # stage cost control penaty
    projection: Callable      # projection function
    penalty_parameter: float  # admm penalty parameter

