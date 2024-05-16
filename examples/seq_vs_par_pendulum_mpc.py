import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics
from jax import lax, debug
from admm_noc.utils import wrap_angle, rollout
from admm_noc.par_admm_optimal_control import par_admm
from admm_noc.seq_admm_optimal_control import seq_admm


# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")

import time


def projection(z):
    control_ub = jnp.array([jnp.inf, jnp.inf, 5.0])
    control_lb = jnp.array([-jnp.inf, -jnp.inf, -5.0])
    return jnp.clip(z, control_lb, control_ub)


def final_cost(state):
    goal_state = jnp.array((jnp.pi, 0.0))
    final_state_cost = jnp.diag(jnp.array([2e1, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    # err = state - goal_state

    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state, action):
    goal_state = jnp.array((jnp.pi, 0.0))
    state_cost = jnp.diag(jnp.array([2e1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    # err = state - goal_state
    c = 0.5 * err.T @ state_cost @ err
    c += 0.5 * action.T @ action_cost @ action
    return c


def pendulum(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length ** 2),
        )
    )

key = jax.random.PRNGKey(1)

def mpc_experiment_seq(disc_step, sim_steps, x0_init, u_init, z_init, l_init, sigma):
    return end - start


disc_step_range = [0.05, 0.025, 0.01, 0.005]
horizon_range = [15, 30, 75, 150]
sim_step_range = [80, 160, 400, 800]
seq_times = []
par_times = []

horizon = horizon_range[3]
disc_step = disc_step_range[3]
sim_steps = sim_step_range[3]
u_init = 0.1 * jax.random.normal(key, shape=(horizon, 1))
x0_init = jnp.array([wrap_angle(0.1), -0.1])
z_init = jnp.zeros((horizon, u_init.shape[1] + x0_init.shape[0]))
l_init = jnp.zeros((horizon, u_init.shape[1] + x0_init.shape[0]))
sigma = 0.1

downsampling = 1
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=disc_step, downsampling=downsampling
)
def mpc_loop_seq(carry, input):
    x0, u, z, l = carry
    x = rollout(dynamics, u, x0)
    x, u, z, l = seq_admm(
        transient_cost, final_cost, dynamics, projection, x, u, z, l, sigma
    )
    return (x[1], u, z, l), (x[1], u[0])


_, (mpc_x, mpc_u) = jax.lax.scan(
    mpc_loop_seq, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)
start = time.time()
_, (mpc_x, mpc_u) = jax.lax.scan(
    mpc_loop_seq, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)
jax.block_until_ready(mpc_x)
end = time.time()
print('seq time ', end-start)

def mpc_loop_par(carry, input):
    x0, u, z, l = carry
    x = rollout(dynamics, u, x0)
    x, u, z, l = par_admm(
        transient_cost, final_cost, dynamics, projection, x, u, z, l, sigma
    )
    return (x[1], u, z, l), (x[1], u[0])

_, (mpc_x, mpc_u) = jax.lax.scan(
    mpc_loop_par, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)
start = time.time()
_, (mpc_x, mpc_u) = jax.lax.scan(
    mpc_loop_par, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)
jax.block_until_ready(mpc_x)
end = time.time()
print('par time ', end-start)
