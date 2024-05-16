import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics
from jax import lax, debug
from admm_noc.utils import wrap_angle, rollout
from admm_noc.par_admm_optimal_control import par_admm
from admm_noc.seq_admm_optimal_control import seq_admm

_jitted_par_admm = jax.jit(par_admm)
_jitted_seq_admm = jax.jit(seq_admm)

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")

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

disc_step_range = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.0016, 0.00125, 0.001]
frequencies = [25, 50, 100, 200, 400, 600, 800, 1000]
horizon_range = [20, 40, 70, 140, 280, 500, 600, 700]
sim_step_range = [100, 200, 400, 800, 1600, 2500, 3200, 4000]
sigma_range = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]

seq_times = []
par_times = []

def mpc_experiment_seq(Ts, H, N, penalty_param):
    u_init = 0.1 * jax.random.normal(key, shape=(H, 1))
    x0_init = jnp.array([wrap_angle(0.1), -0.1])
    z_init = jnp.zeros((H, u_init.shape[1] + x0_init.shape[0]))
    l_init = jnp.zeros((H, u_init.shape[1] + x0_init.shape[0]))

    downsampling = 1
    dynamics = discretize_dynamics(
        ode=pendulum, simulation_step=Ts, downsampling=downsampling
    )

    def mpc_loop_seq(carry, input):
        x0, u, z, l = carry
        x = rollout(dynamics, u, x0)
        x, u, z, l = _jitted_seq_admm(
            transient_cost, final_cost, dynamics, projection, x, u, z, l, penalty_param
        )
        return (x[1], u, z, l), (x[1], u[0])

    start = time.time()
    _, (mpc_x, mpc_u) = jax.lax.scan(
        mpc_loop_seq, (x0_init, u_init, z_init, l_init), xs=None, length=N
    )
    jax.block_until_ready(mpc_x)
    end = time.time()

    return start - end

def mpc_experiment_par(Ts, H, N, penalty_param):
    u_init = 0.1 * jax.random.normal(key, shape=(H, 1))
    x0_init = jnp.array([wrap_angle(0.1), -0.1])
    z_init = jnp.zeros((H, u_init.shape[1] + x0_init.shape[0]))
    l_init = jnp.zeros((H, u_init.shape[1] + x0_init.shape[0]))

    downsampling = 1
    dynamics = discretize_dynamics(
        ode=pendulum, simulation_step=Ts, downsampling=downsampling
    )

    def mpc_loop_seq(carry, input):
        x0, u, z, l = carry
        x = rollout(dynamics, u, x0)
        x, u, z, l = _jitted_par_admm(
            transient_cost, final_cost, dynamics, projection, x, u, z, l, penalty_param
        )
        return (x[1], u, z, l), (x[1], u[0])

    start = time.time()
    _, (mpc_x, mpc_u) = jax.lax.scan(
        mpc_loop_seq, (x0_init, u_init, z_init, l_init), xs=None, length=N
    )
    jax.block_until_ready(mpc_x)
    end = time.time()

    return end - start


mpc_experiment_seq(disc_step_range[0], horizon_range[0], sim_step_range[0], sigma_range[0])
mpc_experiment_par(disc_step_range[0], horizon_range[0], sim_step_range[0], sigma_range[0])

for i in range(2):
    s_time = mpc_experiment_seq(disc_step_range[i], horizon_range[i], sim_step_range[i], sigma_range[i])
    p_time = mpc_experiment_par(disc_step_range[i], horizon_range[i], sim_step_range[i], sigma_range[i])
    seq_times.append(s_time)
    par_times.append(p_time)


plt.plot(jnp.array(frequencies[:2]), jnp.array(seq_times), marker='s', label='seq')
plt.plot(jnp.array(frequencies[:2]), jnp.array(par_times), marker='.', label='par')
plt.grid(which='both')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.show()