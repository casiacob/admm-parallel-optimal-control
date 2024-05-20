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


# _jitted_par_admm = jax.jit(par_admm)
# _jitted_seq_admm = jax.jit(seq_admm)

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

Ts = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.0016, 0.00125, 0.001]
f = [25, 50, 100, 200, 400, 600, 800, 1000]
H = [20, 40, 70, 140, 280, 500, 600, 700]
N = [100, 200, 400, 800, 1600, 2500, 3200, 4000]
sigma = [0.1, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5]

downsampling = 1

#################################### experiment 0 ######################################################################
horizon = H[4]
sim_steps = N[4]
disc_step = Ts[4]
penalty = sigma[4]

dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=disc_step, downsampling=downsampling
)
u_init = 0.1 * jax.random.normal(key, shape=(horizon, 1))
x0_init = jnp.array([wrap_angle(0.1), -0.1])
x_init = rollout(dynamics, u_init, x0_init)
z_init = jnp.zeros((horizon, u_init.shape[1] + x_init.shape[1]))
l_init = jnp.zeros((horizon, u_init.shape[1] + x_init.shape[1]))
def mpc_loop_seq(carry, input):
    x0, u, z, l = carry
    x = rollout(dynamics, u, x0)
    x, u, z, l = seq_admm(
        transient_cost, final_cost, dynamics, projection, x, u, z, l, penalty
    )
    return (x[1], u, z, l), (x[1], u[0])

def mpc_loop_par(carry, input):
    x0, u, z, l = carry
    x = rollout(dynamics, u, x0)
    x, u, z, l = par_admm(
        transient_cost, final_cost, dynamics, projection, x, u, z, l, penalty
    )
    return (x[1], u, z, l), (x[1], u[0])


_jitted_mpc_loop_seq = jax.jit(mpc_loop_seq)
_jitted_mpc_loop_par = jax.jit(mpc_loop_par)

_, (mpc_x, mpc_u) = jax.lax.scan(
    _jitted_mpc_loop_seq, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)

start = time.time()
_, (mpc_x_seq, mpc_u_seq) = jax.lax.scan(
    _jitted_mpc_loop_seq, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)
jax.block_until_ready(mpc_x_seq)
end = time.time()

seq_time = end-start

_, (mpc_x, mpc_u) = jax.lax.scan(
    _jitted_mpc_loop_par, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)

start = time.time()
_, (mpc_x_par, mpc_u_par) = jax.lax.scan(
    _jitted_mpc_loop_par, (x0_init, u_init, z_init, l_init), xs=None, length=sim_steps
)
jax.block_until_ready(mpc_x_par)
end = time.time()

par_time = end-start

print('Sampling period: ', disc_step, ', Horizon: ', horizon, ', Horizon time: ', disc_step * horizon, ', Simulation time: ', disc_step * sim_steps, ', Simulation time steps: ', sim_steps)
print('Sequential time: ', seq_time)
print('Parallel time  : ', par_time)
print('par vs seq solution')
print('u: ', jnp.max(jnp.abs(mpc_u_par - mpc_u_seq)))
print('x_1: ', jnp.max(jnp.abs(mpc_x_par[:, 0] - mpc_x_seq[:, 0])))
print('x_2: ', jnp.max(jnp.abs(mpc_x_par[:, 1] - mpc_x_seq[:, 1])))
#######################################################################################################################

plt.plot(mpc_x_seq[:, 0], label='angle seq')
plt.plot(mpc_x_par[:, 0], label='angle par')
# # plt.show()
plt.plot(mpc_u_seq, label='control seq')
plt.plot(mpc_u_par, label='control par')
plt.legend()
plt.show()