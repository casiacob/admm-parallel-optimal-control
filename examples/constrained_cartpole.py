import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics
from jax import lax, debug
from admm_noc.utils import wrap_angle, rollout
from admm_noc.par_admm_optimal_control import par_admm


# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")

def projection(z):
    x_min = jnp.finfo(jnp.float64).min
    x_max = jnp.finfo(jnp.float64).max
    ub = jnp.array([x_max, x_max, x_max, x_max, 50.])
    lb = jnp.array([x_min, x_min, x_min, x_min, -50.])
    return jnp.clip(z, lb, ub)


def final_cost(state: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = (
        0.5
        * (_wrapped - goal_state).T
        @ final_state_cost
        @ (_wrapped - goal_state)
    )
    return c


def transient_cost(state: jnp.ndarray, action: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c

def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)

def cartpole(
    state: jnp.ndarray, action: jnp.ndarray
) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action
        + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=cartpole, simulation_step=simulation_step, downsampling=downsampling
)

horizon = 15
key = jax.random.PRNGKey(271)
u_init = jnp.array([0.01]) * jax.random.normal(key, shape=(horizon, 1))
x0_init = jnp.array([0.01, wrap_angle(-0.01), 0.01, -0.01])
x_init = rollout(dynamics, u_init, x0_init)
z_init = jnp.zeros((horizon, u_init.shape[1] + x_init.shape[1]))
l_init = jnp.zeros((horizon, u_init.shape[1] + x_init.shape[1]))
sigma = 0.1

def mpc_loop(carry, input):
    jax.debug.print('------------------------')
    x0, u, z, l = carry
    x = rollout(dynamics, u, x0)
    x, u, z, l, iterations = par_admm(
        transient_cost, final_cost, dynamics, projection, x, u, z, l, sigma
    )
    return (x[1], u, z, l), (x[1], u[0], iterations)

jitted_mpc_loop = jax.jit(mpc_loop)
_, (mpc_x, mpc_u, n_iterations) = jax.lax.scan(
    jitted_mpc_loop, (x0_init, u_init, z_init, l_init), xs=None, length=100
)
plt.plot(mpc_x[:, 0])
plt.plot(mpc_x[:, 1])
plt.show()
plt.plot(mpc_u[:, 0])
plt.show()
print(jnp.sum(n_iterations))