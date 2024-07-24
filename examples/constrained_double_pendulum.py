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
    ub = jnp.array([x_max, x_max, x_max, x_max, 15.0, 15.0])
    lb = jnp.array([x_min, x_min, x_min, x_min, -15.0, -15.0])
    return jnp.clip(z, lb, ub)


def final_cost(state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([2e1, 2e1, 1e-1, 1e-1]))
    goal_state = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c

def transient_cost(state: jnp.ndarray, action: jnp.ndarray) -> float:
    goal_state = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    state_cost = jnp.diag(jnp.array([1e1, 1e-1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-2, 1e-2]))

    _wrapped = jnp.hstack(
        (
            wrap_angle(state[0]),
            wrap_angle(state[1]),
            state[2],
            state[3]
        )
    )
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c

def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)

def double_pendulum(
    state: jnp.ndarray, action: jnp.ndarray
) -> jnp.ndarray:

    # https://underactuated.mit.edu/multibody.html#section1

    g = 9.81
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    k1, k2 = 1e-3, 1e-3

    th1, th2, dth1, dth2 = state
    u1, u2 = action

    s1, c1 = jnp.sin(th1), jnp.cos(th1)
    s2, c2 = jnp.sin(th2), jnp.cos(th2)
    s12 = jnp.sin(th1 + th2)

    # inertia
    M = jnp.array(
        [
            [
                (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * c2,
                m2 * l2**2 + m2 * l1 * l2 * c2,
            ],
            [
                m2 * l2**2 + m2 * l1 * l2 * c2,
                m2 * l2**2
            ],
        ]
    )

    # Corliolis
    C = jnp.array(
        [
            [
                0.0,
                -m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2
            ],
            [
                0.5 * m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2,
                -0.5 * m2 * l1 * l2 * dth1 * s2,
            ],
        ]
    )

    # gravity
    tau = -g * jnp.array(
        [
            (m1 + m2) * l1 * s1 + m2 * l2 * s12,
            m2 * l2 * s12
        ]
    )

    B = jnp.eye(2)

    u1 = u1 - k1 * dth1
    u2 = u2 - k2 * dth2

    u = jnp.hstack([u1, u2])
    v = jnp.hstack([dth1, dth2])

    a = jnp.linalg.solve(M, tau + B @ u - C @ v)

    return jnp.hstack((v, a))


simulation_step = 0.005
downsampling = 1
dynamics = discretize_dynamics(
    ode=double_pendulum, simulation_step=simulation_step, downsampling=downsampling
)

horizon = 140
key = jax.random.PRNGKey(271)
u_init = jnp.array([0.01]) * jax.random.normal(key, shape=(horizon, 2))
x0_init = jnp.array(
    [
        wrap_angle(-0.01),
        wrap_angle(0.01),
        -0.01,
        0.01,
    ]
)
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
    jitted_mpc_loop, (x0_init, u_init, z_init, l_init), xs=None, length=800
)
plt.plot(mpc_x[:, 0])
plt.plot(mpc_x[:, 1])
plt.show()
plt.plot(mpc_u[:, 0])
plt.plot(mpc_u[:, 1])
plt.show()
print(jnp.sum(n_iterations))