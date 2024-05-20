import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics, get_QP_problem
from jax import lax, debug
from admm_noc.utils import wrap_angle, rollout
from admm_noc.par_admm_optimal_control import par_admm
import time
from jax import jacrev
from jaxopt import BoxOSQP


# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")


def projection(z):
    control_ub = jnp.array([jnp.inf, jnp.inf, 2.5])
    control_lb = jnp.array([-jnp.inf, -jnp.inf, -2.5])
    return jnp.clip(z, control_lb, control_ub)

def ode(state: jnp.ndarray, control: jnp.ndarray):
    Ac = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Bc = jnp.array([[0.0], [1.0]])
    return Ac @ state + Bc @ control



def transient_cost(state: jnp.ndarray, control: jnp.ndarray):
    X = jnp.diag(jnp.array([1e2, 1e0]))
    U = 1e-1 * jnp.eye(control.shape[0])
    return 0.5 * state.T @ X @ state + 0.5 * control.T @ U @ control


def final_cost(state: jnp.ndarray):
    P = jnp.diag(jnp.array([1e2, 1e0]))
    return 0.5 * state.T @ P @ state


def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(transient_cost)(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


step = 0.1
horizon = 60
downsampling = 1
dynamics = discretize_dynamics(ode, step, downsampling)
x0 = jnp.array([2.0, 1.0])
u = jnp.zeros((horizon, 1))
x = rollout(dynamics, u, x0)
z = jnp.zeros((horizon, u.shape[1] + x.shape[1]))
l = jnp.zeros((horizon, u.shape[1] + x.shape[1]))
sigma = 0.1

anon_par_admm = lambda x, u, z, l, sigma: par_admm(transient_cost, final_cost, dynamics, projection, x, u, z, l, sigma)
_jitted_par_admm = jax.jit(anon_par_admm)

opt_x, opt_u, _, _ = _jitted_par_admm(
     x, u, z, l, sigma
)
start = time.time()
opt_x, opt_u, _, _ = _jitted_par_admm(
     x, u, z, l, sigma
)
jax.block_until_ready(opt_x)
end = time.time()
par_time = end-start

####################################################################################################################
# define batch problem
N = horizon
QN = jnp.diag(jnp.array([1e2, 1e0]))
Q = jnp.diag(jnp.array([1e2, 1e0]))
R = 1e-1 * jnp.eye(1)
x0 = jnp.array([2.0, 1.0])
Ad = jacrev(dynamics, 0)(jnp.array([0., 0.]), jnp.array([0.]))
Bd = jacrev(dynamics, 1)(jnp.array([0., 0.]), jnp.array([0.]))
umin = -2.5
umax = 2.5

H, g, A, upper = get_QP_problem(Ad, Bd, QN, Q, R, N, jnp.eye(1), umax, umin)
lower = -jnp.inf * jnp.ones(upper.shape[0])
qp = BoxOSQP(jit=True, maxiter=50, rho_start=0.1, rho_max=0.1, rho_min=0.1,  sigma=1e-32, stepsize_updates_frequency=1)
sol = qp.run(params_obj=(H, g@x0), params_eq=A, params_ineq=(lower, upper))

start = time.time()
batch_sol = qp.run(params_obj=(H, g@x0), params_eq=A, params_ineq=(lower, upper))
jax.block_until_ready(batch_sol.params.primal[0])
end = time.time()
batch_time = end-start
u_batch = batch_sol.params.primal[0].reshape(-1, 1)
x_batch = rollout(dynamics, u_batch, x0)

par_cost = total_cost(opt_x, opt_u)
batch_cost = total_cost(x_batch, u_batch)




print('par time           : ', par_time)
print('batch time         : ', batch_time)
print('|u_par  - u_batch| : ', jnp.max(jnp.abs(opt_u-u_batch)))
print('batch cost', batch_cost)
print('par cost', par_cost)