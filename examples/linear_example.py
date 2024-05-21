import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics, condense_OCP_to_QP
from jax import lax, debug
from admm_noc.utils import wrap_angle, rollout
from admm_noc.par_admm_lin_opt_con import par_admm_lin
import time
from jax import jacrev, vmap
from admm_noc.optimal_control_problem import ADMM_LIN_OCP
from jaxopt import BoxOSQP


# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")


def projection(z):
    control_ub = jnp.array([2.5])
    control_lb = jnp.array([-2.5])
    return jnp.clip(z, control_lb, control_ub)

def ode(state: jnp.ndarray, control: jnp.ndarray):
    Ac = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Bc = jnp.array([[0.0], [1.0]])
    return Ac @ state + Bc @ control

def stage_cost(x, u):
    Q = jnp.diag(jnp.array([1e2, 1e0]))
    R = 1e-1 * jnp.eye(1)
    return 0.5 * x.T @ Q @ x + 0.5 * u.T @ R @ u

def final_cost(x):
    P = jnp.diag(jnp.array([1e2, 1e0]))
    return 0.5 * x.T @ P @ x

def total_cost(x, u):
    ct = vmap(stage_cost)(x[:-1], u)
    cT = final_cost(x[-1])
    return cT + jnp.sum(ct)



Q = jnp.diag(jnp.array([1e2, 1e0]))
R = 1e-1 * jnp.eye(1)
P = jnp.diag(jnp.array([1e2, 1e0]))

Ts = 0.1
N = 60
downsampling = 1
dynamics = discretize_dynamics(ode, Ts, downsampling)
max_it = 1000

x0 = jnp.array([2.0, 1.0])
u = jnp.zeros((N, 1))
x = rollout(dynamics, u, x0)
z = jnp.zeros((N, u.shape[1]))
l = jnp.zeros((N, u.shape[1]))

Ad = jacrev(dynamics, 0)(x[0], u[0])
Bd = jacrev(dynamics, 1)(x[0], u[0])

# define and solve via OSQP
H, g, C_u, u_lim = condense_OCP_to_QP(Ad, Bd, P, Q, R, N, jnp.eye(1), jnp.array([2.5]), jnp.array([-2.5]))
l_lim = -jnp.inf*jnp.ones(u_lim.shape)


qp_cg = BoxOSQP(momentum=1., rho_start=0.1, rho_min=0.1, rho_max=0.1, stepsize_updates_frequency=0, eq_qp_solve='cg', jit=True, maxiter=max_it)
u_cg = qp_cg.run(params_obj=(H, g @ x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]

start = time.time()
u_cg = qp_cg.run(params_obj=(H, g @ x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]
jax.block_until_ready(u_cg)
end = time.time()
osqp_time_cg = end - start


qp_cg_jacobi = BoxOSQP(momentum=1., rho_start=0.1, rho_min=0.1, rho_max=0.1, stepsize_updates_frequency=0, eq_qp_solve='cg+jacobi', jit=True, maxiter=max_it)
u_cg_jacobi = qp_cg_jacobi.run(params_obj=(H, g @ x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]

start = time.time()
u_cg_jacobi = qp_cg_jacobi.run(params_obj=(H, g @ x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]
jax.block_until_ready(u_cg_jacobi)
end = time.time()
osqp_time_cg_jacobi = end - start


qp_lu = BoxOSQP(momentum=1., rho_start=0.1, rho_min=0.1, rho_max=0.1, stepsize_updates_frequency=0, eq_qp_solve='lu', jit=True, maxiter=max_it)
u_lu = qp_lu.run(params_obj=(H, g @ x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]

start = time.time()
u_lu = qp_lu.run(params_obj=(H, g @ x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]
jax.block_until_ready(u_lu)
end = time.time()
osqp_time_lu = end - start

# define and solve via admm
Q = jnp.kron(jnp.ones((N, 1, 1)), Q)
R = jnp.kron(jnp.ones((N, 1, 1)), R)
Ad = jnp.kron(jnp.ones((N, 1, 1)), Ad)
Bd = jnp.kron(jnp.ones((N, 1, 1)), Bd)

penalty_parameter = 0.1

admm_ocp = ADMM_LIN_OCP(Ad, Bd, P, Q, R, projection, penalty_parameter)

anon_par_admm_lin = lambda x, u, z, l: par_admm_lin(admm_ocp, x, u, z, l, max_it)
_jitted_par_admm_lin = jax.jit(anon_par_admm_lin)

opt_x, opt_u, _, _ = _jitted_par_admm_lin( x, u, z, l)
start = time.time()
opt_x, opt_u, _, _ = _jitted_par_admm_lin( x, u, z, l)
jax.block_until_ready(opt_u)
end = time.time()
par_admm_time = end - start

u_cg = u_cg.reshape(-1, 1)
u_cg_jacobi = u_cg_jacobi.reshape(-1, 1)
u_lu = u_lu.reshape(-1, 1)
x_cg = rollout(dynamics, u_cg, x0)
x_cg_jacobi = rollout(dynamics, u_cg_jacobi, x0)
x_lu = rollout(dynamics, u_lu, x0)

print('Iterations: ', max_it)
print('OSQP time cg         :', osqp_time_cg,        ', cost:', total_cost(x_cg, u_cg))
print('OSQP time cg+jacobi  :', osqp_time_cg_jacobi, ', cost:', total_cost(x_cg_jacobi, u_cg_jacobi))
print('OSQP time lu         :', osqp_time_lu,        ', cost:', total_cost(x_lu, u_lu))
print('Par ADMM time        :', par_admm_time,       ', cost:', total_cost(opt_x, opt_u) )

# plt.plot(opt_u, label='par_admm')
# plt.plot(u_cg, label='osqp_cg')
# plt.plot(u_cg_jacobi, label='osqp_cg_jacobi')
# plt.plot(u_lu, label='osqp_lu')
# plt.legend()
# plt.show()



