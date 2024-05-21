import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics, condense_OCP_to_QP
from jax import lax, debug
from admm_noc.utils import wrap_angle, rollout
from admm_noc.par_admm_lin_opt_con import par_admm_lin
import time
from jax import jacrev
from admm_noc.optimal_control_problem import ADMM_LIN_OCP
from jaxopt import BoxOSQP


# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")


def projection(z):
    control_ub = jnp.array([2.5])
    control_lb = jnp.array([-2.5])
    return jnp.clip(z, control_lb, control_ub)

def ode(state: jnp.ndarray, control: jnp.ndarray):
    Ac = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Bc = jnp.array([[0.0], [1.0]])
    return Ac @ state + Bc @ control


Q = jnp.diag(jnp.array([1e2, 1e0]))
R = 1e-1 * jnp.eye(1)
P = jnp.diag(jnp.array([1e2, 1e0]))

Ts = 0.1
N = 60
downsampling = 1
dynamics = discretize_dynamics(ode, Ts, downsampling)


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
qp = BoxOSQP(momentum=1., rho_start=0.1, rho_min=0.1, rho_max=0.1, stepsize_updates_frequency=1e32, eq_qp_solve='cg')
sol = qp.run(params_obj=(H, g@x0), params_eq=C_u, params_ineq=(l_lim, u_lim)).params.primal[0]


# define and solve via admm
Q = jnp.kron(jnp.ones((N, 1, 1)), Q)
R = jnp.kron(jnp.ones((N, 1, 1)), R)
Ad = jnp.kron(jnp.ones((N, 1, 1)), Ad)
Bd = jnp.kron(jnp.ones((N, 1, 1)), Bd)

penalty_parameter = 0.1

admm_ocp = ADMM_LIN_OCP(Ad, Bd, P, Q, R, projection, penalty_parameter)


opt_x, opt_u, _, _ = par_admm_lin(admm_ocp, x, u, z, l)

plt.plot(opt_u)
plt.plot(sol)
plt.show()
