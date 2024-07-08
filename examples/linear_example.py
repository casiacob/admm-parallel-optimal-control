import jax.numpy as jnp
import jax.random
from jax import config
import matplotlib.pyplot as plt
from admm_noc.utils import discretize_dynamics
from jax import lax, debug
from admm_noc.utils import rollout
from admm_noc.par_admm_lin_opt_con import par_admm_lin
from admm_noc.seq_admm_lin_opt_con import seq_admm_lin
import time
from jax import jacrev, vmap
from admm_noc.optimal_control_problem import ADMM_LIN_OCP
import pandas as pd

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

Ts = [0.1, 0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001]
N = [60, 120, 240, 480, 600, 1200, 2400, 4800, 6000]
seq_time_means = []
seq_time_medians = []
par_time_means = []
par_time_medians = []

for sampling_period, horizon in zip(Ts, N):
    seq_time_Ts = []
    par_time_Ts = []
    for i in range(10):
        downsampling = 1
        dynamics = discretize_dynamics(ode, sampling_period, downsampling)
        max_it = 10

        x0 = jnp.array([2.0, 1.0])
        u = jnp.zeros((horizon, 1))
        x = rollout(dynamics, u, x0)
        z = jnp.zeros((horizon, u.shape[1]))
        l = jnp.zeros((horizon, u.shape[1]))

        Ad = jacrev(dynamics, 0)(x[0], u[0])
        Bd = jacrev(dynamics, 1)(x[0], u[0])

        Q = jnp.diag(jnp.array([1e2, 1e0]))
        R = 1e-1 * jnp.eye(1)
        P = jnp.diag(jnp.array([1e2, 1e0]))

        # define and solve via admm
        Q = jnp.kron(jnp.ones((horizon, 1, 1)), Q)
        R = jnp.kron(jnp.ones((horizon, 1, 1)), R)
        Ad = jnp.kron(jnp.ones((horizon, 1, 1)), Ad)
        Bd = jnp.kron(jnp.ones((horizon, 1, 1)), Bd)

        penalty_parameter = 0.1

        admm_ocp = ADMM_LIN_OCP(Ad, Bd, P, Q, R, projection, penalty_parameter)

        anon_par_admm_lin = lambda x, u, z, l: par_admm_lin(admm_ocp, x, u, z, l, max_it)
        _jitted_par_admm_lin = jax.jit(anon_par_admm_lin)

        par_x, par_u, _, _ = _jitted_par_admm_lin(x, u, z, l)
        start = time.time()
        par_x, par_u, _, _ = _jitted_par_admm_lin(x, u, z, l)
        jax.block_until_ready(par_u)
        end = time.time()

        par_admm_time = end - start

        anon_seq_admm_lin = lambda x, u, z, l: seq_admm_lin(admm_ocp, x, u, z, l, max_it)
        _jitted_seq_admm_lin = jax.jit(anon_seq_admm_lin)

        seq_x, seq_u, _, _ = _jitted_seq_admm_lin(x, u, z, l)
        start = time.time()
        seq_x, seq_u, _, _ = _jitted_seq_admm_lin(x, u, z, l)
        jax.block_until_ready(seq_u)
        end = time.time()
        seq_admm_time = end - start

        seq_time_Ts.append(seq_admm_time)
        par_time_Ts.append(par_admm_time)

    seq_time_means.append(jnp.mean(jnp.array(seq_time_Ts)))
    seq_time_medians.append(jnp.median(jnp.array(seq_time_Ts)))
    par_time_means.append(jnp.mean(jnp.array(par_time_Ts)))
    par_time_medians.append(jnp.median(jnp.array(par_time_Ts)))

seq_time_means_arr = jnp.array(seq_time_means)
seq_time_medians_arr = jnp.array(seq_time_medians)
par_time_means_arr = jnp.array(par_time_means)
par_time_medians_arr = jnp.array(par_time_medians)

df_mean_seq = pd.DataFrame(seq_time_means_arr)
df_median_seq = pd.DataFrame(seq_time_medians_arr)
df_mean_par = pd.DataFrame(par_time_means_arr)
df_median_par = pd.DataFrame(par_time_medians_arr)

df_mean_seq.to_csv("admm_seq_means.csv")
df_median_seq.to_csv("admm_seq_median.csv")
df_mean_par.to_csv("admm_par_means.csv")
df_median_par.to_csv("admm_par_median.csv")






