import jax.numpy as jnp
from jax import vmap, debug
from jax import lax
from admm_noc.optimal_control_problem import ADMM_LIN_OCP
from paroc.lqt_problem import LQT
from paroc import seq_bwd_pass, seq_fwd_pass


def create_lqt(
    ocp: ADMM_LIN_OCP,
    z: jnp.ndarray,
    lamda: jnp.ndarray
):
    T = ocp.Q.shape[0]
    nx = ocp.Q.shape[1]
    nu = ocp.R.shape[1]

    def offsets(Ut, rut):
        st = -jnp.linalg.solve(Ut, rut)
        return st

    R = ocp.R + jnp.kron(jnp.ones((T, 1, 1)), ocp.penalty_parameter / 2)
    s = vmap(offsets)(R, lamda - ocp.penalty_parameter * z)
    H = jnp.eye(nx)
    HT = H
    H = jnp.kron(jnp.ones((T, 1, 1)), H)
    Z = jnp.eye(nu)
    Z = jnp.kron(jnp.ones((T, 1, 1)), Z)
    XT = ocp.P
    rT = jnp.zeros(nx)
    c = jnp.zeros((T, nx))
    r = jnp.zeros((T, nx))
    M = jnp.kron(jnp.ones((T, 1, 1)), jnp.zeros((nx, nu)))
    lqt = LQT(ocp.Ad, ocp.Bd, c, XT, HT, rT, ocp.Q, H, r, R, Z, s, M)
    return lqt


def argmin_xu(
    ocp: ADMM_LIN_OCP,
    states: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    lqt = create_lqt(ocp, consensus, dual)
    Kx_par, d_par, S_par, v_par = seq_bwd_pass(lqt)
    controls, states = seq_fwd_pass(lqt, states[0], Kx_par, d_par)
    return states, controls


def argmin_z(
    ocp: ADMM_LIN_OCP, controls: jnp.ndarray, dual: jnp.ndarray
):
    z = controls + 1 / ocp.penalty_parameter * dual
    return vmap(ocp.projection)(z)


def grad_ascent(
    ocp: ADMM_LIN_OCP,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    return dual + ocp.penalty_parameter * (
        controls - consensus
    )


def primal_residual(controls: jnp.ndarray, consensus: jnp.ndarray):
    return jnp.max(jnp.abs(controls - consensus))


def seq_admm_lin(
    ocp: ADMM_LIN_OCP,
    states0: jnp.ndarray,
    controls0: jnp.ndarray,
    consensus0: jnp.ndarray,
    dual0: jnp.ndarray,
    max_it: int
):

    def admm_iteration(val):
        x, u, z, l, _, _, it_cnt = val
        # debug.print('iteration     {x}', x=it_cnt)
        next_x, next_u = argmin_xu(ocp, x, z, l)

        prev_z = z
        z = argmin_z(ocp, next_u, l)

        l = grad_ascent(ocp, next_u, z, l)

        rp_infty = primal_residual(u, z)
        rd_infty = jnp.max(jnp.abs(z - prev_z))
        it_cnt += 1
        # debug.print('|rp|_inf      {x}', x=rp_infty)
        # debug.print('|rd|_inf      {x}', x=rd_infty)
        # debug.print('------------------------------')
        # debug.breakpoint()
        return next_x, next_u, z, l, rp_infty, rd_infty, it_cnt

    def admm_conv(val):
        _, _, _, _, rp_infty, rd_infty, it_cnt = val
        # exit_condition = jnp.logical_and(rp_infty < 1e-3, rd_infty < 1e-3)
        # exit_condition = jnp.logical_or(exit_condition, it_cnt > max_it)
        # return jnp.logical_not(exit_condition)
        return it_cnt < max_it

    (
        opt_states,
        opt_controls,
        opt_consensus,
        opt_dual,
        _,
        _,
        iterations,
    ) = lax.while_loop(
        admm_conv,
        admm_iteration,
        (states0, controls0, consensus0, dual0, jnp.inf, jnp.inf, 0.0),
    )
    # debug.print("iterations      {x}", x=iterations)
    # debug.print("------------------------------")
    # debug.breakpoint()
    return opt_states, opt_controls, opt_consensus, opt_dual
