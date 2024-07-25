import jax.numpy as jnp
import jax.scipy as jcp
from jax import grad, hessian, jacrev
from jax import vmap, debug
from jax import lax
from admm_noc.optimal_control_problem import ADMM_OCP, Derivatives
from typing import Callable


def compute_derivatives(
    ocp: ADMM_OCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    def body(x, u, z, l):
        cx_k, cu_k = grad(ocp.stage_cost, (0, 1))(x, u, z, l)
        cxx_k = hessian(ocp.stage_cost, 0)(x, u, z, l)
        cuu_k = hessian(ocp.stage_cost, 1)(x, u, z, l)
        cxu_k = jacrev(jacrev(ocp.stage_cost, 0), 1)(x, u, z, l)
        fx_k = jacrev(ocp.dynamics, 0)(x, u)
        fu_k = jacrev(ocp.dynamics, 1)(x, u)
        fxx_k = jacrev(jacrev(ocp.dynamics, 0), 0)(x, u)
        fuu_k = jacrev(jacrev(ocp.dynamics, 1), 1)(x, u)
        fxu_k = jacrev(jacrev(ocp.dynamics, 0), 1)(x, u)
        return cx_k, cu_k, cxx_k, cuu_k, cxu_k, fx_k, fu_k, fxx_k, fuu_k, fxu_k

    cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu = vmap(body)(
        states[:-1], controls, consensus, dual
    )
    return Derivatives(cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu)


def bwd_pass(
    final_cost: Callable,
    final_state: jnp.ndarray,
    d: Derivatives,
    reg_param: float,
):
    # grad_cost_norm = jnp.linalg.norm(d.cu)
    # reg_param = reg_param * grad_cost_norm

    def body(carry, inp):
        Vx, Vxx = carry
        cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu = inp

        Qx = cx + fx.T @ Vx
        Qu = cu + fu.T @ Vx
        Qxx = cxx + fx.T @ Vxx @ fx + jnp.tensordot(Vx, fxx, axes=1)
        Qxu = cxu + fx.T @ Vxx @ fu + jnp.tensordot(Vx, fxu, axes=1)
        Quu = cuu + fu.T @ Vxx @ fu + jnp.tensordot(Vx, fuu, axes=1)
        Quu = Quu + reg_param * jnp.eye(Quu.shape[0])
        eig_vals, _ = jnp.linalg.eigh(Quu)
        pos_def = jnp.all(eig_vals > 0)

        k = -jcp.linalg.solve(Quu, Qu)
        K = -jcp.linalg.solve(Quu, Qxu.T)

        dV = -0.5 * Qu @ jcp.linalg.solve(Quu, Qu)
        Vx = Qx - Qu @ jcp.linalg.solve(Quu, Qxu.T)
        Vxx = Qxx - Qxu @ jcp.linalg.solve(Quu, Qxu.T)
        return (Vx, Vxx), (k, K, dV, pos_def, Qu)

    Vx_final = grad(final_cost)(final_state)
    Vxx_final = hessian(final_cost)(final_state)

    _, (ffgain, gain, cost_diff, feasible_bwd_pass, Hu) = lax.scan(
        body,
        (Vx_final, Vxx_final),
        (d.cx, d.cu, d.cxx, d.cuu, d.cxu, d.fx, d.fu, d.fxx, d.fuu, d.fxu),
        reverse=True,
    )
    pred_reduction = jnp.sum(cost_diff)
    feasible_bwd_pass = jnp.all(feasible_bwd_pass)

    return ffgain, gain, pred_reduction, feasible_bwd_pass, Hu



def nonlin_rollout(
    ocp: ADMM_OCP,
    gain: jnp.ndarray,
    ffgain: jnp.ndarray,
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
):
    def body(x_hat, inp):
        K, k, x, u = inp
        u_hat = u + k + K @ (x_hat - x)
        next_x_hat = ocp.dynamics(x_hat, u_hat)
        return next_x_hat, (x_hat, u_hat)

    new_final_state, (new_states, new_controls) = lax.scan(
        body, nominal_states[0], (gain, ffgain, nominal_states[:-1], nominal_controls)
    )
    new_states = jnp.vstack((new_states, new_final_state))
    return new_states, new_controls


def argmin_xu(
    ocp: ADMM_OCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    mu0 = 1.
    nu0 = 2.0

    def while_body(val):
        x, u, z, l, iteration_counter, reg_param, reg_inc, _ = val
        # debug.print("Iteration:    {x}", x=it_cnt)

        cost = ocp.total_cost(x, u, z, l)
        # debug.print("cost:         {x}", x=cost)

        derivatives = compute_derivatives(ocp, x, u, z, l)

        def while_inner_loop(inner_val):
            _, _, _, _, rp, r_inc, inner_it_counter = inner_val
            ffgain, gain, pred_reduction, feasible_bwd_pass, Hu = bwd_pass(
                ocp.final_cost, x[-1], derivatives, rp
            )
            temp_x, temp_u = nonlin_rollout(ocp, gain, ffgain, x, u)
            Hu_norm = jnp.max(jnp.abs(Hu))
            new_cost = ocp.total_cost(temp_x, temp_u, z, l)
            actual_reduction = new_cost - cost
            gain_ratio = actual_reduction / pred_reduction
            succesful_minimzation = jnp.logical_and(gain_ratio > 0, feasible_bwd_pass)
            rp = jnp.where(
                succesful_minimzation,
                rp * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                rp * reg_inc,
            )
            r_inc = jnp.where(succesful_minimzation, 2.0, 2 * r_inc)
            rp = jnp.clip(rp, 1e-16, 1e16)
            inner_it_counter += 1
            return (
                temp_x,
                temp_u,
                succesful_minimzation,
                Hu_norm,
                rp,
                r_inc,
                inner_it_counter,
            )

        def while_inner_cond(inner_val):
            _, _, succesful_minimzation, _, _, _, inner_it_counter = inner_val
            exit_cond = jnp.logical_or(
                succesful_minimzation, inner_it_counter > 500
            )
            return jnp.logical_not(exit_cond)

        x, u, _, Hamiltonian_norm, reg_param, reg_inc, _ = lax.while_loop(
            while_inner_cond,
            while_inner_loop,
            (x, u, jnp.bool_(0.0), 0.0, reg_param, reg_inc, 0),
        )
        iteration_counter += 1
        return x, u, z, l, iteration_counter, reg_param, reg_inc, Hamiltonian_norm

    def while_cond(val):
        _, _, _, _, iteration_counter, _, _, Hu_norm = val
        exit_cond = jnp.logical_or(Hu_norm < 1e-4, iteration_counter > 500)
        return jnp.logical_not(exit_cond)

    opt_x, opt_u, _, _, iterations, _, _, _ = lax.while_loop(
        while_cond,
        while_body,
        (states, controls, consensus, dual, 0, mu0, nu0, jnp.array(1.0))
    )
    # debug.print('{x}', x = iterations)
    # jax.debug.breakpoint()
    return opt_x, opt_u, iterations


def argmin_z(
    ocp: ADMM_OCP, states: jnp.ndarray, controls: jnp.ndarray, dual: jnp.ndarray
):
    z = jnp.hstack((states[:-1], controls)) + 1 / ocp.penalty_parameter * dual
    return vmap(ocp.projection)(z)


def grad_ascent(
    ocp: ADMM_OCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    return dual + ocp.penalty_parameter * (
        jnp.hstack((states[:-1], controls)) - consensus
    )


def primal_residual(states: jnp.ndarray, controls: jnp.ndarray, dual: jnp.ndarray):
    return jnp.max(jnp.abs(jnp.hstack((states[:-1], controls)) - dual))


def ddp_admm(
    stage_cost: Callable,
    final_cost: Callable,
    dynamics: Callable,
    projection: Callable,
    states0: jnp.ndarray,
    controls0: jnp.ndarray,
    consensus0: jnp.ndarray,
    dual0: jnp.ndarray,
    penalty_param: float,
):
    def admm_stage_cost(x, u, z, l):
        y = jnp.hstack((x, u))
        sql2norm = (y - z + 1.0 / penalty_param * l).T @ (
            y - z + 1.0 / penalty_param * l
        )
        return stage_cost(x, u) + penalty_param / 2 * sql2norm

    def admm_total_cost(states, controls, consensus, dual):
        ct = vmap(admm_stage_cost)(states[:-1], controls, consensus, dual)
        cT = final_cost(states[-1])
        return cT + jnp.sum(ct)

    admm_ocp = ADMM_OCP(
        dynamics,
        projection,
        admm_stage_cost,
        final_cost,
        admm_total_cost,
        penalty_param,
    )

    def admm_iteration(val):
        x, u, z, l, _, _, it_cnt = val
        # debug.print('iteration     {x}', x=it_cnt)
        x, u, it = argmin_xu(admm_ocp, x, u, z, l)
        it_cnt += it
        prev_z = z
        z = argmin_z(admm_ocp, x, u, l)

        l = grad_ascent(admm_ocp, x, u, z, l)

        rp_infty = primal_residual(x, u, z)
        rd_infty = jnp.max(jnp.abs(z - prev_z))
        it_cnt += 1
        return x, u, z, l, rp_infty, rd_infty, it_cnt

    def admm_conv(val):
        _, _, _, _, rp_infty, rd_infty, _ = val
        exit_condition = jnp.logical_and(rp_infty < 1e-2, rd_infty < 1e-2)
        return jnp.logical_not(exit_condition)
        # return it_cnt < 50

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
    return opt_states, opt_controls, opt_consensus, opt_dual, iterations
