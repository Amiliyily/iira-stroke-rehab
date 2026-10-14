"""Minimum-jerk reaching references (SPEC 2.3).

8 directions (horizontal plane, 45 deg apart) x 5 distances
{0.10,0.14,0.18,0.22,0.26} m, T = 2.0 s fixed, dt = 0.01 s.
Joint reference from damped-least-squares IK (lam = 1e-3) propagated along the
path; joint velocities from the DLS Jacobian pseudo-inverse; accelerations by
central differences; feedforward torque tau_ff = rne(q_ref, qd_ref, qdd_ref).
Targets are reachability-checked; unreachable directions are shortened in
1 cm steps and the actual distance recorded.
"""

import numpy as np
import arm6dof as arm

DT = 0.01
T_DEFAULT = 2.0
DIRECTIONS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
DISTANCES = [0.10, 0.14, 0.18, 0.22, 0.26]
IK_LAM = 1e-3
REACH_TOL = 2e-3   # m


def min_jerk(T, dt=DT):
    """s(t), ds(t) for 10(t/T)^3 - 15(t/T)^4 + 6(t/T)^5."""
    t = np.arange(0.0, T + 0.5 * dt, dt)
    u = t / T
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    ds = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
    return t, s, ds


def _dls_step(q, v_task, lam=IK_LAM):
    J = arm.jacobian(q)[:3]
    A = J @ J.T + lam ** 2 * np.eye(3)
    return J.T @ np.linalg.solve(A, v_task)


def ik_point(q_init, p_target, lam=IK_LAM, iters=400, tol=1e-6):
    q = q_init.copy()
    for _ in range(iters):
        p, _ = arm.fkine(q)
        err = p_target - p
        if np.linalg.norm(err) < tol:
            break
        q = q + _dls_step(q, err, lam)
    p, _ = arm.fkine(q)
    return q, float(np.linalg.norm(p_target - p))


def build_reference(direction_deg, d_req, T=T_DEFAULT, dt=DT):
    """One reaching reference; shortens d until reachable (records actual d)."""
    p0, _ = arm.fkine(arm.Q0_HOME)
    phi = np.deg2rad(direction_deg)
    u = np.array([np.cos(phi), np.sin(phi), 0.0])
    d = d_req
    reachable = True
    while True:
        _, err = ik_point(arm.Q0_HOME, p0 + d * u)
        if err < REACH_TOL or d <= 0.05:
            reachable = err < REACH_TOL
            break
        d -= 0.01
    t, s, ds = min_jerk(T, dt)
    N = len(t)
    p_ref = p0[None, :] + d * np.outer(s, u)
    v_ref = d * np.outer(ds, u)
    q_ref = np.zeros((N, 6))
    qd_ref = np.zeros((N, 6))
    q = arm.Q0_HOME.copy()
    for k in range(N):
        q, _ = ik_point(q, p_ref[k], iters=20, tol=1e-8)
        q_ref[k] = q
        qd_ref[k] = _dls_step(q, v_ref[k])
    qdd_ref = np.gradient(qd_ref, dt, axis=0, edge_order=2)
    tau_ff = np.zeros((N, 6))
    for k in range(N):
        tau_ff[k] = arm.rne(q_ref[k], qd_ref[k], qdd_ref[k])
    return dict(t=t, p_ref=p_ref, v_ref=v_ref, q_ref=q_ref, qd_ref=qd_ref,
                qdd_ref=qdd_ref, tau_ff=tau_ff, direction=float(direction_deg),
                d_req=float(d_req), d_actual=float(d), reachable=bool(reachable),
                T=float(T), dt=float(dt))


def build_all_references():
    refs = []
    for direc in DIRECTIONS_DEG:
        for d in DISTANCES:
            refs.append(build_reference(direc, d))
    return refs


def test_references(refs):
    """Test protocol (SPEC 6): 8 directions at the middle distance d = 0.18 m."""
    return [r for r in refs if abs(r["d_req"] - 0.18) < 1e-9]
