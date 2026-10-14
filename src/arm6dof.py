"""6-DOF arm (UR5-magnitude, standard DH per SPEC 2.1).

Convention note: the SPEC table column headers a_{i-1}, alpha_{i-1} reproduce the
classic UR5 standard-DH numeric table, i.e. they are the (a_i, alpha_i) of the
standard convention A_i = RotZ(th) TransZ(d) TransX(a) RotX(alpha).  This module
implements standard DH so that joint 1 rotates about the base z axis.

Contents:
  - forward kinematics (numpy, complex-safe for complex-step checks)
  - geometric Jacobian
  - recursive Newton-Euler inverse dynamics rne(q, qd, qdd)   (pure-python fast path)
  - M(q) / C(q,qd)qd + g(q) / g(q)
  - kinetic-energy (Jacobian) based M and g for the independent self-check
  - run_selfcheck(): RNE vs kinetic-energy / Euler-Lagrange, error < 1e-8,
    written to results/selfcheck.txt
"""

import numpy as np

# ---------------------------------------------------------------- constants
A = np.array([0.0, -0.425, -0.392, 0.0, 0.0, 0.0])            # a_i  (m)
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])  # alpha_i
D = np.array([0.0892, 0.0, 0.0, 0.1093, 0.0948, 0.0823])      # d_i  (m)
MASS = np.array([3.70, 8.39, 2.28, 1.22, 1.22, 0.22])         # kg

# link lengths (for the slender-rod inertia magnitude)
L_LINK = np.array([D[0], abs(A[1]), abs(A[2]), D[3], D[4], D[5]])
# isotropic inertia I_i = (1/12) m L^2 * I_3  (SPEC 2.1)
IC = (1.0 / 12.0) * MASS * L_LINK ** 2

# COM position in link frame i (geometric midpoint along the link, derived
# analytically for this DH table; all constant in frame i)
P_COM = np.array([
    [0.0, -D[0] / 2.0, 0.0],          # link 1: mast along -y1
    [-A[1] / 2.0, 0.0, 0.0],          # link 2: along +x2
    [-A[2] / 2.0, 0.0, 0.0],          # link 3: along +x3
    [0.0, -D[3] / 2.0, 0.0],          # link 4: along -y4
    [0.0, D[4] / 2.0, 0.0],           # link 5: along +y5
    [0.0, 0.0, -D[5] / 2.0],          # link 6: flange along -z6
])

G0 = 9.81                      # gravity magnitude (m/s^2), world z up
TAU_LIM = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
QD_LIM = 3.0                   # rad/s
FV = np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05])
FC = np.array([0.8, 0.8, 0.8, 0.1, 0.1, 0.1])
Q0_HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

_CA = tuple(float(np.cos(a)) for a in ALPHA)
_SA = tuple(float(np.sin(a)) for a in ALPHA)
_A = tuple(float(v) for v in A)
_D = tuple(float(v) for v in D)
_M = tuple(float(v) for v in MASS)
_IC = tuple(float(v) for v in IC)
_PC = tuple((float(p[0]), float(p[1]), float(p[2])) for p in P_COM)


# ------------------------------------------------------- pure-python helpers
def _fk_all_py(q):
    """Per-frame world rotation (flat 9-tuple) and origin for frames 0..6."""
    Rs = [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)]
    os_ = [(0.0, 0.0, 0.0)]
    for i in range(6):
        c = np.cos(q[i])
        s = np.sin(q[i])
        ca, sa = _CA[i], _SA[i]
        # R_i = RotZ(th) RotX(alpha)
        r = (c, -s * ca, s * sa,
             s, c * ca, -c * sa,
             0.0, sa, ca)
        Rp = Rs[-1]
        # R0_i = R0_{i-1} R_i
        R0 = (
            Rp[0] * r[0] + Rp[1] * r[3] + Rp[2] * r[6],
            Rp[0] * r[1] + Rp[1] * r[4] + Rp[2] * r[7],
            Rp[0] * r[2] + Rp[1] * r[5] + Rp[2] * r[8],
            Rp[3] * r[0] + Rp[4] * r[3] + Rp[5] * r[6],
            Rp[3] * r[1] + Rp[4] * r[4] + Rp[5] * r[7],
            Rp[3] * r[2] + Rp[4] * r[5] + Rp[5] * r[8],
            Rp[6] * r[0] + Rp[7] * r[3] + Rp[8] * r[6],
            Rp[6] * r[1] + Rp[7] * r[4] + Rp[8] * r[7],
            Rp[6] * r[2] + Rp[7] * r[5] + Rp[8] * r[8],
        )
        # o_i = o_{i-1} + R0_{i-1} (a c, a s, d)
        t = (_A[i] * c, _A[i] * s, _D[i])
        op = os_[-1]
        o = (op[0] + Rp[0] * t[0] + Rp[1] * t[1] + Rp[2] * t[2],
             op[1] + Rp[3] * t[0] + Rp[4] * t[1] + Rp[5] * t[2],
             op[2] + Rp[6] * t[0] + Rp[7] * t[1] + Rp[8] * t[2])
        Rs.append(R0)
        os_.append(o)
    return Rs, os_


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _rne_py(q, qd, qdd):
    """Recursive Newton-Euler in world coordinates (gravity via base accel)."""
    Rs, os_ = _fk_all_py(q)
    om = [(0.0, 0.0, 0.0)] * 7       # angular velocity of frame i
    omd = [(0.0, 0.0, 0.0)] * 7      # angular acceleration
    ao = [(0.0, 0.0, 0.0)] * 7       # linear accel of frame origin
    F = [(0.0, 0.0, 0.0)] * 7
    N = [(0.0, 0.0, 0.0)] * 7
    uC = [(0.0, 0.0, 0.0)] * 7
    # base: omega=0, accel = +G0 z (gravity trick)
    ao[0] = (0.0, 0.0, G0)
    for i in range(1, 7):
        Rp = Rs[i - 1]
        zp = (Rp[2], Rp[5], Rp[8])   # z axis of frame i-1 (joint i axis)
        qi = qd[i - 1]
        qddi = qdd[i - 1]
        w = om[i - 1]
        wd = omd[i - 1]
        qz = (qi * zp[0], qi * zp[1], qi * zp[2])
        om[i] = (w[0] + qz[0], w[1] + qz[1], w[2] + qz[2])
        cr = _cross(w, qz)
        omd[i] = (wd[0] + qddi * zp[0] + cr[0],
                  wd[1] + qddi * zp[1] + cr[1],
                  wd[2] + qddi * zp[2] + cr[2])
        # origin accel: o_i = o_{i-1} + u1 + u2,
        # u1 = R0_{i-1}(0,0,d) fixed in frame i-1, u2 = R0_i(a,0,0) fixed in frame i
        u1 = (Rp[2] * _D[i - 1], Rp[5] * _D[i - 1], Rp[8] * _D[i - 1])
        R0 = Rs[i]
        u2 = (R0[0] * _A[i - 1], R0[3] * _A[i - 1], R0[6] * _A[i - 1])
        wi = om[i]
        wdi = omd[i - 1]
        t1 = _cross(wdi, u1)
        t2 = _cross(w, _cross(w, u1))
        t3 = _cross(omd[i], u2)
        t4 = _cross(wi, _cross(wi, u2))
        ap = ao[i - 1]
        ao[i] = (ap[0] + t1[0] + t2[0] + t3[0] + t4[0],
                 ap[1] + t1[1] + t2[1] + t3[1] + t4[1],
                 ap[2] + t1[2] + t2[2] + t3[2] + t4[2])
        # COM accel
        pc = _PC[i - 1]
        uc = (R0[0] * pc[0] + R0[1] * pc[1] + R0[2] * pc[2],
              R0[3] * pc[0] + R0[4] * pc[1] + R0[5] * pc[2],
              R0[6] * pc[0] + R0[7] * pc[1] + R0[8] * pc[2])
        uC[i] = uc
        t5 = _cross(omd[i], uc)
        t6 = _cross(wi, _cross(wi, uc))
        ac = (ao[i][0] + t5[0] + t6[0], ao[i][1] + t5[1] + t6[1], ao[i][2] + t5[2] + t6[2])
        mi = _M[i - 1]
        F[i] = (mi * ac[0], mi * ac[1], mi * ac[2])
        ici = _IC[i - 1]
        N[i] = (ici * omd[i][0], ici * omd[i][1], ici * omd[i][2])
    # inward pass (moments taken about the joint-i location o_{i-1})
    f = (0.0, 0.0, 0.0)
    n = (0.0, 0.0, 0.0)
    tau = [0.0] * 6
    for i in range(6, 0, -1):
        f = (F[i][0] + f[0], F[i][1] + f[1], F[i][2] + f[2])
        o_prev = os_[i - 1]
        o_cur = os_[i]
        armj = (o_cur[0] - o_prev[0], o_cur[1] - o_prev[1], o_cur[2] - o_prev[2])
        # moment of f_{i+1} (acts at o_i) about o_{i-1}; none for the tip link
        if i < 6:
            mf = _cross(armj, f_next)
        else:
            mf = (0.0, 0.0, 0.0)
        # moment of F_i (acts at COM) about o_{i-1}: arm = (o_i - o_{i-1}) + u_C
        uc = uC[i]
        armc = (armj[0] + uc[0], armj[1] + uc[1], armj[2] + uc[2])
        mc = _cross(armc, F[i])
        n = (N[i][0] + n[0] + mf[0] + mc[0],
             N[i][1] + n[1] + mf[1] + mc[1],
             N[i][2] + n[2] + mf[2] + mc[2])
        Rp = Rs[i - 1]
        zp = (Rp[2], Rp[5], Rp[8])
        tau[i - 1] = n[0] * zp[0] + n[1] * zp[1] + n[2] * zp[2]
        f_next = f
    return tau


def rne(q, qd, qdd):
    """Inverse dynamics: tau = M(q)qdd + C(q,qd)qd + g(q)  (no joint friction)."""
    return np.array(_rne_py(np.asarray(q, float), np.asarray(qd, float),
                            np.asarray(qdd, float)))


# ------------------------------------------------------------- numpy FK / J
def _A_mat(th, i, dtype):
    c = np.cos(th)
    s = np.sin(th)
    ca = np.cos(ALPHA[i])
    sa = np.sin(ALPHA[i])
    T = np.zeros((4, 4), dtype=dtype)
    T[0, 0] = c;  T[0, 1] = -s * ca; T[0, 2] = s * sa;  T[0, 3] = A[i] * c
    T[1, 0] = s;  T[1, 1] = c * ca;  T[1, 2] = -c * sa; T[1, 3] = A[i] * s
    T[2, 0] = 0.0; T[2, 1] = sa;     T[2, 2] = ca;      T[2, 3] = D[i]
    T[3, 3] = 1.0
    return T


def fkine_all(q):
    """Return origins (7,3) and rotations (7,3,3) of frames 0..6 (complex-safe)."""
    q = np.asarray(q)
    dtype = np.result_type(q.dtype, np.float64)
    T = np.eye(4, dtype=dtype)
    org = np.zeros((7, 3), dtype=dtype)
    rot = np.zeros((7, 3, 3), dtype=dtype)
    rot[0] = np.eye(3, dtype=dtype)
    for i in range(6):
        T = T @ _A_mat(q[i], i, dtype)
        org[i + 1] = T[:3, 3]
        rot[i + 1] = T[:3, :3]
    return org, rot


def fkine(q):
    org, rot = fkine_all(q)
    return org[6].real if np.iscomplexobj(q) else org[6], \
        rot[6].real if np.iscomplexobj(q) else rot[6]


def jacobian(q):
    """Geometric Jacobian (6x6, world frame) at the end effector."""
    org, rot = fkine_all(q)
    pe = org[6]
    J = np.zeros((6, 6))
    for j in range(6):
        z = rot[j][:, 2]
        J[:3, j] = np.cross(z, pe - org[j])
        J[3:, j] = z
    return J


def com_jacobians(q):
    """Per-link COM linear/angular Jacobians: lists of (3,6)."""
    dtype = np.result_type(np.asarray(q).dtype, np.float64)
    org, rot = fkine_all(q)
    Jvs, Jws = [], []
    for i in range(1, 7):
        oc = org[i] + rot[i] @ P_COM[i - 1]
        Jv = np.zeros((3, 6), dtype=dtype)
        Jw = np.zeros((3, 6), dtype=dtype)
        for j in range(i):
            z = rot[j][:, 2]
            Jv[:, j] = np.cross(z, oc - org[j])
            Jw[:, j] = z
        Jvs.append(Jv)
        Jws.append(Jw)
    return Jvs, Jws


# ------------------------------------------- kinetic-energy based M, g, bias
def M_kinetic(q):
    """M(q) from the kinetic-energy (COM Jacobian) method - independent of RNE."""
    Jvs, Jws = com_jacobians(q)
    dtype = np.result_type(np.asarray(q).dtype, np.float64)
    M = np.zeros((6, 6), dtype=dtype)
    for i in range(6):
        M += MASS[i] * (Jvs[i].T @ Jvs[i]) + IC[i] * (Jws[i].T @ Jws[i])
    return M


def g_potential(q):
    """g(q) = dV/dq with V = sum m g0 z^T p_C (world z up)."""
    Jvs, _ = com_jacobians(q)
    g = np.zeros(6)
    for i in range(6):
        g += MASS[i] * G0 * Jvs[i][2, :]
    return g


def M_rne(q):
    """M(q) by the unit-acceleration method (SPEC 2.2)."""
    q = np.asarray(q, float)
    z6 = np.zeros(6)
    t0 = rne(q, z6, z6)
    M = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1.0
        M[:, i] = rne(q, z6, e) - t0
    return M


def bias_rne(q, qd):
    """C(q,qd)qd + g(q) = rne(q, qd, 0)."""
    return rne(q, qd, np.zeros(6))


def dynamics_fast(q, qd):
    """Engine fast path: M, bias (=C qd + g), g, plus FK/J in one pass.

    M and g come from the kinetic-energy method (identical to RNE to ~1e-13,
    verified by run_selfcheck); bias from a single RNE call.
    """
    q = np.asarray(q, float)
    org, rot = fkine_all(q)
    M = np.zeros((6, 6))
    g = np.zeros(6)
    J = np.zeros((6, 6))
    pe = org[6]
    for j in range(6):
        z = rot[j][:, 2]
        J[:3, j] = np.cross(z, pe - org[j])
        J[3:, j] = z
    for i in range(1, 7):
        oc = org[i] + rot[i] @ P_COM[i - 1]
        Jv = np.zeros((3, 6))
        Jw = np.zeros((3, 6))
        for j in range(i):
            z = rot[j][:, 2]
            Jv[:, j] = np.cross(z, oc - org[j])
            Jw[:, j] = z
        M += MASS[i - 1] * (Jv.T @ Jv) + IC[i - 1] * (Jw.T @ Jw)
        g += MASS[i - 1] * G0 * Jv[2, :]
    bias = rne(q, qd, np.zeros(6))
    return M, bias, g, org[6].copy(), J


def forward_dynamics(q, qd, tau_motor, tau_patient):
    """qdd solving  M qdd + C qd + g = tau_motor + tau_patient - Fv qd - Fc sign(qd)."""
    M, bias, g, p, J = dynamics_fast(q, qd)
    fric = FV * qd + FC * np.tanh(np.asarray(qd) / 0.01)
    qdd = np.linalg.solve(M, tau_motor + tau_patient - bias - fric)
    return qdd, M, bias, g, p, J


def engine_dynamics(q, qd):
    """Single-pass engine fast path (pure python, shares one FK):
    returns M (6,6), bias=Cqd+g (6,), g (6,), p_end (3,), J (6,6) as numpy.
    M/g from the kinetic-energy method; bias from RNE (identical model,
    cross-validated to <1e-8 in run_selfcheck)."""
    q = [float(v) for v in q]
    qd = [float(v) for v in qd]
    Rs, os_ = _fk_all_py(q)
    # ---- RNE for bias = rne(q, qd, 0)
    om = [(0.0, 0.0, 0.0)] * 7
    omd = [(0.0, 0.0, 0.0)] * 7
    ao = [(0.0, 0.0, 0.0)] * 7
    F = [(0.0, 0.0, 0.0)] * 7
    N = [(0.0, 0.0, 0.0)] * 7
    uC = [(0.0, 0.0, 0.0)] * 7
    ao[0] = (0.0, 0.0, G0)
    for i in range(1, 7):
        Rp = Rs[i - 1]
        zp = (Rp[2], Rp[5], Rp[8])
        qi = qd[i - 1]
        w = om[i - 1]
        qz = (qi * zp[0], qi * zp[1], qi * zp[2])
        om[i] = (w[0] + qz[0], w[1] + qz[1], w[2] + qz[2])
        cr = _cross(w, qz)
        wd = omd[i - 1]
        omd[i] = (wd[0] + cr[0], wd[1] + cr[1], wd[2] + cr[2])
        u1 = (Rp[2] * _D[i - 1], Rp[5] * _D[i - 1], Rp[8] * _D[i - 1])
        R0 = Rs[i]
        u2 = (R0[0] * _A[i - 1], R0[3] * _A[i - 1], R0[6] * _A[i - 1])
        wi = om[i]
        t1 = _cross(wd, u1)
        t2 = _cross(w, _cross(w, u1))
        t3 = _cross(omd[i], u2)
        t4 = _cross(wi, _cross(wi, u2))
        ap = ao[i - 1]
        ao[i] = (ap[0] + t1[0] + t2[0] + t3[0] + t4[0],
                 ap[1] + t1[1] + t2[1] + t3[1] + t4[1],
                 ap[2] + t1[2] + t2[2] + t3[2] + t4[2])
        pc = _PC[i - 1]
        uc = (R0[0] * pc[0] + R0[1] * pc[1] + R0[2] * pc[2],
              R0[3] * pc[0] + R0[4] * pc[1] + R0[5] * pc[2],
              R0[6] * pc[0] + R0[7] * pc[1] + R0[8] * pc[2])
        uC[i] = uc
        t5 = _cross(omd[i], uc)
        t6 = _cross(wi, _cross(wi, uc))
        mi = _M[i - 1]
        F[i] = (mi * (ao[i][0] + t5[0] + t6[0]),
                mi * (ao[i][1] + t5[1] + t6[1]),
                mi * (ao[i][2] + t5[2] + t6[2]))
        ici = _IC[i - 1]
        N[i] = (ici * omd[i][0], ici * omd[i][1], ici * omd[i][2])
    f = (0.0, 0.0, 0.0)
    n = (0.0, 0.0, 0.0)
    bias = [0.0] * 6
    for i in range(6, 0, -1):
        f = (F[i][0] + f[0], F[i][1] + f[1], F[i][2] + f[2])
        o_prev = os_[i - 1]
        o_cur = os_[i]
        armj = (o_cur[0] - o_prev[0], o_cur[1] - o_prev[1], o_cur[2] - o_prev[2])
        if i < 6:
            mf = _cross(armj, f_next)
        else:
            mf = (0.0, 0.0, 0.0)
        uc = uC[i]
        mc = _cross((armj[0] + uc[0], armj[1] + uc[1], armj[2] + uc[2]), F[i])
        n = (N[i][0] + n[0] + mf[0] + mc[0],
             N[i][1] + n[1] + mf[1] + mc[1],
             N[i][2] + n[2] + mf[2] + mc[2])
        Rp = Rs[i - 1]
        bias[i - 1] = n[0] * Rp[2] + n[1] * Rp[5] + n[2] * Rp[8]
        f_next = f
    # ---- KE method for M and g; geometric J for the end effector
    Ml = [[0.0] * 6 for _ in range(6)]
    gl = [0.0] * 6
    zs = [(Rs[j][2], Rs[j][5], Rs[j][8]) for j in range(7)]
    pe = os_[6]
    Jl = [[0.0] * 6 for _ in range(6)]
    for j in range(6):
        c = _cross(zs[j], (pe[0] - os_[j][0], pe[1] - os_[j][1], pe[2] - os_[j][2]))
        Jl[0][j] = c[0]; Jl[1][j] = c[1]; Jl[2][j] = c[2]
        Jl[3][j] = zs[j][0]; Jl[4][j] = zs[j][1]; Jl[5][j] = zs[j][2]
    for i in range(1, 7):
        uc = uC[i]
        oc = (os_[i][0] + uc[0], os_[i][1] + uc[1], os_[i][2] + uc[2])
        mi = _M[i - 1]
        ici = _IC[i - 1]
        jv = [None] * 6
        for j in range(i):
            jv[j] = _cross(zs[j], (oc[0] - os_[j][0], oc[1] - os_[j][1], oc[2] - os_[j][2]))
            gl[j] += mi * G0 * jv[j][2]
        for j in range(i):
            a_ = jv[j]
            zj = zs[j]
            rowM = Ml[j]
            for k in range(j, i):
                b_ = jv[k]
                zk = zs[k]
                v = mi * (a_[0] * b_[0] + a_[1] * b_[1] + a_[2] * b_[2]) \
                    + ici * (zj[0] * zk[0] + zj[1] * zk[1] + zj[2] * zk[2])
                rowM[k] += v
                if k != j:
                    Ml[k][j] += v
    M = np.array(Ml)
    g = np.array(gl)
    return M, np.array(bias), g, np.array(pe), np.array(Jl)


# --------------------------------------------------------------- self-check
def _tau_euler_lagrange(q0, amp, om, ph, t0):
    """Numerical Euler-Lagrange torque along q(t)=q0+amp*sin(om t+ph).

    T = 1/2 qd^T M(q) qd  with M from the kinetic-energy method;
    d/dt(M qd) by central time differencing; dT/dq and dV/dq by complex step.
    """
    def qf(t):
        return q0 + amp * np.sin(om * t + ph)

    def qdf(t):
        return amp * om * np.cos(om * t + ph)

    def qddf(t):
        return -amp * om * om * np.sin(om * t + ph)

    h = 1e-6
    dMdt_qd = ((M_kinetic(qf(t0 + h)) @ qdf(t0 + h))
               - (M_kinetic(qf(t0 - h)) @ qdf(t0 - h))) / (2 * h)
    # dT/dq via complex step on M
    qc = qf(t0).astype(complex)
    qd = qdf(t0)
    dTdq = np.zeros(6)
    dVdq = np.zeros(6)
    hc = 1e-30
    for k in range(6):
        dq = np.zeros(6, dtype=complex)
        dq[k] = 1j * hc
        Mk = M_kinetic(qc + dq)
        dTdq[k] = 0.5 * qd @ (Mk.imag / hc) @ qd
        # V = sum m g0 * z-component of COM
        org, rot = fkine_all(qc + dq)
        V = 0.0
        for i in range(1, 7):
            oc = org[i] + rot[i] @ P_COM[i - 1]
            V = V + MASS[i - 1] * G0 * oc[2]
        dVdq[k] = V.imag / hc
    return dMdt_qd - dTdq + dVdq


def run_selfcheck(out_path):
    """RNE vs kinetic-energy / numerical Euler-Lagrange. Returns dict of errors."""
    rng = np.random.default_rng(20261011)
    err_M = err_g = err_EL = 0.0
    for _ in range(10):
        q = rng.uniform(-1.5, 1.5, 6)
        # M: unit-acceleration RNE vs kinetic energy
        dM = np.abs(M_rne(q) - M_kinetic(q)).max()
        # g: rne(q,0,0) vs dV/dq
        dg = np.abs(rne(q, np.zeros(6), np.zeros(6)) - g_potential(q)).max()
        err_M = max(err_M, float(dM))
        err_g = max(err_g, float(dg))
        # full torque along a random smooth trajectory
        amp = rng.uniform(0.2, 0.8, 6)
        om = rng.uniform(0.5, 2.0, 6)
        ph = rng.uniform(0, 2 * np.pi, 6)
        t0 = float(rng.uniform(0.2, 1.0))
        q_t = q + amp * np.sin(om * t0 + ph)
        qd_t = amp * om * np.cos(om * t0 + ph)
        qdd_t = -amp * om * om * np.sin(om * t0 + ph)
        tau_rne = rne(q_t, qd_t, qdd_t)
        tau_el = _tau_euler_lagrange(q, amp, om, ph, t0)
        scale = max(1.0, float(np.abs(tau_rne).max()))
        err_EL = max(err_EL, float(np.abs(tau_rne - tau_el).max()) / scale)
    ok = (err_M < 1e-8) and (err_g < 1e-8) and (err_EL < 1e-8)
    lines = [
        "IIRA 6-DOF arm dynamics self-check (SPEC 2.2)",
        "seed=20261011, 10 random configurations",
        f"max |M_rne - M_kinetic|            : {err_M:.3e}",
        f"max |rne(q,0,0) - dV/dq|           : {err_g:.3e}",
        f"max |rne - EulerLagrange| (rel.)   : {err_EL:.3e}",
        f"tolerance                          : 1e-8",
        f"RESULT                             : {'PASS' if ok else 'FAIL'}",
    ]
    txt = "\n".join(lines) + "\n"
    print(txt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    if not ok:
        raise AssertionError("RNE self-check failed; see " + out_path)
    return dict(err_M=err_M, err_g=err_g, err_EL=err_EL)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    run_selfcheck(os.path.join(root, "results", "selfcheck.txt"))
