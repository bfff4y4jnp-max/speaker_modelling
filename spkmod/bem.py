"""Exterior acoustic BEM (bempp-cl), Burton-Miller formulation.

Internal convention e^{-iwt}, outgoing Green's function e^{ikr}/(4 pi r).
Unknowns: P1 surface pressure. Data: piecewise-constant normal derivative
g = dp/dn = i w rho v_n on the "active" triangles (apertures and vibrating
faces); all other triangles are rigid (g = 0).

Speed features
  * Only active triangles enter the right-hand-side operators.
  * Rigid ground plane z=z0 by the image method: the mesh is mirrored,
    assembled once, then folded back to the original unknowns (half-size LU).
  * Frequency interpolation: matrices are assembled at a few Chebyshev
    wavenumbers per band; the slowly varying part A(k) e^{-ik r_ij} is
    interpolated, so each extra frequency costs O(N^2) instead of a full
    assembly (verified to ~1e-8 relative error).
  * Field points use a vectorised numpy quadrature (no per-column calls).
"""
from __future__ import annotations

import os
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

_QUAD_PTS = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
_QUAD_W = np.array([1 / 3, 1 / 3, 1 / 3])
# 6-point degree-4 rule for field evaluation
_a, _b = 0.445948490915965, 0.091576213509771
_Q6 = np.array([[_a, _a], [1 - 2 * _a, _a], [_a, 1 - 2 * _a], [_b, _b], [1 - 2 * _b, _b], [_b, 1 - 2 * _b]])
_W6 = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def _bempp():
    import bempp_cl.api as bempp
    return bempp


_DEVICE = None


def device():
    """'opencl' when a CPU OpenCL driver is present (2-5x faster, pip install pyopencl
    pocl-binary-distribution), else 'numba'. Override with SPKMOD_BEM_DEVICE."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = os.environ.get("SPKMOD_BEM_DEVICE")
        if not _DEVICE:
            try:
                _DEVICE = "opencl" if _bempp().CPU_OPENCL_DRIVER_FOUND else "numba"
            except Exception:
                _DEVICE = "numba"
    return _DEVICE


def _dense(op):
    a = _bempp().as_matrix(op.weak_form())
    return np.asarray(a.toarray() if hasattr(a, "toarray") else a)


class ExteriorBEM:
    """Exterior Neumann problem on a closed (union of closed) surface(s).

    vertices (n,3), triangles (m,3) with normals pointing into the fluid.
    active: indices of triangles that may carry normal velocity.
    ground: None or z-coordinate of a rigid ground plane.
    """

    def __init__(self, vertices, triangles, active, ground=None, quad_order=3, tol=1e-7):
        self.v0 = np.asarray(vertices, float)
        self.t0 = np.asarray(triangles, np.int64)
        self.active0 = np.asarray(active, np.int64)
        self.ground = ground
        self.quad_order = quad_order
        v, t, node_fold, act_u, act_fold = self._union(ground, tol)
        self.vertices, self.triangles = v, t
        self.N = len(v)
        # folding operators: union -> solved (used original) unknowns
        self.F = sp.csr_matrix((np.ones(self.N), (np.arange(self.N), node_fold)), shape=(self.N, self.n))
        self.act_u = act_u
        self.Fe = sp.csr_matrix((np.ones(len(act_u)), (np.arange(len(act_u)), act_fold)),
                                shape=(len(act_u), len(self.active0)))
        bempp = _bempp()
        bempp.GLOBAL_PARAMETERS.quadrature.regular = quad_order
        bempp.GLOBAL_PARAMETERS.quadrature.singular = quad_order
        dom = np.zeros(len(t), np.uint32)
        dom[act_u] = 1
        self.grid = bempp.Grid(v.T.copy(), t.T.astype(np.uint32).copy(), domain_indices=dom)
        self.P1 = bempp.function_space(self.grid, "P", 1)
        self.D0 = bempp.function_space(self.grid, "DP", 0, segments=[1])
        # bempp may reorder DOFs; build maps from our numbering to theirs
        self._p1_perm = self._p1_map()
        self._d0_perm = self._d0_map()
        tri = v[t]
        self.centroids = tri.mean(1)
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.areas = 0.5 * np.linalg.norm(cr, axis=1)
        self.normals = cr / (2 * self.areas[:, None])
        self._R11 = np.linalg.norm(v[: self.n, None] - v[None], axis=-1)       # rows: orig nodes
        self._R10 = np.linalg.norm(v[: self.n, None] - self.centroids[act_u][None], axis=-1)
        self._M11 = None

    # ------------------------------------------------------------------ setup
    def _union(self, z0, tol):
        """Union mesh (original + mirror image) and the folding maps.

        Solved unknowns are the original vertices that remain in use
        (self.used); results are expanded with to_orig().
        """
        v, t = self.v0, self.t0
        keep = np.arange(len(t))
        if z0 is not None:
            if v[:, 2].min() < z0 - 1e-6:
                raise ValueError("geometry extends below the ground plane")
            on_tri = (np.abs(v[t][:, :, 2] - z0) < tol).all(1)
            if on_tri[self.active0].any():
                raise ValueError("active (vibrating/aperture) faces lie on the ground plane")
            keep = np.where(~on_tri)[0]
        self.used = np.unique(t[keep])
        self.n = len(self.used)
        red = np.full(len(v), -1)
        red[self.used] = np.arange(self.n)
        vr, tk = v[self.used], red[t[keep]]
        pos = np.full(len(t), -1)
        pos[keep] = np.arange(len(keep))
        a_keep = pos[self.active0]
        if z0 is None:
            return vr, tk, np.arange(self.n), a_keep, np.arange(len(a_keep))
        on = np.abs(vr[:, 2] - z0) < tol
        off = np.where(~on)[0]
        mir_idx = np.arange(self.n)
        mir_idx[off] = self.n + np.arange(len(off))
        vm = vr[off].copy()
        vm[:, 2] = 2 * z0 - vm[:, 2]
        V = np.vstack([vr, vm])
        T = np.vstack([tk, mir_idx[tk][:, ::-1]])         # reflection flips orientation
        node_fold = np.concatenate([np.arange(self.n), off])
        act_u = np.concatenate([a_keep, a_keep + len(keep)])
        act_fold = np.concatenate([np.arange(len(a_keep))] * 2)
        return V, T, node_fold, act_u, act_fold

    def to_orig(self, P):
        """Expand solved-node rows to the caller's vertex numbering (unused rows = 0)."""
        out = np.zeros((len(self.v0),) + P.shape[1:], P.dtype)
        out[self.used] = P
        return out

    def _p1_map(self):
        # P1 dof for each vertex
        perm = np.zeros(self.N, np.int64)
        l2g = self.P1.local2global
        for e, tri in enumerate(self.triangles):
            perm[tri] = l2g[e]
        return perm

    def _d0_map(self):
        l2g = self.D0.local2global
        return np.array([l2g[e][0] for e in self.act_u])

    # -------------------------------------------------------------- assembly
    def assemble(self, k):
        """Folded system matrices at wavenumber k (scaled by k for smoothness).

        Returns A (n x n) and B (n x n_active) such that  A p = B g.
        """
        Au, Bu = self._assemble_union(k)
        return self._fold(Au, Bu)

    def _assemble_union(self, k):
        from bempp_cl.api.operators.boundary import helmholtz as H, sparse
        P1, D0 = self.P1, self.D0
        if self._M11 is None:
            self._M11 = _dense(sparse.identity(P1, P1, P1))
            self._M10 = _dense(sparse.identity(D0, P1, P1))
        d = device()
        K = _dense(H.double_layer(P1, P1, P1, k, device_interface=d))
        W = _dense(H.hypersingular(P1, P1, P1, k, device_interface=d))
        V = _dense(H.single_layer(D0, P1, P1, k, device_interface=d))
        Kp = _dense(H.adjoint_double_layer(D0, P1, P1, k, device_interface=d))
        # k * (1/2 M - K + (i/k) W)  and  k * (-V - (i/k)(1/2 M + K'))
        A = k * (0.5 * self._M11 - K) + 1j * W
        B = -k * V - 1j * (0.5 * self._M10 + Kp)
        pr, pd = self._p1_perm, self._d0_perm
        rows = pr[: self.n]
        return A[np.ix_(rows, pr)], B[np.ix_(rows, pd)]

    def _fold(self, Au, Bu):
        return np.asarray(Au @ self.F), np.asarray(Bu @ self.Fe)

    def band(self, kmin, kmax, n_nodes=None, ks_needed=None):
        """Precompute interpolation data for wavenumbers in [kmin, kmax].

        ks_needed: the wavenumbers that will be requested; when there are no more of them than
        interpolation nodes they are used as the nodes (exact, fewer assemblies).
        """
        D = float(self._R11.max()) if self.n else 0.0
        if n_nodes is None:
            n_nodes = int(np.clip(np.ceil((kmax - kmin) * D / 4) + 3, 3, 9))
        if ks_needed is not None and 0 < len(ks_needed) <= n_nodes:
            ks = np.unique(np.asarray(ks_needed, float))
            data = []
            for k in ks:
                Au, Bu = self._assemble_union(k)
                data.append((Au * np.exp(-1j * k * self._R11), Bu * np.exp(-1j * k * self._R10)))
            return _Band(self, ks, data)
        if kmax - kmin < 1e-9 or n_nodes <= 1:
            n_nodes = 1
            ks = np.array([0.5 * (kmin + kmax)])
        else:
            j = np.arange(n_nodes)
            ks = 0.5 * (kmin + kmax) + 0.5 * (kmax - kmin) * np.cos(np.pi * (j + 0.5) / n_nodes)
        data = []
        for k in ks:
            Au, Bu = self._assemble_union(k)
            data.append((Au * np.exp(-1j * k * self._R11), Bu * np.exp(-1j * k * self._R10)))
        return _Band(self, ks, data)

    # -------------------------------------------------------------- field
    def field_matrices(self, points, k, far=False):
        """Return (Dm, Sm) so that p(x) = Dm @ p_surf - Sm @ g (folded unknowns).

        far=True: points are unit directions; result is the far-field pattern
        p ~ e^{ikr}/r * (Dm p - Sm g).
        """
        pts = np.asarray(points, float)
        v, t = self.vertices, self.triangles
        tri = v[t]
        qp = (tri[:, 0, None] * (1 - _Q6[:, 0] - _Q6[:, 1])[None, :, None]
              + tri[:, 1, None] * _Q6[None, :, 0, None] + tri[:, 2, None] * _Q6[None, :, 1, None])  # (m,q,3)
        wq = self.areas[:, None] * _W6[None]                                                           # (m,q)
        phi = np.stack([1 - _Q6[:, 0] - _Q6[:, 1], _Q6[:, 0], _Q6[:, 1]], 1)                         # (q,3)
        nrm = self.normals
        m = len(t)
        Dm = np.zeros((len(pts), self.N), complex)
        Sm = np.zeros((len(pts), len(self.act_u)), complex)
        is_act = np.full(m, -1)
        is_act[self.act_u] = np.arange(len(self.act_u))
        cols = t.reshape(-1)
        Pm = sp.csr_matrix((np.ones(3 * m), (np.arange(3 * m), cols)), shape=(3 * m, self.N))
        chunk = max(1, int(3e6 // (m * 6)))
        for s in range(0, len(pts), chunk):
            x = pts[s:s + chunk]
            if far:
                xd = np.einsum("pd,mqd->pmq", x, qp)
                G = np.exp(-1j * k * xd) / (4 * np.pi)
                dG = -1j * k * np.einsum("pd,md->pm", x, nrm)[:, :, None] * G
            else:
                d = x[:, None, None, :] - qp[None]                   # (p,m,q,3)
                r = np.linalg.norm(d, axis=-1)
                G = np.exp(1j * k * r) / (4 * np.pi * r)
                # dG/dn_y = (x-y).n /r * (1/r - ik) G
                cos = np.einsum("pmqd,md->pmq", d, nrm) / r
                dG = cos * (1 / r - 1j * k) * G
            Gw, dGw = G * wq[None], dG * wq[None]
            Sm[s:s + chunk] = Gw[:, self.act_u].sum(-1)
            Dl = np.einsum("pmq,qa->pma", dGw, phi)                   # (p,m,3)
            Dm[s:s + chunk] = (Pm.T @ Dl.reshape(len(x), -1).T).T
        Dm = np.asarray(Dm @ self.F)
        Sm = np.asarray(Sm @ self.Fe)
        return Dm, Sm


class _Band:
    """Frequency-interpolated folded BEM matrices over one band."""

    def __init__(self, bem: ExteriorBEM, ks, data):
        self.bem, self.ks, self.data = bem, np.asarray(ks), data

    def matrices(self, k):
        ks = self.ks
        if len(ks) == 1:
            L = [1.0]
        else:
            L = [np.prod([(k - ks[j]) / (ks[i] - ks[j]) for j in range(len(ks)) if j != i]) for i in range(len(ks))]
        b = self.bem
        Au = sum(l * d[0] for l, d in zip(L, self.data)) * np.exp(1j * k * b._R11)
        Bu = sum(l * d[1] for l, d in zip(L, self.data)) * np.exp(1j * k * b._R10)
        return b._fold(Au, Bu)

    def solve(self, k):
        """Surface pressure per unit g on each active triangle: (n x n_active)."""
        A, B = self.matrices(k)
        return sla.lu_solve(sla.lu_factor(A, overwrite_a=True, check_finite=False), B, check_finite=False)


def memory_estimate(n_orig, n_union, n_active_union, n_nodes):
    """Approximate peak bytes for one band."""
    c = 16
    assembly = c * (4 * n_union ** 2 + 2 * n_union * n_active_union)
    stored = c * n_nodes * n_orig * (n_union + n_active_union)
    return assembly + stored


def available_memory():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.5)
