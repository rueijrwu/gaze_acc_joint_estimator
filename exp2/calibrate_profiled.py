"""Joint bounded variable-projection calibration, gated captures 1–4 only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", os.environ["OMP_NUM_THREADS"].split(",")[0])
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse.linalg import LinearOperator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from observations import measurements
import m2_model

DEMANDS = np.array([1000/2775, 4., 3., 2.])
KNOTS = np.sort(DEMANDS)  # Protocol-defined, never inferred from held-out observations.
TARGETS = np.array([-15., -7.5, 0., 7.5, 15.])


SETTINGS = dict(theta_bounds_deg=[-20,20],A_bounds_diopters=[0,6],
                theta_anchor_strength=1.,A_anchor_strength=1.,smooth_theta_strength=.1,smooth_A_strength=.1,
                theta_anchor_scale_deg=1.,A_anchor_scale_diopters=.25,
                smooth_theta_scale_deg=1.,smooth_A_scale_diopters=.25,model_prior_strength=.1,
                equal_fixation_optical_weight=True,objective_factor=.5,solver_residual_scale='sqrt(frame_count)',
                curvature_prior=0.,invalid_gap_links='broken',extrapolation='endpoint-linear displacement')


def interpolation(A, knots):
    """Continuous linear interpolation/extrapolation; right derivative at knots."""
    A = np.atleast_1d(A)
    k = np.clip(np.searchsorted(knots, A, side="right")-1, 0, len(knots)-2)
    width = knots[k+1]-knots[k]
    alpha = (A-knots[k])/width
    weights, derivative = np.zeros((len(A), len(knots))), np.zeros((len(A), len(knots)))
    rows = np.arange(len(A))
    weights[rows, k], weights[rows, k+1] = 1-alpha, alpha
    derivative[rows, k], derivative[rows, k+1] = -1/width, 1/width
    return weights, derivative


def design(theta, a):
    t = theta/15
    basis = np.column_stack([np.ones(len(t)), t, t*t])
    return np.column_stack([basis, basis*a[:, None]])


def normal_solve(R,v):
    # Solve R.T R without constructing/squaring the condition number in a
    # normal-equation factorization. Only small coefficient-space solves.
    return solve_triangular(R,solve_triangular(R.T,v,lower=True))

def basis(theta, a, p, derivatives=True, curvature=False):
    A = a**(1/p)
    w, dw = interpolation(A, KNOTS)
    n = len(theta)
    H = np.zeros((n, 2, 15 if curvature else 14))
    H[:, 0, :4], H[:, 0, 4:8] = w, w*theta[:, None]
    H[:, 1, 8:14] = design(theta, a)
    if curvature: H[:, 0, 14] = (theta/15)**2
    if not derivatives:
        return H, None, None
    dt, da = np.zeros_like(H), np.zeros_like(H)
    dt[:, 0, 4:8] = w
    t = theta/15
    dt[:, 1, 8:14] = np.column_stack([np.zeros(n), np.ones(n)/15, 2*t/15,
                                   np.zeros(n), a/15, 2*a*t/15])
    dA = (1/p)*a**(1/p-1)
    da[:, 0, :4], da[:, 0, 4:8] = dw*dA[:, None], dw*(dA*theta)[:, None]
    da[:, 1, 11:14] = np.column_stack([np.ones(n), t, t*t])
    if curvature: dt[:, 0, 14] = 2*theta/225
    return H, dt, da


def load_data(experiment, interval_path, heldout=None, target_overrides=None):
    raw = interval_path.read_bytes()
    report = json.loads(raw)
    rows = report["fixations"]
    if len(rows) != 20 or report.get("capture5_used") is not False or not report.get("require_valid_pupil"):
        raise ValueError("Select the twenty pupil-gated capture1–4 intervals explicitly")
    expected = {f"capture_{i}_detections.pkl" for i in range(1, 5)}
    if set(r["capture"] for r in rows) != expected:
        raise ValueError("Only captures1–4 are authorized")
    hashes = {r["capture"]: r["sha256"] for r in report["sources"]}

    # Load and validate target overrides if provided
    override_targets = None
    override_sha256 = None
    override_captures = None
    if target_overrides is not None:
        target_overrides = Path(target_overrides)
        override_raw = target_overrides.read_bytes()
        override_sha256 = hashlib.sha256(override_raw).hexdigest()
        override_spec = json.loads(override_raw)
        if override_spec.get("schema_version") != 1:
            raise ValueError("Target override schema_version must be 1")
        if override_spec.get("units") != "deg":
            raise ValueError("Target override units must be 'deg'")
        if hashlib.sha256(raw).hexdigest() != override_spec.get("base_intervals_sha256"):
            raise ValueError("Target override base_intervals_sha256 mismatch")
        override_targets = override_spec.get("targets", {})
        if not isinstance(override_targets, dict) or not override_targets:
            raise ValueError("Target override targets must be a nonempty dict")
        override_captures = set(override_targets.keys())
        for capture_name in override_captures:
            if capture_name not in expected:
                raise ValueError(f"Unknown capture name in overrides: {capture_name}")
            targets = override_targets[capture_name]
            if not isinstance(targets, list) or len(targets) != 5:
                raise ValueError(f"Override targets for {capture_name} must be a list of exactly 5 elements")
            for t in targets:
                if not isinstance(t, (int, float)) or not np.isfinite(t):
                    raise ValueError(f"Override targets must be finite floats")
                if abs(t) > 20:
                    raise ValueError(f"Override target {t} exceeds bounds [-20, 20]")
            if not all(targets[i] < targets[i+1] for i in range(4)):
                raise ValueError(f"Override targets for {capture_name} must be strictly increasing")

    records, meta, ylist, flist, glist = [], [], [], [], []
    for capture in range(1, 5):
        name = f"capture_{capture}_detections.pkl"
        payload = (experiment/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != hashes[name]:
            raise ValueError("Selected interval source hash mismatch")
        arrays = pickle.loads(payload)["arrays"]
        obs, good = measurements(arrays, "absolute_x")
        obs[:, 0] *= -1
        pupil = np.asarray(arrays["pupil_valid"])
        if pupil.dtype != np.bool_ or pupil.shape != good.shape:
            raise ValueError("Stored pupil validity must be a matching boolean array")
        good &= pupil
        frame = np.asarray(arrays["frame_index"])
        if np.any(np.diff(frame) != 1):
            raise ValueError("Central-frame cuts require consecutive original frame indices")
        selected = [r for r in rows if r["capture"] == name]
        if [r["target_theta_deg"] for r in selected] != TARGETS.tolist():
            raise ValueError("Ordered target labels mismatch")
        full, core, lookup = np.zeros(len(good), bool), np.zeros(len(good), bool), np.full(len(good), -1)
        for r in selected:
            j = len(meta)
            start, end = r["start_row"], r["end_row_exclusive"]
            if not 0 <= start < end <= len(good) or r["full_valid_count"] != int(good[start:end].sum()):
                raise ValueError("Interval bounds/count mismatch under stored pupil gate")
            if r["first_frame"] != frame[start] or r["last_frame_inclusive"] != frame[end-1]:
                raise ValueError("Interval frame provenance mismatch")
            if r["demand_diopters_label"] != DEMANDS[capture-1]:
                raise ValueError("Demand label mismatch")
            cut = int(np.floor(.1*(end-start)))
            c0, c1 = start+cut, end-cut
            full[start:end], core[c0:c1] = True, True
            index = np.arange(c0, c1)[good[c0:c1]]
            if not len(index):
                raise ValueError("Empty valid core")
            lookup[index] = j
            meta.append(dict(r, core_start_row=c0, core_end_row_exclusive=c1,
                             core_valid_count=len(index), core_invalid_count=c1-c0-len(index),
                             excluded_event_count=0, heldout=j == heldout))
            ylist.append(obs[index]); flist.append(frame[index]); glist.append(np.full(len(index), j))
        records.append(dict(capture=capture, arrays=arrays, y=obs, valid=good, full=full, core=core, groups=lookup))

    # Apply overrides to meta if provided
    if override_targets is not None:
        selected = [dict(r) for r in meta]
        for j, r in enumerate(selected):
            if r["capture"] in override_captures:
                targets_list = override_targets[r["capture"]]
                fixation_index = r.get("fixation_index_in_capture", j % 5)  # fallback to derived index
                # Find the target index based on the original target value
                try:
                    orig_idx = TARGETS.tolist().index(r["target_theta_deg"])
                except (ValueError, IndexError):
                    raise ValueError(f"Cannot map original target {r['target_theta_deg']} for override")
                r["nominal_target_theta_deg_frozen"] = r["target_theta_deg"]
                r["target_theta_deg"] = float(targets_list[orig_idx])
        meta = selected

    provenance = {"selected_interval_path":str(interval_path), "selected_interval_sha256":hashlib.sha256(raw).hexdigest(),
                  "source_sha256":hashes, "validity_policy":"measurement_valid AND stored pupil_valid",
                  "native_detection_rerun":False, "interval_version_frozen":True, "edge_fraction":.1,
                  "event_mask":"empty; S_j=V_j", "capture5_used":False}
    if override_targets is not None:
        provenance.update(target_override_path=str(target_overrides), target_override_sha256=override_sha256,
                         target_override_captures=list(override_captures), target_override_targets=override_targets)
    return np.vstack(ylist), np.concatenate(flist), np.concatenate(glist), meta, records, provenance


def initial_model(y, groups, meta, selected, fixed_p=None):
    """Training-only ordinary means; never use held-out observations or labels."""
    means = np.array([y[groups == j].mean(axis=0) for j in selected])
    targets = np.array([meta[j]["target_theta_deg"] for j in selected])
    demands = np.array([meta[j]["demand_diopters_label"] for j in selected])
    def exponent_loss(p):
        X = design(targets, demands**p)
        c = np.linalg.lstsq(X, means[:, 1], rcond=None)[0]
        return np.sum((X@c-means[:, 1])**2)
    p = float(fixed_p) if fixed_p is not None else float(minimize_scalar(exponent_loss, bounds=(.15, 1.), method="bounded").x)
    if not 0 < p <= 1:
        raise ValueError("Transformed-a zero bound requires 0<p<=1")
    b, s = np.zeros(4), np.zeros(4)
    theta = np.zeros(len(y)); A = np.zeros(len(y))
    for capture in range(1, 5):
        slots = np.array([k for k,j in enumerate(selected) if meta[j]["capture"] == f"capture_{capture}_detections.pkl"])
        if len(slots) < 2:
            raise ValueError("At least two training fixations needed per capture")
        line = np.linalg.lstsq(np.column_stack([np.ones(len(slots)), targets[slots]]), means[slots, 0], rcond=None)[0]
        if abs(line[1]) < 1e-8:
            raise ValueError("Insufficient displacement gain")
        knot = np.flatnonzero(np.isclose(KNOTS, DEMANDS[capture-1]))[0]
        b[knot], s[knot] = line
        mask = np.isin(groups, [selected[k] for k in slots])
        theta[mask] = np.clip((y[mask, 0]-line[0])/line[1], -20, 20)
        A[mask] = DEMANDS[capture-1]
    rho = np.linalg.lstsq(design(targets, demands**p), means[:, 1], rcond=None)[0]
    coef = np.r_[b,s,rho]
    return p, coef, theta, A


def noise_covariance(y, frames, groups):
    """Robust consecutive differences, training-only, frozen SPD estimate.

    MAD standardized radial winsorization preserves correlation, but motion can
    contaminate differences. This empirical covariance is not reference truth.
    """
    mask = (groups[1:] == groups[:-1]) & (np.diff(frames) == 1)
    delta = np.diff(y, axis=0)[mask]/np.sqrt(2)
    if len(delta) < 10:
        raise ValueError("Too few consecutive differences for noise covariance")
    center = np.median(delta, axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(delta-center), axis=0), 1e-7)
    radial = np.linalg.norm((delta-center)/scale, axis=1)
    winsor = (delta-center)*np.minimum(1, 3/np.maximum(radial, 1e-12))[:, None]
    cov = winsor.T@winsor/len(winsor)
    cov += np.diag(np.maximum(np.diag(cov)*1e-4, 1e-14))
    np.linalg.cholesky(cov)
    return cov


class ProfiledProblem:
    def __init__(self, y, frames, groups, targets, demands, p, initial_coef, W, prior_W=None,
                 coefficient_map=None, curvature=False, curvature_strength=0., curvature_scale_output=1.,
                 previous_means=None, previous_mean_strength=0., previous_mean_scale_deg=1.,
                 previous_mean_provenance=None, model='piecewise', theta_anchor_scale_deg=1.):
        if model not in ('piecewise', 'm2'):
            raise ValueError('Unknown model')
        self.model = model
        if not np.isfinite(theta_anchor_scale_deg) or theta_anchor_scale_deg <= 0:
            raise ValueError('Invalid theta anchor scale')
        self.theta_anchor_scale = float(theta_anchor_scale_deg)
        if model == 'm2' and (curvature or coefficient_map is not None or p is not None):
            raise ValueError('m2 has no exponent, curvature term or coefficient map')
        self.y, self.frames, self.groups = np.asarray(y), np.asarray(frames), np.asarray(groups)
        self.targets, self.demands = np.asarray(targets), np.asarray(demands)
        self.p, self.initial_coef = p, initial_coef.copy()
        self.coefficient_map = None
        self.curvature = bool(curvature)
        self.coefficient_count = m2_model.N_COEF if model == 'm2' else 15 if self.curvature else 14
        if np.shape(initial_coef) != (self.coefficient_count,) or not np.isfinite(initial_coef).all():
            raise ValueError('Initial coefficients do not match model schema')
        if (not np.isfinite(curvature_strength) or curvature_strength < 0
                or not np.isfinite(curvature_scale_output) or curvature_scale_output <= 0
                or (not self.curvature and curvature_strength != 0)):
            raise ValueError('Invalid curvature penalty')
        self.curvature_strength = float(curvature_strength)
        self.curvature_scale_output = float(curvature_scale_output)
        self.free_coefficient_count = self.coefficient_count
        if coefficient_map is not None:
            E = np.asarray(coefficient_map, dtype=float)
            if (E.ndim != 2 or E.shape[0] != self.coefficient_count or not 1 <= E.shape[1] <= self.coefficient_count
                    or not np.isfinite(E).all() or np.linalg.matrix_rank(E) != E.shape[1]):
                raise ValueError('Coefficient map must match schema with full column rank')
            gamma = np.linalg.lstsq(E, initial_coef, rcond=None)[0]
            if not np.allclose(E@gamma, initial_coef, atol=1e-12, rtol=1e-12):
                raise ValueError('Initial coefficient prior violates the coefficient map')
            self.coefficient_map = E.copy()
            self.coefficient_map.setflags(write=False)
            self.free_coefficient_count = E.shape[1]
        self.n, self.J = len(y), len(targets)
        if (not np.isfinite(previous_mean_strength) or previous_mean_strength < 0
                or not np.isfinite(previous_mean_scale_deg) or previous_mean_scale_deg <= 0):
            raise ValueError('Invalid previous-mean prior strength/scale')
        self.previous_mean_strength = float(previous_mean_strength)
        self.previous_mean_scale_deg = float(previous_mean_scale_deg)
        self.previous_mean_provenance = previous_mean_provenance
        self.previous_means = None if previous_means is None else np.asarray(previous_means, dtype=float).copy()
        if self.previous_means is not None:
            if self.previous_means.shape != (self.J,) or not np.isfinite(self.previous_means).all():
                raise ValueError('Previous gaze means must match all training fixations')
            self.previous_means.setflags(write=False)
        if self.previous_mean_strength > 0 and self.previous_means is None:
            raise ValueError('Positive previous-mean strength requires frozen training means')
        self.counts = np.bincount(groups, minlength=self.J)
        if np.any(self.counts == 0): raise ValueError("Empty training group")
        self.alpha = 1/(self.J*self.counts[groups])
        self.W = np.asarray(W)
        if self.W.shape!=(2,2) or not np.allclose(self.W,self.W.T): raise ValueError('Precision must be symmetric2x2 SPD')
        self.L = np.linalg.cholesky(self.W).T
        self.prior_W = self.W.copy() if prior_W is None else np.asarray(prior_W)
        self.prior_L = np.linalg.cholesky(self.prior_W).T
        self.left = np.flatnonzero((groups[1:] == groups[:-1]) & (np.diff(frames) == 1))
        self.right = self.left+1
        links = np.bincount(groups[self.left], minlength=self.J)
        self.link_scale = np.sqrt(.1/(self.J*np.maximum(links[groups[self.left]], 1)))
        th, aa = np.meshgrid(np.linspace(-15,15,9), np.linspace(KNOTS[0], KNOTS[-1],9))
        H = (m2_model.basis_m2(th.ravel(), aa.ravel(), derivatives=False)[0] if model == 'm2' else
             basis(th.ravel(), aa.ravel()**p, p, curvature=self.curvature)[0])
        self.prior_M = np.einsum('ab,nbk->nak',self.prior_L,H).reshape(-1,self.coefficient_count)*np.sqrt(.1/len(H))
        self.prior_v = self.prior_M@initial_coef
        self.function_prior_rows = len(self.prior_v)
        if self.curvature and self.curvature_strength > 0:
            row = np.zeros((1, self.coefficient_count))
            row[0, 14] = np.sqrt(self.curvature_strength)/self.curvature_scale_output
            self.prior_M = np.vstack([self.prior_M, row])
            self.prior_v = np.r_[self.prior_v, 0.]
        self.a_scale = m2_model.A_SCALE if model == 'm2' else 4**p
        self.cache = None
        self._geometry = None

    @property
    def zero_A_singular(self):
        """True only for the piecewise power transform with p<1 (singular ratio derivative at A=0)."""
        return self.p is not None and self.p < 1

    def _basis(self, theta, a, derivatives=True):
        # For m2 `a` is A itself (identity state transform); for piecewise it is A**p.
        if self.model == 'm2':
            return m2_model.basis_m2(theta, a, derivatives)
        return basis(theta, a, self.p, derivatives=derivatives, curvature=self.curvature)

    def encode(self, theta, A):
        if self.model == 'm2':
            return np.column_stack([theta/15, A/self.a_scale]).ravel()
        return np.column_stack([theta/15, A**self.p/self.a_scale]).ravel()

    def decode(self, x):
        q = x.reshape(-1,2)
        th,a = 15*q[:,0],self.a_scale*q[:,1]
        if self.model == 'm2':
            return th,a,a,np.full(len(th),self.a_scale)
        A = a**(1/self.p)
        dA = self.a_scale/self.p*a**(1/self.p-1)
        return th,a,A,dA

    def bounds(self):
        if self.model == 'm2':
            return np.tile([-20/15,0.],self.n),np.tile([20/15,6/self.a_scale],self.n)
        return np.tile([-20/15,0.],self.n),np.tile([20/15,6**self.p/self.a_scale],self.n)

    def penalties(self, x):
        th,a,A,dA = self.decode(x)
        mt = np.bincount(self.groups,weights=th)/self.counts
        ma = np.bincount(self.groups,weights=A)/self.counts
        anchor = np.column_stack([(mt-self.targets)/(np.sqrt(self.J)*self.theta_anchor_scale),
                                  (ma-self.demands)/(.25*np.sqrt(self.J))]).ravel()
        temporal = np.column_stack([(th[self.right]-th[self.left])*self.link_scale,
                                     (A[self.right]-A[self.left])*self.link_scale/.25]).ravel()
        legacy = np.r_[anchor,temporal]
        if self.previous_mean_strength == 0: return legacy
        previous = (mt-self.previous_means)*np.sqrt(self.previous_mean_strength/self.J)/self.previous_mean_scale_deg
        return np.r_[legacy, previous]

    def penalty_jv(self, v, dA):
        q=v.reshape(-1,2); dt=15*q[:,0]; dA=dA*q[:,1]
        anchor=np.column_stack([np.bincount(self.groups,weights=dt)/self.counts/(np.sqrt(self.J)*self.theta_anchor_scale),
                               np.bincount(self.groups,weights=dA)/self.counts/(.25*np.sqrt(self.J))]).ravel()
        temporal=np.column_stack([(dt[self.right]-dt[self.left])*self.link_scale,
                                  (dA[self.right]-dA[self.left])*self.link_scale/.25]).ravel()
        legacy = np.r_[anchor,temporal]
        if self.previous_mean_strength == 0: return legacy
        previous = np.bincount(self.groups,weights=dt)/self.counts*np.sqrt(self.previous_mean_strength/self.J)/self.previous_mean_scale_deg
        return np.r_[legacy,previous]

    def penalty_jtv(self,w,dA):
        anchors=w[:2*self.J].reshape(-1,2); end=2*self.J+2*len(self.left)
        temporal=w[2*self.J:end].reshape(-1,2)
        dt=anchors[self.groups,0]/(self.counts[self.groups]*np.sqrt(self.J)*self.theta_anchor_scale)
        dAa=anchors[self.groups,1]/(self.counts[self.groups]*.25*np.sqrt(self.J))
        if self.previous_mean_strength > 0:
            dt += w[end:][self.groups]*np.sqrt(self.previous_mean_strength/self.J)/(self.previous_mean_scale_deg*self.counts[self.groups])
        vt=temporal[:,0]*self.link_scale; va=temporal[:,1]*self.link_scale/.25
        np.add.at(dt,self.left,-vt); np.add.at(dt,self.right,vt)
        np.add.at(dAa,self.left,-va); np.add.at(dAa,self.right,va)
        return np.column_stack([15*dt,dAa*dA]).ravel()

    def forward_matrix(self, x):
        """Avoid derivative allocations when only predicted observations are needed."""
        if self._geometry is not None and np.array_equal(x, self._geometry['x']):
            return self._geometry['H']
        theta, a, _, _ = self.decode(x)
        return self._basis(theta, a, derivatives=False)[0]

    def profile(self,x,omega):
        if self.cache is not None and np.array_equal(x,self.cache['x']) and np.array_equal(omega,self.cache['omega']):
            return self.cache
        if self._geometry is None or not np.array_equal(x, self._geometry['x']):
            th,a,A,dA = self.decode(x)
            H,dt,da=self._basis(th,a)
            self._geometry = dict(x=x.copy(), H=H, dt=dt, da=da, dA=dA)
        geometry = self._geometry
        H,dt,da,dA = geometry['H'],geometry['dt'],geometry['da'],geometry['dA']
        scale=np.sqrt(self.alpha*omega)
        def whiten(B): return np.einsum('ab,nbk->nak',self.L,B)*scale[:,None,None]
        Mopt=whiten(H).reshape(-1,self.coefficient_count)
        M=np.vstack([Mopt,self.prior_M])
        if self.coefficient_map is not None:
            M = M@self.coefficient_map
        v=np.r_[(self.y@self.L.T*scale[:,None]).ravel(),self.prior_v]
        # QR rather than normal-equation coefficient fitting; small triangular
        # system also supplies the low-rank projection without an N-by-N matrix.
        Q,R=np.linalg.qr(M,mode='reduced')
        singular=np.linalg.svd(R,compute_uv=False)
        if singular[-1] <= singular[0]*1e-12: raise ValueError("Rank deficient coefficient profile")
        gamma=np.linalg.solve(R,Q.T@v)
        e=M@gamma-v
        Dt,Da=whiten(dt)*15,whiten(da)*self.a_scale
        if self.coefficient_map is not None:
            Dt,Da = Dt@self.coefficient_map,Da@self.coefficient_map
        local=np.stack([Dt@gamma,Da@gamma],axis=2)
        coef = gamma if self.coefficient_map is None else self.coefficient_map@gamma
        eo=e[:2*self.n].reshape(-1,2)
        C=np.stack([np.einsum('nbk,nb->nk',Dt,eo),np.einsum('nbk,nb->nk',Da,eo)],axis=1)
        factor=R
        data=dict(x=x.copy(),omega=omega.copy(),coef=coef,M=M,e=e,local=local,C=C,dA=dA,
                  factor=factor,condition=float(singular[0]/singular[-1]))
        self.cache=data
        return data

    def residual(self,x,omega):
        d=self.profile(x,omega)
        return np.r_[d['e'],self.penalties(x)]*np.sqrt(self.n)

    def jacobian(self,x,omega):
        d=self.profile(x,omega); M,e,B,C=d['M'],d['e'],d['local'],d['C']
        m=len(e); sqrtN=np.sqrt(self.n)
        def mv(v):
            v=np.asarray(v).ravel(); vv=v.reshape(-1,2)
            q=np.r_[np.einsum('nbs,ns->nb',B,vv).ravel(),np.zeros(len(self.prior_v))]
            correction=normal_solve(d['factor'],M.T@q+np.einsum('nsk,ns->k',C,vv))
            return np.r_[q-M@correction,self.penalty_jv(v,d['dA'])]*sqrtN
        def rmv(w):
            w=np.asarray(w).ravel(); z=w[:m]
            small=normal_solve(d['factor'],M.T@z)
            projected=(z-M@small)[:2*self.n].reshape(-1,2)
            grad=np.einsum('nbs,nb->ns',B,projected)-np.einsum('nsk,k->ns',C,small)
            return (grad.ravel()+self.penalty_jtv(w[m:],d['dA']))*sqrtN
        return LinearOperator((m+len(self.penalties(x)),2*self.n),matvec=mv,rmatvec=rmv,dtype=float)

    def true_objective(self,x,coef,kappa=None):
        residual=(self.forward_matrix(x)@coef-self.y)@self.L.T
        s=np.sum(residual**2,axis=1)
        loss=s if kappa is None else 2*kappa*kappa*(np.sqrt(1+s/kappa**2)-1)
        obs=.5*np.sum(self.alpha*loss)
        prior=.5*np.sum((self.prior_M@coef-self.prior_v)**2)
        pen=self.penalties(x)
        end = 2*self.J+2*len(self.left)
        parts = dict(observations=float(obs),model_prior=float(prior),
                     anchors=float(.5*np.dot(pen[:2*self.J],pen[:2*self.J])),
                     temporal=float(.5*np.dot(pen[2*self.J:end],pen[2*self.J:end])))
        if self.curvature:
            qprior = .5*np.sum((self.prior_M@coef-self.prior_v)[self.function_prior_rows:]**2)
            parts['curvature_prior'] = float(qprior)
            parts['model_prior'] = float(prior-qprior)
        if self.previous_mean_strength > 0:
            parts['previous_mean_prior'] = float(.5*np.dot(pen[end:],pen[end:]))
        return obs+prior+.5*np.dot(pen,pen),parts

    def robust_coefficients(self,x,kappa=None,maxiter=100,tol=1e-8):
        omega=np.ones(self.n)
        for i in range(maxiter):
            d=self.profile(x,omega); coef=d['coef']
            r=(self.forward_matrix(x)@coef-self.y)@self.L.T
            updated=np.ones(self.n) if kappa is None else 1/np.sqrt(1+np.sum(r*r,axis=1)/kappa**2)
            change=float(np.max(np.abs(updated-omega)))
            value=self.true_objective(x,coef,kappa)[0]
            if change < tol:
                coef=self.profile(x,updated)['coef'].copy()
                value=self.true_objective(x,coef,kappa)[0]
                return coef,updated,dict(converged=True,iterations=i+1,weight_change=change,objective=value)
            omega=updated
        return coef,omega,dict(converged=False,iterations=maxiter,weight_change=change,objective=value)






def free_inverse(problem,coef,observation):
    """Segment-wise multistart; no target/demand inputs or tie breaking."""
    candidates=[]
    for lo,hi in zip(np.r_[0,KNOTS,6][:-1],np.r_[0,KNOTS,6][1:]):
        def fun(z):return problem.L@(basis(np.array([z[0]]),np.array([z[1]]),problem.p,curvature=problem.curvature)[0][0]@coef-observation)
        def jac(z):
            _,dt,da=basis(np.array([z[0]]),np.array([z[1]]),problem.p,curvature=problem.curvature)
            return problem.L@np.column_stack([dt[0]@coef,da[0]@coef])
        for th in [-20,-10,0,10,20]:
            opt=least_squares(fun,[th,(lo**problem.p+hi**problem.p)/2],jac=jac,
                              bounds=([-20,lo**problem.p],[20,hi**problem.p]),max_nfev=100)
            candidates.append([float(np.dot(opt.fun,opt.fun)),float(opt.x[0]),float(opt.x[1]**(1/problem.p))])
    best=min(candidates,key=lambda q:(q[0],q[2],q[1]))
    roots=[q for q in candidates if q[0]<=best[0]+1e-7]
    unique=[]
    for q in sorted(roots,key=lambda q:(q[2],q[1])):
        if not any(np.linalg.norm(np.array(q[1:])-r[1:])<1e-4 for r in unique):unique.append(q)
    chosen=unique[0]
    A=chosen[2];theta=chosen[1]
    w,dw=interpolation(np.array([A]),KNOTS);rho=coef[8:14];t=theta/15;a=A**problem.p
    dt=np.array([(w@coef[4:8])[0],(rho[1]+2*t*rho[2]+a*(rho[4]+2*t*rho[5]))/15])
    if problem.curvature: dt[0] += 2*coef[14]*theta/225
    if A>0:
        dA=np.array([(dw@coef[:4]+(dw@coef[4:8])*theta)[0],(rho[3]+t*rho[4]+t*t*rho[5])*problem.p*A**(problem.p-1)])
        scaled=problem.L@np.column_stack([dt,dA])@np.diag([1.,.25])
        sv=np.linalg.svd(scaled,compute_uv=False)
        conditioning=dict(physical_derivative_defined=True,scaled_singular_values=sv.tolist(),
                          condition=float(sv[0]/max(sv[1],1e-14)))
    else:
        conditioning=dict(physical_derivative_defined=False,reason='Ratio derivative at A=0 undefined for p<1')
    return dict(weighted_cost=chosen[0],theta_deg=chosen[1],A_diopters=chosen[2],
                equivalent_minima=unique,exact_root_cost_tolerance=1e-10,is_near_exact_root=chosen[0]<1e-10,extrapolation=bool(chosen[2]<KNOTS[0] or chosen[2]>KNOTS[-1]),
                branch_rule='Smallest A then theta among equivalent minima; no labels',conditioning=conditioning,
                theta_bound=bool(abs(chosen[1])>=20-1e-5),A_bound=bool(chosen[2]<=1e-5 or chosen[2]>=6-1e-5))


def save(output,problem,initial,x,coef,history,converged,meta,records,selected,provenance,covariance,optical_loss='quadratic'):
    th,a,A,_=problem.decode(x); pred=problem.forward_matrix(x)@coef
    initial_th,_,initial_A,_=problem.decode(initial)
    residual=pred-problem.y
    summary=[]
    for group,j in enumerate(selected):
        mask=problem.groups==group
        local_A=A[mask]
        if problem.model == 'm2':
            m2_phys=m2_model.forward_jac_m2(th[mask],local_A,coef)[1]
            valid_derivative=np.ones(len(local_A),bool)
        else:
            w,dw=interpolation(local_A,KNOTS)
            local_t=th[mask]/15;rho=coef[8:14]
            dtheta=np.column_stack([w@coef[4:8],(rho[1]+2*local_t*rho[2]+a[mask]*(rho[4]+2*local_t*rho[5]))/15])
            if problem.curvature: dtheta[:, 0] += 2*coef[14]*th[mask]/225
            valid_derivative=local_A>0
            dphysical=np.column_stack([dw@coef[:4]+(dw@coef[4:8])*th[mask],
                                      (rho[3]+local_t*rho[4]+local_t**2*rho[5])*problem.p*np.maximum(local_A,1e-12)**(problem.p-1)])
        # A=0 has an undefined ratio derivative for p<1. Exclude it from
        # conditioning statistics and report its count, rather than inventing
        # a finite physical Jacobian by multiplying transformed zero/infinity.
        phys=(m2_phys if problem.model == 'm2' else np.stack([dtheta,dphysical],axis=2))[valid_derivative]
        scaled=np.einsum('ab,nbc->nac',problem.L,phys)*np.array([1.,.25])[None,None,:]
        singular=np.linalg.svd(scaled,compute_uv=False)
        conditions=singular[:,0]/np.maximum(singular[:,1],1e-14)
        summary.append(dict(fixation_index=j,capture=meta[j]['capture'],target_theta_deg=problem.targets[group],
                            nominal_D=problem.demands[group],count=int(mask.sum()),theta_mean=float(th[mask].mean()),
                            A_mean=float(A[mask].mean()),theta_std=float(th[mask].std()),A_std=float(A[mask].std()),
                            mean_d_bias=float(residual[mask,0].mean()),mean_rho4_bias=float(residual[mask,1].mean()),
                            d_rmse=float(np.sqrt(np.mean(residual[mask,0]**2))),
                            rho4_rmse=float(np.sqrt(np.mean(residual[mask,1]**2))),
                            scaled_noise_jacobian_condition_quantiles=np.quantile(conditions,[.05,.5,.95]).tolist() if len(conditions) else [],
                            physical_derivative_undefined_count=int((~valid_derivative).sum()),
                            scaled_min_singular_quantiles=np.quantile(singular[:,1],[.05,.5,.95]).tolist() if len(singular) else [],
                            optical_near_singular_fraction=float(np.mean(singular[:,1]<1e-3*singular[:,0])) if len(singular) else None,
                            anchor_free_mean_inverse=(m2_model.free_inverse_m2(coef,problem.W,problem.y[mask].mean(axis=0))
                                                      if problem.model == 'm2' else
                                                      free_inverse(problem,coef,problem.y[mask].mean(axis=0)))))
        summary[-1]['mean_theta_anchor_offset_deg']=float(th[mask].mean()-problem.targets[group])
        summary[-1]['mean_A_anchor_offset_D']=float(A[mask].mean()-problem.demands[group])
        summary[-1]['free_theta_offset_nominal_deg']=summary[-1]['anchor_free_mean_inverse']['theta_deg']-float(problem.targets[group])
        summary[-1]['free_A_offset_nominal_D']=summary[-1]['anchor_free_mean_inverse']['A_diopters']-float(problem.demands[group])
    grid_theta,grid_A=np.meshgrid(np.linspace(-15,15,9),np.linspace(KNOTS[0],KNOTS[-1],9))
    grid_th,grid_A=grid_theta.ravel(),grid_A.ravel()
    if problem.model == 'm2':
        grid_phys=m2_model.forward_jac_m2(grid_th,grid_A,coef)[1]
    else:
        grid_H,grid_dt,grid_da=basis(grid_th,grid_A**problem.p,problem.p,curvature=problem.curvature)
        grid_phys=np.stack([grid_dt@coef,(grid_da@coef)*(problem.p*grid_A**(problem.p-1))[:,None]],axis=2)
    grid_J=np.einsum('ab,nbc->nac',problem.L,grid_phys)*np.array([1.,.25])[None,None,:]
    grid_sv=np.linalg.svd(grid_J,compute_uv=False)
    grid_condition=grid_sv[:,0]/np.maximum(grid_sv[:,1],1e-14)
    grid_diagnostics=dict(noise_physical_scales=[1.,.25],theta_support=[-15,15],A_support=[float(KNOTS[0]),float(KNOTS[-1])],
                          condition_quantiles=np.quantile(grid_condition,[.05,.5,.95]).tolist(),
                          minimum_singular_quantiles=np.quantile(grid_sv[:,1],[.05,.5,.95]).tolist(),
                          near_singular_fraction=float(np.mean(grid_sv[:,1]<1e-3*grid_sv[:,0])))
    effective_settings = SETTINGS.copy()
    effective_settings.update(theta_anchor_scale_deg=problem.theta_anchor_scale)
    if problem.model == 'm2':
        effective_settings.update(extrapolation='none (smooth analytic m2 forms)')
    if problem.curvature:
        effective_settings.update(curvature_prior=problem.curvature_strength,
                                  curvature_scale_output=problem.curvature_scale_output)
    if problem.previous_means is not None:
        effective_settings.update(previous_mean_strength=problem.previous_mean_strength,
                                  previous_mean_scale_deg=problem.previous_mean_scale_deg)
    diag=dict(calibrated_grid_conditioning=grid_diagnostics,optical_loss=optical_loss,coefficient_rank=problem.free_coefficient_count,settings=effective_settings,measurement_support='All gated valid central-core frames; S_j=V_j, no event exclusions',provenance=provenance,p=problem.p,state_count=len(th),converged=converged,history=history,
              rmse=np.sqrt(np.mean(residual**2,axis=0)).tolist(),fixations=summary,covariance=covariance.tolist(),
              theta_bound_fraction=float(np.mean(np.abs(th)>=20-1e-6)),
              A_bound_fraction=float(np.mean((A<=1e-6)|(A>=6-1e-6))),
              A_extrapolation_fraction=float(np.mean((A<KNOTS[0])|(A>KNOTS[-1]))),
              validation='In-sample optical residuals and assumed nominal mean anchors, not measured state accuracy',
              termination='stationarity criteria satisfied' if converged else 'budget/iteration limit or nonsmooth knot stall; not converged')
    model=dict(schema='profiled_piecewise_displacement_power_ratio_v1',coefficient_order='b[4],s[4],rho[1,t,t²,a,at,at²]',
               coefficients=coef.tolist(),initial_coefficients=problem.initial_coef.tolist(),p=problem.p,
               demand_knots=KNOTS.tolist(),theta_bounds=[-20,20],A_bounds=[0,6],
               precision=problem.W.tolist(),prior_precision=problem.prior_W.tolist(),diagnostics=diag)
    if problem.curvature:
        model.update(schema='profiled_curved_displacement_power_ratio_v2',
                     coefficient_order='b[4],s[4],rho[1,t,t²,a,at,at²],q',
                     curvature_strength=problem.curvature_strength,
                     curvature_scale_output=problem.curvature_scale_output)
    if problem.previous_means is not None:
        model.update(previous_mean_prior=dict(strength=problem.previous_mean_strength,
                     scale_deg=problem.previous_mean_scale_deg, means_deg=problem.previous_means.tolist(),
                     source=problem.previous_mean_provenance, nominal_anchor_policy='retained unchanged'))
    if problem.coefficient_map is not None:
        model.update(coefficient_map=problem.coefficient_map.tolist(),
                     free_coefficient_count=problem.free_coefficient_count)
        diag.update(coefficient_map=problem.coefficient_map.tolist(),
                    free_coefficient_count=problem.free_coefficient_count)
    if problem.model == 'm2':
        model.update(schema=m2_model.SCHEMA,coefficient_order=m2_model.COEFFICIENT_ORDER,model_type=m2_model.MODEL_TYPE,
                     coefficient_names=m2_model.COEFFICIENT_NAMES,p=None,state_encoding='theta/15, A/4 (A in diopters)',
                     basis_description=m2_model.BASIS_DESCRIPTION)
        model.pop('demand_knots')
    if problem.model != 'piecewise' or problem.theta_anchor_scale != 1.:
        model.update(theta_anchor_scale_deg=problem.theta_anchor_scale)
    (output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    (output/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    (output/'core_intervals.json').write_text(json.dumps(meta,indent=2)+'\n')
    flat=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in summary]
    with (output/'fixation_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    fig,axes=plt.subplots(4,2,figsize=(17,12),layout='constrained')
    optfig,optaxes=plt.subplots(4,2,figsize=(17,12),layout='constrained')
    cursor=0
    mapping={j:g for g,j in enumerate(selected)}
    for record in records:
        c=record['capture']; frame=record['arrays']['frame_index']; n=len(frame)
        lookup=np.full(n,-1,dtype=int)
        for j in selected:
            indices=np.flatnonzero(record['groups']==j)
            if len(indices):lookup[indices]=np.arange(cursor,cursor+len(indices));cursor+=len(indices)
        with (output/f'capture_{c}_states.csv').open('w',newline='') as f:
            fields=['frame_index','timestamp_ms','status','fixation_index','d','rho4','theta_deg','A_diopters','predicted_d','predicted_rho4']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i in range(n):
                k=lookup[i];j=int(record['groups'][i]);row=dict(frame_index=int(frame[i]),timestamp_ms=float(record['arrays']['timestamp_ms'][i]),fixation_index=j)
                row['status']='invalid_detection' if not record['valid'][i] else 'calibration_core' if k>=0 else 'heldout_core' if j>=0 else 'excluded_fixation_edge' if record['full'][i] else 'freeview'
                if record['valid'][i]:row.update(d=record['y'][i,0],rho4=record['y'][i,1])
                if k>=0:row.update(theta_deg=th[k],A_diopters=A[k],predicted_d=pred[k,0],predicted_rho4=pred[k,1])
                writer.writerow(row)
        for col,(values,initial_values,label) in enumerate([(th,initial_th,'Gaze (degrees)'),(A,initial_A,'Accommodation (D)')]):
            full=np.full(n,np.nan);before=full.copy();good=lookup>=0
            full[good]=values[lookup[good]];before[good]=initial_values[lookup[good]]
            axes[c-1,col].plot(frame,before,alpha=.35,lw=.4,label='Initialization')
            axes[c-1,col].plot(frame,full,lw=.5,label='Joint profiled')
            for j in selected:
                if meta[j]['capture']!=f'capture_{c}_detections.pkl':continue
                m=meta[j];target=m['target_theta_deg'] if col==0 else m['demand_diopters_label']
                axes[c-1,col].hlines(target,frame[m['core_start_row']],frame[m['core_end_row_exclusive']-1],color='black',linestyles='dashed',lw=.8)
            axes[c-1,col].set_title(f'Capture {c}: {label}');axes[c-1,col].set_xlabel('Original frame index')
        for col in range(2):
            measured=np.where(record['valid'],record['y'][:,col],np.nan);predicted=np.full(n,np.nan);good=lookup>=0;predicted[good]=pred[lookup[good],col]
            optaxes[c-1,col].plot(frame,measured,color='gray',alpha=.4,lw=.4,label='Measured')
            optaxes[c-1,col].plot(frame,predicted,lw=.5,label='Predicted')
            optaxes[c-1,col].set_title(f'Capture {c}: '+['d','rho4'][col]);optaxes[c-1,col].set_xlabel('Original frame index')
    axes[0,0].legend();optaxes[0,0].legend()
    fig.suptitle('Joint profiled states; dashed nominal MEAN anchors, not per-frame truth')
    fig.savefig(output/'state_overview.png',dpi=140);plt.close(fig)
    optfig.suptitle('Measured and predicted observables; retained pupil-gated central cores')
    optfig.savefig(output/'observable_overview.png',dpi=140);plt.close(optfig)
    return diag
