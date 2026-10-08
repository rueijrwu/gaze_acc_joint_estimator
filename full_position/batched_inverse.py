"""FP64 array bounded TRF and exact-Newton certificates for power responses.

Frames and all starts share device kernels, but have independent trust radii,
evaluation budgets, termination and polish masks. NumPy is a diagnostic backend.
The TRF equations follow SciPy's BSD implementation; see SCIPY_LICENSE.txt.
There is no scalar optimizer fallback and no held measurement input here.
"""
from __future__ import annotations

import numpy as np
from .accommodation import PowerResponseModel, response_arrays
from .invert import STARTS, STATIONARITY_TOLERANCE, PHYSICAL_CORRECTION_TOLERANCE
from .model import STATE_SCALE, THETA_SCALE, LOWER, UPPER
from .noise import whitening


def backend_array(backend, device=0):
    if backend == "numpy":
        return np, np.asarray
    if backend == "torch":
        from .torch_arrays import TorchArrays
        xp=TorchArrays(device)
        return xp,xp.asnumpy
    if backend != "cupy":
        raise ValueError("backend must be numpy, cupy or torch")
    import cupy as cp
    cp.cuda.Device(device).use()
    return cp, cp.asnumpy


class Objective:
    """Precontract context coefficients once; no (...,6,27) design allocation."""
    def __init__(self, model, r, y, W, indices, xp):
        self.xp, self.exponent = xp, model.exponent
        self.basis=getattr(model,"response_basis","shifted_boxcox")
        self.lower,self.upper=getattr(model,"lower",LOWER),getattr(model,"upper",UPPER)
        b, r = xp.asarray(model.beta), xp.asarray(r)
        n = len(r)
        C = xp.zeros((n, 6, 11), dtype=xp.float64)
        C[:, 0::2, :7] = b[:7]
        C[:, 0::2, 7:] = r[:, :, 0, None]*b[11:15]+r[:, :, 1, None]*b[15:19]
        C[:, 1::2, 7:] = b[7:11]+r[:, :, 0, None]*b[19:23]+r[:, :, 1, None]*b[23:27]
        retained = C[:, indices] if indices.ndim == 1 else xp.take_along_axis(C, xp.asarray(indices)[:,:,None],axis=1)
        self.C = xp.einsum("nij,njk->nik", W, retained)
        self.target = xp.einsum("nij,nj->ni", W, y)

    def evaluate(self, z, second=False):
        xp = self.xp
        t, A = z[:, 0]*STATE_SCALE[0]/THETA_SCALE, z[:, 1]*STATE_SCALE[1]
        p,pa,paa = response_arrays(A,self.exponent,xp,self.basis)
        pa = pa*STATE_SCALE[1]
        paa = paa*STATE_SCALE[1]**2
        k = STATE_SCALE[0]/THETA_SCALE
        o, q = xp.ones_like(t), xp.zeros_like(t)
        B = xp.stack((o,A,t,t*p,t*t,t*t*p,t**3,o,t,p,t*p), -1)
        Bt = xp.stack((q,q,o,p,2*t,2*t*p,3*t*t,q,o,q,p), -1)*k
        Ba = xp.stack((q,o*STATE_SCALE[1],q,t*pa,q,t*t*pa,q,q,q,pa,t*pa), -1)
        D = xp.stack((Bt, Ba), -1)
        f = xp.einsum("ncb,nb->nc", self.C, B)-self.target
        J = xp.einsum("ncb,nbk->nck", self.C, D)
        cost = xp.sum(f*f, -1)/2
        g = xp.einsum("nck,nc->nk", J, f)
        H = xp.einsum("nci,ncj->nij", J, J)
        if second:
            Btt = xp.stack((q,q,q,q,2*o,2*p,6*t,q,q,q,q), -1)*k*k
            Bta = xp.stack((q,q,q,pa,q,2*t*pa,q,q,q,q,pa), -1)*k
            Baa = xp.stack((q,q,q,t*paa,q,t*t*paa,q,q,q,paa,t*paa), -1)
            weights = xp.einsum("nc,ncb->nb", f, self.C)
            H[:, 0, 0] += xp.sum(weights*Btt, -1)
            H[:, 0, 1] += xp.sum(weights*Bta, -1)
            H[:, 1, 0] = H[:, 0, 1]
            H[:, 1, 1] += xp.sum(weights*Baa, -1)
        return cost, f, J, g, H


def _norm(x, xp):
    return xp.sqrt(xp.sum(x*x, -1))


def _strict(x, lo, hi, xp, relative=False):
    ld, ud = x-lo, hi-x
    if relative:
        lm = ld <= xp.minimum(ud, 1e-10*xp.maximum(1, xp.abs(lo)))
        um = ud <= xp.minimum(ld, 1e-10*xp.maximum(1, xp.abs(hi)))
        a, b = lo+1e-10*xp.maximum(1, xp.abs(lo)), hi-1e-10*xp.maximum(1, xp.abs(hi))
    else:
        lm, um = x <= lo, x >= hi
        a, b = xp.nextafter(lo, hi), xp.nextafter(hi, lo)
    out = xp.where(um, b, xp.where(lm, a, x))
    return xp.where((out < lo) | (out > hi), (lo+hi)/2, out)


def _bound_stride(x, s, lo, hi, xp):
    safe = xp.where(s != 0, s, 1)
    strides = xp.where(s != 0, xp.maximum((lo-x)/safe, (hi-x)/safe), xp.inf)
    stride = xp.min(strides, -1)
    return stride, strides == stride[:, None]


def _qvalue(H, g, p, xp):
    return .5*xp.einsum("ni,nij,nj->n", p, H, p)+xp.sum(g*p, -1)


def _qline(H, g, s, xp, origin=None):
    a = .5*xp.einsum("ni,nij,nj->n", s, H, s)
    b = xp.sum(g*s, -1)
    c = xp.zeros_like(a)
    if origin is not None:
        b += xp.einsum("ni,nij,nj->n", origin, H, s)
        c = _qvalue(H, g, origin, xp)
    return a, b, c


def _qmin(a, b, lo, hi, c, xp):
    extremum = -.5*b/xp.where(a != 0, a, 1)
    ts = xp.stack((lo, hi, extremum), -1)
    vals = ts*(a[:, None]*ts+b[:, None])+c[:, None]
    vals[:, 2] = xp.where((a != 0) & (extremum > lo) & (extremum < hi), vals[:, 2], xp.inf)
    k = xp.argmin(vals, -1)
    rows = xp.arange(len(a))
    return ts[rows, k], vals[rows, k]


def _trust(s, uf, V, delta, alpha, m, xp):
    """More secular equation, independent stopping masks, SciPy rtol=.01."""
    tiny = np.finfo(float).tiny
    full = s[:, -1] > np.finfo(float).eps*m*s[:, 0]
    gn = -xp.einsum("nij,nj->ni", V, uf/xp.maximum(s, tiny))
    use_gn = full & (_norm(gn, xp) <= delta)
    suf, s2 = s*uf, s*s
    def phi(a):
        den = xp.maximum(s2+a[:, None], tiny)
        norm = _norm(suf/den, xp)
        derivative = -xp.sum(suf*suf/den**3, -1)/xp.maximum(norm, tiny)
        return norm-delta, derivative
    high = _norm(suf, xp)/xp.maximum(delta, tiny)
    f, df = phi(xp.zeros_like(delta))
    low = xp.where(full, -f/xp.where(df != 0, df, -1), 0)
    low = xp.maximum(low, 0)
    a = xp.where((~full) & (alpha == 0), xp.maximum(.001*high, xp.sqrt(low*high)), alpha)
    live = ~use_gn
    for _ in range(10):
        a = xp.where(live & ((a < low) | (a > high)), xp.maximum(.001*high, xp.sqrt(low*high)), a)
        f, df = phi(a)
        high = xp.where(live & (f < 0), a, high)
        ratio = f/xp.where(df != 0, df, -1)
        low = xp.where(live, xp.maximum(low, a-ratio), low)
        a = xp.where(live, a-(f+delta)*ratio/xp.maximum(delta, tiny), a)
        live &= xp.abs(f) >= .01*delta
    p = -xp.einsum("nij,nj->ni", V, suf/xp.maximum(s2+a[:, None], tiny))
    p *= (delta/xp.maximum(_norm(p, xp), tiny))[:, None]
    return xp.where(use_gn[:, None], gn, p), xp.where(use_gn, 0, a)


def _select(x, H, g, ph, d, delta, lo, hi, theta, xp):
    p = d*ph
    interior = xp.all((x+p >= lo) & (x+p <= hi), -1)
    stride, hit = _bound_stride(x, p, lo, hi, xp)
    # Interior rows do not need reflection, but keep masked arithmetic finite.
    stride = xp.where(interior, 1, stride)
    rh = xp.where(hit, -ph, ph)
    p0 = ph*stride[:, None]
    r = d*rh
    a = xp.sum(rh*rh, -1)
    b = xp.sum(p0*rh, -1)
    c = xp.minimum(xp.sum(p0*p0, -1)-delta*delta, 0)
    disc = xp.sqrt(xp.maximum(b*b-a*c, 0))
    q = -(b+xp.copysign(disc, b))
    t1 = q/xp.where(a != 0, a, 1)
    t2 = c/xp.where(q != 0, q, 1)
    to_tr = xp.maximum(t1, t2)
    to_bound, _ = _bound_stride(x+d*p0, r, lo, hi, xp)
    limit = xp.minimum(to_bound, to_tr)
    rl = xp.where(limit > 0, (1-theta)*stride/xp.where(limit > 0, limit, 1), 0)
    ru = xp.where(limit > 0, xp.where(limit == to_bound, theta*to_bound, to_tr), -1)
    aa, bb, cc = _qline(H, g, rh, xp, p0)
    rt, rv = _qmin(aa, bb, rl, ru, cc, xp)
    rv = xp.where(rl <= ru, rv, xp.inf)
    reflected = p0+rt[:, None]*rh
    shortened = p0*theta[:, None]
    pv = _qvalue(H, g, shortened, xp)
    agh = -g
    tr = delta/xp.maximum(_norm(agh, xp), np.finfo(float).tiny)
    tb, _ = _bound_stride(x, d*agh, lo, hi, xp)
    alimit = xp.where(tb < tr, theta*tb, tr)
    aa, bb, cc = _qline(H, g, agh, xp)
    at, av = _qmin(aa, bb, xp.zeros_like(delta), alimit, cc, xp)
    ag = at[:, None]*agh
    pick_p = (pv < rv) & (pv < av)
    pick_r = (rv < pv) & (rv < av)
    selected = xp.where(pick_p[:, None], shortened, xp.where(pick_r[:, None], reflected, ag))
    selected = xp.where(interior[:, None], ph, selected)
    return d*selected, selected, -_qvalue(H, g, selected, xp)


def _trf(obj, initial, max_nfev, xp):
    lo, hi = xp.asarray(obj.lower/STATE_SCALE), xp.asarray(obj.upper/STATE_SCALE)
    x = _strict(initial, lo, hi, xp, relative=True)
    cost, f, J, g, _ = obj.evaluate(x)
    v = xp.where(g < 0, hi-x, xp.where(g > 0, x-lo, 1))
    delta = _norm(x/xp.sqrt(v), xp)
    delta = xp.where(delta == 0, 1, delta)
    alpha = xp.zeros(len(x))
    status = xp.zeros(len(x), dtype=xp.int32)
    nfev = xp.ones(len(x), dtype=xp.int32)
    for _ in range(max_nfev):
        v = xp.where(g < 0, hi-x, xp.where(g > 0, x-lo, 1))
        dv = xp.sign(g)
        gn = xp.max(xp.abs(g*v), -1)
        status = xp.where((status == 0) & (gn < 1e-8), 1, status)
        active = (status == 0) & (nfev < max_nfev)
        if not bool(xp.any(active)):
            break
        d = xp.sqrt(v)
        diagonal, gh = g*dv, d*g
        Jh = J*d[:, None, :]
        augmented = xp.zeros((len(x), J.shape[1]+2, 2))
        augmented[:, :J.shape[1]] = Jh
        augmented[:, -2, 0] = xp.sqrt(xp.maximum(diagonal[:, 0], 0))
        augmented[:, -1, 1] = xp.sqrt(xp.maximum(diagonal[:, 1], 0))
        U, s, Vt = xp.linalg.svd(augmented, full_matrices=False)
        uf = xp.einsum("nmi,nm->ni", U[:, :J.shape[1]], f)
        Hh = xp.einsum("nmi,nmj->nij", Jh, Jh)
        Hh[:, 0, 0] += diagonal[:, 0]
        Hh[:, 1, 1] += diagonal[:, 1]
        theta = xp.maximum(.995, 1-gn)
        pending = xp.copy(active)
        for _ in range(max_nfev):
            ph, candidate_alpha = _trust(s, uf, Vt.swapaxes(-1,-2), delta, alpha, J.shape[1], xp)
            step, sh, predicted = _select(x, Hh, gh, ph, d, delta, lo, hi, theta, xp)
            trial = _strict(x+step, lo, hi, xp)
            tc, tf, tJ, tg, _ = obj.evaluate(trial)
            nfev += xp.asarray(pending,dtype=xp.int32)
            actual = cost-tc
            finite = xp.isfinite(tc)
            ratio = xp.where(predicted > 0, actual/xp.where(predicted > 0,predicted,1),
                             xp.where((predicted == 0) & (actual == 0), 1, 0))
            sn = _norm(sh, xp)
            new_delta = xp.where(ratio < .25, .25*sn,
                                 xp.where((ratio > .75) & (sn > .95*delta), 2*delta, delta))
            small = _norm(step,xp) < 1e-12*(1e-12+_norm(x,xp))
            status = xp.where(pending & finite & small, 3, status)
            accepted = pending & finite & (actual > 0)
            x = xp.where(accepted[:,None],trial,x)
            cost = xp.where(accepted,tc,cost)
            f = xp.where(accepted[:,None],tf,f)
            J = xp.where(accepted[:,None,None],tJ,J)
            g = xp.where(accepted[:,None],tg,g)
            update = pending & finite & ~small
            alpha = xp.where(update,candidate_alpha*delta/xp.maximum(new_delta,np.finfo(float).tiny),alpha)
            delta = xp.where(pending & ~finite,.25*sn,xp.where(update,new_delta,delta))
            pending &= ~accepted & (status == 0) & (nfev < max_nfev)
            if not bool(xp.any(pending)):
                break
    pg = xp.max(xp.abs(x-xp.clip(x-g,lo,hi)), -1)
    return x, cost, pg, status, nfev


def _polish(obj, z, xp, max_steps=20):
    lo, hi = xp.asarray(obj.lower/STATE_SCALE), xp.asarray(obj.upper/STATE_SCALE)
    scale = xp.asarray(STATE_SCALE)
    live = xp.ones(len(z),dtype=bool)
    certified = xp.zeros(len(z),dtype=bool)
    last_change = xp.full(len(z),xp.nan)
    steps = xp.zeros(len(z),dtype=xp.int32)
    for iteration in range(max_steps+1):
        cost, _, _, g, H = obj.evaluate(z, second=True)
        active = ((z-lo < 1e-9) & (g > 1e-7)) | ((hi-z < 1e-9) & (g < -1e-7))
        free = ~active
        both = xp.all(free,-1)
        one = xp.sum(free,-1) == 1
        eig, vec = xp.linalg.eigh(H)
        solo = xp.sum(xp.diagonal(H,axis1=-2,axis2=-1)*free,-1)
        eig_min = xp.where(both,eig[:,0],xp.where(one,solo,0))
        curvature_scale = xp.maximum(1,xp.where(both,xp.max(xp.abs(eig),-1),xp.abs(solo)))
        minimum = eig_min >= -1e-10*curvature_scale
        shift = xp.maximum(0,1e-12*curvature_scale-eig_min)
        h = H*free[:,:,None]*free[:,None,:]
        h[:,0,0] += xp.where(free[:,0],shift,1)
        h[:,1,1] += xp.where(free[:,1],shift,1)
        step = xp.linalg.solve(h,(-g*free)[...,None])[...,0]
        negative = ~minimum & (_norm(g*free,xp) < 1e-8)
        direction = xp.where(both[:,None],vec[:,:,0],xp.asarray(free,dtype=xp.float64))
        step = xp.where(negative[:,None],.05*direction,step)
        projected = xp.clip(z+step,lo,hi)-z
        correction = xp.max(xp.abs(projected*scale),-1)
        stationarity = xp.max(xp.abs(z-xp.clip(z-g,lo,hi)),-1)
        probe = obj.evaluate(z+projected)[0]
        stable = (correction < PHYSICAL_CORRECTION_TOLERANCE) & (xp.abs(probe-cost) <= 1e-10*(1+cost))
        accepted_cert = live & (stationarity <= STATIONARITY_TOLERANCE) & minimum & (correction <= PHYSICAL_CORRECTION_TOLERANCE) & stable
        certified |= accepted_cert
        steps = xp.where(live,iteration,steps)
        live &= ~accepted_cert
        if iteration == max_steps or not bool(xp.any(live)):
            break
        step *= (.25/xp.maximum(xp.max(xp.abs(step),-1),.25))[:,None]
        waiting = xp.copy(live)
        next_z = xp.copy(z)
        for power in range(24):
            trial = xp.clip(z+step*2.**(-power),lo,hi)
            tc,_,_,tg,_ = obj.evaluate(trial)
            pg = xp.max(xp.abs(trial-xp.clip(trial-tg,lo,hi)),-1)
            roundoff = minimum & (correction <= PHYSICAL_CORRECTION_TOLERANCE) & (xp.abs(tc-cost) <= 1e-10*(1+cost)) & (pg < .5*stationarity)
            descent = tc <= cost+xp.minimum(0,1e-4*xp.sum(g*(trial-z),-1))+8*np.finfo(float).eps*(1+cost)
            ok = waiting & (descent | roundoff)
            next_z = xp.where(ok[:,None],trial,next_z)
            last_change = xp.where(ok,tc-cost,last_change)
            waiting &= ~ok
            if not bool(xp.any(waiting)):
                break
        z = next_z
        live &= ~waiting
    # Diagnostics must describe each candidate's frozen final state.
    cost,_,J,g,H = obj.evaluate(z,second=True)
    active = ((z-lo < 1e-9) & (g > 1e-7)) | ((hi-z < 1e-9) & (g < -1e-7))
    return dict(z=z,cost=cost,J=J,certified=certified,steps=steps,
                stationarity=stationarity,correction=correction,stable=stable,
                minimum=minimum,free=~active,eigenvalues=xp.linalg.eigvalsh(H),
                Hessian=H,last_change=last_change)


def solve_batch(model, r, y, covariance, indices=None, *, backend="cupy", device=0,
                starts=STARTS, max_nfev=100):
    """Retained observations only. All frames × starts are device-vectorized."""
    if not isinstance(model,PowerResponseModel):
        raise ValueError("This backend requires a fixed power-response model")
    r,y,covariance = (np.asarray(v,float) for v in (r,y,covariance))
    indices = np.arange(6) if indices is None else np.asarray(indices,int)
    lower,upper=getattr(model,"lower",LOWER),getattr(model,"upper",UPPER)
    starts = np.clip(starts,lower,upper) if starts is STARTS and hasattr(model,"lower") else np.asarray(starts,float)
    m = indices.shape[-1] if indices.ndim in (1,2) else 0
    if (r.shape != (len(y),3,2) or y.shape != (len(r),m)
            or covariance.shape != (len(r),m,m)
            or (indices.ndim == 2 and indices.shape[0] != len(r))):
        raise ValueError("Supply exactly retained observations and covariance marginals")
    ids = np.broadcast_to(indices,(len(r),m))
    if (m < 4 or np.any(ids < 0) or np.any(ids > 5)
            or np.any(np.diff(np.sort(ids,axis=1),axis=1)==0)):
        raise ValueError("At least two retained P4 points are required")
    if (not all(np.isfinite(v).all() for v in (r,y,covariance,starts))
            or starts.ndim != 2 or starts.shape[1] != 2 or not len(starts)
            or np.any(starts < lower) or np.any(starts > upper) or max_nfev < 1):
        raise ValueError("Finite inputs, bounded starts and positive budget required")
    xp,to_cpu = backend_array(backend,device)
    n,k = len(r),len(starts)
    if not n:
        return []
    if backend != "numpy":
        chol=xp.linalg.cholesky(xp.asarray(covariance))
        W=xp.linalg.solve(chol,xp.broadcast_to(xp.eye(m),chol.shape))
    else:
        W=whitening(covariance)
    batch_indices=indices if indices.ndim==1 else np.repeat(indices,k,axis=0)
    obj = Objective(model,xp.asarray(np.repeat(r,k,axis=0)),
                    xp.asarray(np.repeat(y,k,axis=0)),xp.repeat(xp.asarray(W),k,axis=0),batch_indices,xp)
    initial = xp.asarray(np.tile(starts/STATE_SCALE,(n,1)))
    z,cost,pg,status,nfev = _trf(obj,initial,max_nfev,xp)
    before = dict(state=to_cpu(z*xp.asarray(STATE_SCALE)),cost=to_cpu(2*cost),
                  pg=to_cpu(pg),status=to_cpu(status),nfev=to_cpu(nfev))
    device_solved=_polish(obj,z,xp)
    device_sv=xp.linalg.svd(device_solved["J"]/xp.asarray(STATE_SCALE),compute_uv=False)
    solved = {key:to_cpu(value) for key,value in device_solved.items()}
    solved["state"] = solved["z"]*STATE_SCALE
    # Rank is computed on physical derivatives, independently of encoded units.
    solved["sv"] = to_cpu(device_sv)
    return _assemble(solved,before,starts,n,k,backend,lower,upper)


def _assemble(s,before,starts,n,k,backend,lower=LOWER,upper=UPPER):
    results=[]
    for row in range(n):
        branches,candidates=[],[]
        for j in range(k):
            i=row*k+j
            free=s["free"][i]
            eig=np.linalg.eigvalsh(s["Hessian"][i][np.ix_(free,free)]).tolist() if free.any() else []
            certificate=dict(certified=bool(s["certified"][i]),steps=int(s["steps"][i]),
                stationarity_encoded=float(s["stationarity"][i]),physical_correction=float(s["correction"][i]),
                stable_cost=bool(s["stable"][i]),local_minimum=bool(s["minimum"][i]),
                free_curvature_eigenvalues=eig,last_cost_change=None if np.isnan(s["last_change"][i]) else float(s["last_change"][i]),
                reason="certified_stationary_minimum" if s["certified"][i] else "negative_curvature" if not s["minimum"][i] else "unresolved_stationarity_or_correction")
            status=int(before["status"][i]); success=status>0
            initial=dict(state=before["state"][i].tolist(),cost=float(before["cost"][i]),status=status,
                message="Batched bounded TRF",nfev=int(before["nfev"][i]),success=success,
                stationarity_encoded=float(before["pg"][i]),accepted_before_polish=bool(success and before["pg"][i]<=STATIONARITY_TOLERANCE),
                termination="budget_exhausted" if status==0 else "small_step" if status==3 else "gradient_stop")
            x=s["state"][i]; cost=float(2*s["cost"][i])
            candidates.append(dict(start=j,starting_state=starts[j].tolist(),initial=initial,state=x.tolist(),cost=cost,polish=certificate,accepted=certificate["certified"]))
            if not certificate["certified"]:
                continue
            sv=s["sv"][i]
            record=dict(state=x.tolist(),cost=cost,stationarity_encoded=certificate["stationarity_encoded"],
                singular_values_physical=sv.tolist(),certificate=certificate,
                rank=int(np.sum(sv>max(sv[0]*1e-6,1e-8))),at_bound=bool(np.any((x-lower<1e-5)|(upper-x<1e-5))))
            old=next((b for b in branches if np.all(np.abs(x-np.array(b["state"]))<(.01,.01))),None)
            if old is None: branches.append(record)
            elif cost<old["cost"]: old.update(record)
        common=dict(candidates=candidates,start_count=k,failed_starts=sum(not c["accepted"] for c in candidates),backend=backend)
        if not branches:
            results.append(dict(available=False,reason="no_converged_inverse",branches=[],**common)); continue
        branches.sort(key=lambda b:b["cost"])
        best=branches[0]["cost"]
        ties=[b for b in branches if b["cost"]<=best+1e-6*(1+best)]
        primary=min(ties,key=lambda b:(b["state"][1],b["state"][0]))
        plausible=[b for b in branches if b["cost"]<=best+2.]
        results.append(dict(available=True,reason="ok" if primary["rank"]==2 else "weak_rank",
            **{key:primary[key] for key in ("state","cost","stationarity_encoded","at_bound","rank","singular_values_physical")},
            ambiguous=len(plausible)>1,numerical_ties=len(ties),branches=branches,plausible_branches=plausible,
            plausible_delta=2.,recovered_starts=sum(c["accepted"] and not c["initial"]["accepted_before_polish"] for c in candidates),
            acceptance_contract="projected_gradient_1e-4; physical_correction_1e-5; stable_cost; critical_face_curvature",**common))
    return results


def solve_devices(model,r,y,covariance,indices,*,devices,backend="cupy",starts=STARTS,max_nfev=100):
    """Independent ordered frame chunks on all selected GPUs, concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    import cupy as cp
    devices=list(devices)
    if not devices or len(set(devices))!=len(devices):
        raise ValueError("Select at least one distinct GPU")
    count=cp.cuda.runtime.getDeviceCount()
    if any(d<0 or d>=count for d in devices):
        raise ValueError("Selected GPU does not exist")
    if len(devices)==1 or len(r)<len(devices):
        return solve_batch(model,r,y,covariance,indices,backend=backend,device=devices[0],starts=starts,max_nfev=max_nfev)
    weights=np.array([cp.cuda.Device(d).attributes["MultiProcessorCount"] for d in devices],float)
    boundaries=np.r_[0,np.rint(np.cumsum(weights)/weights.sum()*len(r)).astype(int)]
    boundaries[-1]=len(r)
    jobs=[]
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for j,device in enumerate(devices):
            part=slice(boundaries[j],boundaries[j+1])
            if boundaries[j]==boundaries[j+1]: continue
            ix=indices if np.ndim(indices)==1 else indices[part]
            jobs.append(executor.submit(solve_batch,model,r[part],y[part],covariance[part],ix,
                                        backend=backend,device=device,starts=starts,max_nfev=max_nfev))
        return [row for job in jobs for row in job.result()]
