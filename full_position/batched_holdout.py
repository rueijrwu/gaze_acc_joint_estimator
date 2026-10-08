"""Ordered raw holdouts with a sealed retained-only GPU solver boundary."""
import numpy as np
from .geometry import context
from .noise import reference_covariance,predictive_covariance
from .batched_inverse import solve_batch,solve_devices
from .invert import STARTS


def predict_batch(model,p,q,valid,held,pilot,reference,sigma,*,backend="cupy",device=0,devices=None,starts=STARTS,max_nfev=100):
    p,q=np.asarray(p,float),np.asarray(q,float)
    valid,held=np.asarray(valid,bool),np.asarray(held,int)
    n=len(p)
    if p.shape!=(n,3,2) or q.shape!=(n,3,2) or valid.shape!=(n,3) or held.shape!=(n,) or np.any((held<0)|(held>2)):
        raise ValueError("Raw batches require P1/P4 (...,3,2), fixed point validity and held indices")
    if not n: return []
    ctx=context(p)
    indices=np.array([[i for i in range(6) if i//2!=j] for j in held],dtype=int)
    point_ids=indices[:,::2]//2
    retained_valid=np.take_along_axis(valid,point_ids,axis=1).all(axis=1)
    # Excluded coordinates and validity are never read by the numerical path.
    raw=np.take_along_axis(q.reshape(n,6),indices,axis=1)
    finite=np.isfinite(raw).all(axis=1)
    available=ctx.valid & retained_valid & finite
    results=[dict(available=False,held_point=int(held[i]),branches=[],
                  reason="invalid_P1_geometry" if not ctx.valid[i] else "insufficient_retained_P4" if not retained_valid[i] else "invalid_subset_input") for i in range(n)]
    take=np.flatnonzero(available)
    if not len(take): return results
    centroids=np.take_along_axis(ctx.c[take],indices[take]%2,axis=1)
    y=(raw[take]-centroids)/ctx.ell[take,None]
    full_cov=reference_covariance(p[take],pilot,np.asarray(reference),np.asarray(sigma))
    ids=indices[take]
    cov=full_cov[np.arange(len(take))[:,None,None],ids[:,:,None],ids[:,None,:]]
    if devices is None:
        solved=solve_batch(model,ctx.r[take],y,cov,ids,backend=backend,device=device,starts=starts,max_nfev=max_nfev)
    else:
        solved=solve_devices(model,ctx.r[take],y,cov,ids,backend=backend,devices=devices,starts=starts,max_nfev=max_nfev)
    # Assemble small output objects on the host; optimization/certification is
    # already complete and no withheld measurements are supplied here.
    for local,i in enumerate(take):
        result=solved[local]
        result.update(held_point=int(held[i]),retained_image_channels="xy",retained_indices=ids[local].tolist())
        if result["available"]:
            omitted=np.array([2*held[i],2*held[i]+1])
            result["predictions"]=[]
            for branch in result["plausible_branches"]:
                value=model.predict(np.asarray(branch["state"]),ctx.r[i])[omitted]
                result["predictions"].append(dict(state=branch["state"],normalized=value.tolist(),pixel=(ctx.c[i]+ctx.ell[i]*value).tolist()))
            result["testable"]=bool(result["rank"]==2 and not result["ambiguous"])
            if result["testable"] and not result["at_bound"]:
                _,J=model.predict(np.asarray(result["state"]),ctx.r[i],True)
                V=predictive_covariance(full_cov[local],J,ids[local],omitted)
                if np.linalg.eigvalsh(V).min()>0:
                    result["predictive_covariance_normalized"]=V.tolist()
        results[i]=result
    return results


class BatchedPredictions:
    """Reuse the saved population in bounded chunks; no CPU inverse pool."""
    def __init__(self,root,output,fold,batch_size=4096,backend="cupy",device=0,devices=None):
        from .accommodation_full_accelerated import ParallelPredictions
        self.stream=iter(ParallelPredictions(root,output,fold,1).task_stream())
        self.batch_size,self.backend,self.device=batch_size,backend,device
        self.devices=devices
        self.ready=[]; self.expected=[]; self.offset=0
    def __call__(self,model,p,q,valid,held,pilot,reference,sigma,mask="xy"):
        if mask!="xy": raise ValueError("Full batched agreement requires xy")
        if self.offset==len(self.ready):
            from itertools import islice
            batch=list(islice(self.stream,self.batch_size))
            if not batch: raise RuntimeError("Request outside frozen schedule")
            ps,qs,vs,hs=zip(*batch)
            self.ready=predict_batch(model,ps,qs,vs,hs,pilot,reference,sigma,backend=self.backend,device=self.device,devices=self.devices)
            self.expected=batch; self.offset=0
        ep,eq,ev,eh=self.expected[self.offset]
        kept=np.arange(3)!=held
        if eh!=held or not np.array_equal(p,ep,equal_nan=True) or not np.array_equal(q[kept],eq[kept],equal_nan=True) or not np.array_equal(valid[kept],ev[kept]):
            raise RuntimeError("Inverse task order differs from frozen population")
        result=self.ready[self.offset]; self.offset+=1
        return result
    def close(self):
        self.ready=[]; self.expected=[]
