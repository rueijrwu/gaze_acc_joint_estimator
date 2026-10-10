"""Generic numerical helpers for the raw-keystone pipeline.

Calibration/statistics are copied verbatim from the reviewed reverse-transform
stage01 runner; bounded fitting logic is copied verbatim from stage03. These
functions contain no optical-model implementation or saved coefficient loading.
"""
import numpy as np
from scipy.optimize import minimize


def radius(points):
    """Centroid-centered RMS radius, supporting batches of triangles."""
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=-2, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=-1), axis=-1))


def statistics(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    if not len(a):return {'count':0}
    return {'count':int(a.size),'mean':float(a.mean()),'std':float(a.std()),
            'min':float(a.min()),'p05':float(np.percentile(a,5)),'median':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)),
            'max':float(a.max()),'rms':float(np.sqrt(np.mean(a*a)))}

def calibrate_gaze(delta,exposure,targets):
    """Quadratic conversion with equal fixation weights and exact zero mean.

    Fit frame-mean polynomial moments to the five nominal targets, rather
    than pretending polynomial(mean x) equals mean(polynomial(x)).
    """
    zero=np.flatnonzero(np.asarray(targets)==0).item()
    delta_x=delta[:,0]
    means=np.array([delta_x[exposure==k].mean() for k in range(5)])
    origin=means[zero]; scale=(means.max()-means.min())/2
    if scale<=0:raise ValueError('centroid gaze signal is degenerate')
    u=(delta_x-origin)/scale
    moments=np.array([[u[exposure==k].mean(),(u[exposure==k]**2).mean()] for k in range(5)])
    design=moments-moments[zero]
    ab=np.linalg.lstsq(design,targets,rcond=None)[0]
    c=-moments[zero]@ab
    gaze=c+ab[0]*u+ab[1]*u*u
    coefficient_raw=[float(c-ab[0]*origin/scale+ab[1]*origin**2/scale**2),
                     float(ab[0]/scale-2*ab[1]*origin/scale**2),float(ab[1]/scale**2)]
    fitted_means=np.array([gaze[exposure==k].mean() for k in range(5)])
    derivative=(ab[0]+2*ab[1]*u)/scale
    if np.min(derivative)*np.max(derivative)<=0:raise ValueError('nonmonotone gaze polynomial over observed frames')
    # User assumption: equal x/y first-order slope at the zero-gaze reference.
    slope=ab[0]/scale
    origin_y=delta[exposure==zero,1].mean()
    gaze_y=slope*(delta[:,1]-origin_y)
    return np.stack((gaze,gaze_y),axis=1),{'degree':2,'input':'mean(P4)-mean(P1), native px',
        'delta_x_origin_px':float(origin),'delta_x_unit_px':float(scale),
        'coefficients_normalized_ascending':[float(c),float(ab[0]),float(ab[1])],
        'coefficients_native_ascending_deg':[float(v) for v in coefficient_raw],
        'fixation_mean_delta_x_px':means.tolist(),'nominal_targets_deg':list(map(float,targets)),
        'fixation_mean_estimated_gaze_deg':fitted_means.tolist(),
        'fixation_mean_calibration_error_deg':(fitted_means-targets).tolist(),
        'derivative_deg_per_px_range':[float(derivative.min()),float(derivative.max())],
        'shared_first_order_slope_deg_per_px':float(slope),'delta_y_origin_px':float(origin_y),
        'vertical_mapping':'theta_y=shared_reference_slope*(delta_y-zero_fixation_mean_delta_y)',
        'fixation_mean_estimated_vertical_gaze_deg':[float(gaze_y[exposure==k].mean()) for k in range(5)],
        'zero_reference_exposure':int(zero),'scope':'fixation-label calibration, not framewise ground truth'}


def limits(gaze,b,units,barrel):
    extent=np.max(abs(gaze/units),axis=0);r=radius(b)
    widths=list(np.r_[.5/np.maximum(extent**2,1e-12),
                     .4/np.maximum(1.25*extent*np.max(abs(b[:,[1,0]]),axis=0)/r,1e-12)])
    if barrel: widths.append(.25/np.max(np.sum(b*b,axis=1)/r**2))
    return np.asarray(widths)

def fitting(model,widths):
    n=len(widths);fractions=np.zeros((6,n))
    fractions[1,:4]=[.1,.1,0,0];fractions[2,:4]=[-.1,-.1,0,0]
    fractions[3,:4]=[0,0,.1,.1];fractions[4,:4]=[0,0,-.1,-.1]
    fractions[5,:4]=[.1,-.1,.1,-.1]
    if n==5: fractions[1:,4]=[.1,-.1,.1,-.1,0]
    starts=[]
    for start in fractions*widths:
        sol=minimize(model.evaluate,start,jac=True,method='L-BFGS-B',bounds=list(zip(-widths,widths)),
                     options={'maxiter':400,'ftol':1e-14,'gtol':1e-9,'maxls':50})
        starts.append({'coefficients':sol.x.tolist(),'objective_point_px2':float(sol.fun),
                       'success':bool(sol.success),'message':str(sol.message),'iterations':int(sol.nit)})
    best=min(starts,key=lambda s:s['objective_point_px2']);k=np.array(best['coefficients'])
    value,gradient=model.evaluate(k);polish=[]
    for _ in range(3):
        constrained=((k<=-widths+1e-7)&(gradient>0))|((k>=widths-1e-7)&(gradient<0))
        projected=gradient.copy();projected[constrained]=0;free=np.flatnonzero(~constrained)
        if max(abs(projected))<1e-7 or not len(free):break
        hessian=np.column_stack([(model.evaluate(k+np.eye(n)[j]*1e-5)[1]-
                                  model.evaluate(k-np.eye(n)[j]*1e-5)[1])/(2e-5) for j in range(n)])
        hessian=(hessian+hessian.T)/2;hf=hessian[np.ix_(free,free)]
        if np.linalg.eigvalsh(hf).min()<=0:break
        step=np.zeros(n);step[free]=np.linalg.solve(hf,-gradient[free]);accepted=False
        for power in range(20):
            proposal=k+step*.5**power
            if np.any(abs(proposal)>widths):continue
            pv,pg=model.evaluate(proposal);pp=pg.copy()
            pp[((proposal<=-widths+1e-7)&(pg>0))|((proposal>=widths-1e-7)&(pg<0))]=0
            if pv<=value+1e-10 and max(abs(pp))<max(abs(projected)):
                polish.append({'objective_before':value,'objective_after':pv})
                k,value,gradient=proposal,pv,pg;accepted=True;break
        if not accepted:break
    hessian=np.column_stack([(model.evaluate(k+np.eye(n)[j]*1e-5)[1]-
                              model.evaluate(k-np.eye(n)[j]*1e-5)[1])/(2e-5) for j in range(n)])
    active=abs(abs(k)-widths)<1e-7;free=np.flatnonzero(~active)
    eigen=np.linalg.eigvalsh(((hessian+hessian.T)/2)[np.ix_(free,free)])
    projected=gradient.copy()
    projected[((k<=-widths+1e-7)&(gradient>0))|((k>=widths-1e-7)&(gradient<0))]=0
    checks={'projected_gradient_inf':float(max(abs(projected))), 'active_bounds':active.tolist(),
            'free_hessian_eigenvalues':eigen.tolist(), 'domain':model.domain(k),
            'multistart_objective_range':float(max(s['objective_point_px2'] for s in starts)-min(s['objective_point_px2'] for s in starts)),
            'stationary':bool(max(abs(projected))<1e-6 and eigen.min()>0)}
    return k,{'coefficients_scaled':k.tolist(),'bounds_scaled':list(zip(-widths,widths)),
              'objective_point_px2':value,'checks':checks,'starts':starts,'numerical_polish':polish}


def fit_accommodation(model,lower,upper,start):
    sol=minimize(model.evaluate,np.clip(start,lower,upper),jac=True,method='L-BFGS-B',
                 bounds=list(zip(lower,upper)),options={'maxiter':600,'ftol':1e-15,'gtol':1e-7,'maxls':40,'maxcor':10})
    a=sol.x.copy();value,gradient=model.evaluate(a);polish=[]
    # Exact diagonal-plus-one-mean-per-fixation Newton system. No dense Hessian.
    exposure=model.host(model.e);counts=model.host(model.count);q=model.host(model.q)
    for iteration in range(8):
        cert,curvature=model.certificate(a,lower,upper)
        if cert['projected_gradient_inf']<1e-7:break
        constrained=((a<=lower+1e-7)&(gradient>0))|((a>=upper-1e-7)&(gradient<0))
        free=~constrained
        if np.any(curvature[free]<=0) or not np.isfinite(curvature[free]).all():break
        inverse=np.where(free,1/np.maximum(curvature,1e-12),0.)
        local_gradient=gradient/q
        sums=np.bincount(exposure,weights=inverse,minlength=len(counts))
        terms=np.bincount(exposure,weights=local_gradient*inverse,minlength=len(counts))
        rank=2*model.strength/counts
        correction=rank*terms/(1+rank*sums)
        step=(-local_gradient+correction[exposure])*inverse
        accepted=False
        for power in range(25):
            proposal=np.clip(a+step*.5**power,lower,upper)
            new_value,new_grad=model.evaluate(proposal)
            projected=new_grad.copy()
            projected[((proposal<=lower+1e-7)&(new_grad>0))|((proposal>=upper-1e-7)&(new_grad<0))]=0
            if new_value<=value+1e-7 and np.max(abs(projected))<cert['projected_gradient_inf']:
                polish.append({'iteration':iteration,'objective_before':value,'objective_after':new_value})
                a,value,gradient=proposal,new_value,new_grad;accepted=True;break
        if not accepted:break
    cert,_=model.certificate(a,lower,upper)
    return a,{'objective_scaled_sum':value,'objective_equal_fixation_mean_px2':value/model.n,
              'optimizer_success':bool(sol.success),'optimizer_message':str(sol.message),'iterations':int(sol.nit),
              'certificate':cert,'numerical_polish':polish}
