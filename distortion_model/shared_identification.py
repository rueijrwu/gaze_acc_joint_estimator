"""Post-fit data-versus-prior information audit for frozen shared S0 fits.

This module does not change a fit. All matrices are Gauss Newton information
for normalized J=N*L, rather than physical measurement uncertainty or the exact
residual Hessian. States are profiled as nuisance parameters at the fitted point.
"""
from hashlib import sha256
from pathlib import Path
import numpy as np


def _spectrum(matrix):
    matrix=np.asarray(matrix,dtype=float)
    matrix=(matrix+matrix.T)/2
    values,vectors=np.linalg.eigh(matrix)
    scale=max(1.,float(np.linalg.norm(matrix,ord=2)))
    tolerance=1e-10*scale
    positive=values>tolerance
    return dict(eigenvalues=values.tolist(),eigenvectors_columns=vectors.tolist(),
        matrix=matrix.tolist(),numerical_zero_tolerance=tolerance,
        positive_rank=int(np.sum(positive)),negative_count=int(np.sum(values < -tolerance)),
        positive_condition_number=float(values[positive].max()/values[positive].min()) if positive.any() else None)


def _state_spectrum(H,xp,host,tolerance):
    eig,vec=xp.linalg.eigh((H+H.transpose(0,2,1))/2)
    thresholds=xp.maximum(xp.max(xp.abs(eig),axis=1),1e-30)*tolerance
    positive=eig>thresholds[:,None]
    rank=xp.sum(positive,axis=1)
    inverse=xp.where(positive,1/xp.maximum(eig,1e-300),0.)
    pinv=xp.einsum('nij,nj,nkj->nik',vec,inverse,vec)
    ranks=host(rank)
    spectrum=host(eig)
    report=dict(rank_counts={str(k):int(np.sum(ranks==k)) for k in range(4)},
        relative_eigenvalue_threshold=tolerance,
        smallest_eigenvalue_quantiles=np.quantile(spectrum[:,0],[0,.05,.5,.95,1]).tolist(),
        largest_eigenvalue_quantiles=np.quantile(spectrum[:,-1],[0,.05,.5,.95,1]).tolist())
    return pinv,report,eig


def diagnostics(model,q,states,relative_state_tolerance=1e-10):
    """Compare data-only and anchored information and check all known gauges.

    Global profile matrices include all frame states and the full dynamic
    fixation-mean accommodation prior. Box/domain restrictions do not contribute
    independent measurement information and are excluded from these matrices.
    """
    xp=model.xp;host=model.host
    q=xp.asarray(q,dtype=xp.float64);states=xp.asarray(states,dtype=xp.float64)
    ev=model.evaluate(q,states,hessians='joint')
    if not ev['valid']:
        return dict(status='INVALID_FORWARD_DOMAIN',margins=ev['margins'])
    scale_s=model.state_scale;scale_q=model.q_scale
    augmented_D=ev['state_hessian']*model.n*scale_s[None,:,None]*scale_s[None,None,:]
    cross=ev['state_q_hessian']*model.n*scale_s[None,:,None]*scale_q[None,None,:]
    augmented_Q=ev['q_hessian']*model.n*scale_q[:,None]*scale_q[None,:]
    data_D=augmented_D.copy();data_Q=augmented_Q.copy()
    for k in range(2):
        data_D[:,k,k]-=2*model.n*model.weights*float(scale_s[k])**2/model.sigma_theta[k]**2
    if model.optical_prior:
        data_Q-=xp.eye(9)*(2*model.n)
    data_inverse,state_report,_=_state_spectrum(data_D,xp,host,relative_state_tolerance)
    inverse_cross=xp.einsum('nij,njk->nik',data_inverse,cross)
    data_profile=data_Q-xp.einsum('nsi,nsj->ij',cross,inverse_cross)
    data_profile=(data_profile+data_profile.T)/2
    data_Qh=host(data_Q);data_profile_h=host(data_profile)
    report=dict(status='COMPLETED',method='normalized J=N*L Gauss Newton information',
        normalization='theta/.5 deg, A/.25 D, q/sigma_q',complete_frames=model.n,
        fixation_count=model.nfix,state_data_information=state_report,
        unprofiled_global_data=_spectrum(data_Qh),profiled_global_data=_spectrum(data_profile_h),
        constraint_information='Operational box/nonlinear restrictions excluded; anchors are not independent measurements',
        limitation='Local linearized information and quadratic profiles; not nonlinear likelihood confidence intervals')
    # Known data-only scaling directions, expressed in physical coordinates and
    # then translated into the very same normalized fit basis.
    zeros=xp.zeros_like(states);qzero=xp.zeros(9,dtype=xp.float64)
    directions=[]
    ds=zeros.copy();dq=qzero.copy();ds[:,2]=states[:,2]-model.A_ref;dq[8]=-q[8]
    directions.append(('accommodation_scale',ds,dq))
    ds=zeros.copy();dq=qzero.copy();ds[:,0]=states[:,0]
    dq[[0,4]]=-2*q[[0,4]];dq[[2,6]]=-q[[2,6]]
    directions.append(('horizontal_gaze_scale',ds,dq))
    ds=zeros.copy();dq=qzero.copy();ds[:,1]=states[:,1]
    dq[[1,5]]=-2*q[[1,5]];dq[[3,7]]=-q[[3,7]]
    directions.append(('vertical_gaze_scale',ds,dq))
    j1,j4=ev['geometry']['jacobian1'],ev['geometry']['jacobian4']
    retained_count=j4.shape[1]
    global_gauges=[];gauge_reports=[]
    for name,ds,dq in directions:
        shift1=xp.einsum('njcs,ns->njc',j1[...,:3],ds)+xp.einsum('njcq,q->njc',j1[...,3:],dq)
        shift4=xp.einsum('njcs,ns->njc',j4[...,:3],ds)+xp.einsum('njcq,q->njc',j4[...,3:],dq)
        cost=2*model.n*xp.sum(model.weights*(xp.sum(shift1*shift1,axis=(1,2))/3+
                            xp.sum(shift4*shift4,axis=(1,2))/retained_count))
        global_vector=host(dq/scale_q)
        global_gauges.append(global_vector)
        normalized_ds=ds/scale_s
        augmented_cost=(xp.sum(normalized_ds*xp.einsum('nij,nj->ni',augmented_D,normalized_ds))+
             2*xp.sum(normalized_ds*xp.einsum('nsi,i->ns',cross,dq/scale_q))+
             xp.dot(dq/scale_q,augmented_Q@(dq/scale_q)))
        # Exact dynamic mean-A rank-one anchor contribution.
        anchor_mean=model.means(ds[:,2])/model.sigma_A
        augmented_cost+=2*model.n*xp.mean(anchor_mean*anchor_mean)
        gauge_reports.append(dict(name=name,global_direction_normalized=global_vector.tolist(),
             state_direction_normalized_rms=host(xp.sqrt(xp.mean(normalized_ds**2,axis=0))).tolist(),
             prediction_derivative_max_abs_px=max(float(host(xp.max(xp.abs(shift1)))),float(host(xp.max(xp.abs(shift4))))),
             data_directional_information=float(host(cost)),
             augmented_directional_information=float(host(augmented_cost)),
             data_profile_global_residual_norm=float(np.linalg.norm(data_profile_h@global_vector))))
    report['known_gauges']=gauge_reports
    # Remove only the declared gauge span for the nongauge profile comparison.
    gauge_matrix=np.column_stack(global_gauges)
    U,singular,_=np.linalg.svd(gauge_matrix,full_matrices=True)
    tolerance=1e-10*max(1.,float(singular.max()))
    rank=int(np.sum(singular>tolerance));nongauge=U[:,rank:]
    report['known_global_gauge_rank']=rank
    report['nongauge_global_basis_columns']=nongauge.tolist()
    report['nongauge_profiled_global_data']=_spectrum(nongauge.T@data_profile_h@nongauge)
    # Each quadratic profile represents exact minimization over the linearized
    # frame nuisance coordinates, without interpreting priors as data evidence.
    def profiles(matrix):
        values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
        amplitudes=[-.1,-.05,0.,.05,.1]
        return [dict(eigenvalue=float(value),direction_normalized=vectors[:,k].tolist(),
             amplitudes=amplitudes,delta_J_quadratic=[float(.5*value*a*a) for a in amplitudes])
             for k,value in enumerate(values)]
    report['linearized_global_tradeoff_profiles_data']=profiles(data_profile_h)
    # Augmented inversion uses positive state blocks and the exact rank-one A
    # means. No jitter is introduced to manufacture positive information.
    _,augmented_report,augmented_eigen=_state_spectrum(augmented_D,xp,host,relative_state_tolerance)
    report['state_augmented_block_information']=augmented_report
    if bool(xp.all(augmented_eigen[:,0]>0)):
        all_free=xp.ones((model.n,3),dtype=bool)
        augmented_inverse_cross=model._mean_coupled_solve(augmented_D,cross,all_free)
        augmented_profile=augmented_Q-xp.einsum('nsi,nsj->ij',cross,augmented_inverse_cross)
        augmented_profile_h=host((augmented_profile+augmented_profile.T)/2)
        report['profiled_global_augmented']=_spectrum(augmented_profile_h)
        report['nongauge_profiled_global_augmented']=_spectrum(nongauge.T@augmented_profile_h@nongauge)
        report['linearized_global_tradeoff_profiles_augmented']=profiles(augmented_profile_h)
    else:
        report['profiled_global_augmented']=dict(status='UNRESOLVED_SINGULAR_STATE_BLOCKS',
            reason='No regularized inversion substituted for an unidentified state block')
    report['diagnostic_source_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    report['fitting_source_sha256']=sha256(Path(__file__).with_name('shared_keystone.py').read_bytes()).hexdigest()
    return report
