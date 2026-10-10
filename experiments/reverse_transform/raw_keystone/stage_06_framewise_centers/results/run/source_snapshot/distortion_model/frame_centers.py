"""Raw geometry and model-corrected optical origins for every frame.

One P1-profiled magnification is used for both patterns. Centers are derived
from each frame's measured centroid and transformed reference mean.
"""
import numpy as np


def frame_geometry(p1, p4, reference_p1, reference_p4, gaze, units, k1, k4,
                   accommodation, slope, reference_A):
    p1, p4 = np.asarray(p1), np.asarray(p4)
    t = np.asarray(gaze)/units
    outputs, means, domains = [], [], []
    for b, k, radial in ((reference_p1,k1,False),(reference_p4,k4,True)):
        b, k = np.asarray(b), np.asarray(k)
        r = np.sqrt(np.mean(np.sum(b*b,axis=1)))
        r2 = np.sum(b*b,axis=1)
        kap = slope*(np.asarray(accommodation)-reference_A) if radial else np.zeros(len(gaze))
        pre = b[None]*(1+kap[:,None]*r2)[...,None]
        exponent = k[:,0]*t[:,0]**2-k[:,1]*t[:,1]**2
        factors = np.stack((np.exp(exponent),np.exp(-exponent)),axis=1)[:,None]
        den = 1+(k[:,2,None]*t[:,0,None]*pre[...,1]+k[:,3,None]*t[:,1,None]*pre[...,0])/r
        f = pre*factors/den[...,None]
        outputs.append(f)
        means.append(f.mean(axis=1))
        domains.append((den > 1e-8).all(axis=1) & (1+3*kap[:,None]*r2 > 1e-8).all(axis=1)
                       & np.isfinite(f).all(axis=(1,2)))
    c1, c4 = p1.mean(axis=1), p4.mean(axis=1)
    h1, h4 = outputs[0]-means[0][:,None], outputs[1]-means[1][:,None]
    x1, x4 = p1-c1[:,None], p4-c4[:,None]
    scale = np.sum(x1*h1,axis=(1,2))/np.sum(h1*h1,axis=(1,2))
    center1, center4 = c1-scale[:,None]*means[0], c4-scale[:,None]*means[1]
    valid = domains[0] & domains[1] & (scale > 0) & np.isfinite(scale)
    return dict(magnification=scale,center_p1_xy=center1,center_p4_xy=center4,
        model_mu_p1_xy=means[0],model_mu_p4_xy=means[1],centroid_p1_xy=c1,centroid_p4_xy=c4,
        corrected_separation_xy=(center4-center1)/scale[:,None],
        center_correction_xy=scale[:,None]*(means[1]-means[0]),
        prediction_p1=scale[:,None,None]*h1,prediction_p4=scale[:,None,None]*h4,
        forward_residual_p1=x1-scale[:,None,None]*h1,forward_residual_p4=x4-scale[:,None,None]*h4,
        uncentered_p1=outputs[0],uncentered_p4=outputs[1],valid=valid)
