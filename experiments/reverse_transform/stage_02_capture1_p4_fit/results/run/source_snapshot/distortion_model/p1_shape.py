"""Fresh P1 shape fit: fixed empirical reference and one total magnification.

Keystone changes shape only after normalizing the model triangle's RMS radius.
Observed triangles are never area-normalized. Angles use visual degrees.
"""
import numpy as np


def centered(points):
    points = np.asarray(points, dtype=np.float64)
    return points - points.mean(axis=-2, keepdims=True)


def shape_model(theta_deg, reference, coefficients, derivatives=False, gaze_units=(10.,10.)):
    """Return a centered keystone triangle with the reference's RMS radius.

    coefficients=(eta_x,eta_y,beta_x,beta_y), t=theta/gaze_units:
    sx=exp(eta_x*t_x²-eta_y*t_y²), sy=1/sx,
    denominator=1+beta_x*t_x*B_y/R_B+beta_y*t_y*B_x/R_B.
    This extends the agreed horizontal denominator along B_y to vertical
    gaze along B_x. The size normalization is a
    deterministic shape convention, never a second fitted magnification.
    """
    t = np.asarray(theta_deg, dtype=float)/np.asarray(gaze_units)
    tx,ty=t[...,0],t[...,1]
    b = np.asarray(reference, dtype=float)
    eta_x,eta_y,beta_x,beta_y = coefficients
    radius = np.sqrt(np.mean(np.sum(b*b, axis=-1)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('nondegenerate reference required')
    stretch = np.exp(eta_x*tx*tx-eta_y*ty*ty)
    vx = tx[..., None]*b[:, 1]/radius
    vy = ty[..., None]*b[:, 0]/radius
    denominator = 1+beta_x*vx+beta_y*vy
    valid = np.all(denominator > 1e-8, axis=-1)
    factors = np.stack((stretch,1/stretch),axis=-1)[..., None, :]
    with np.errstate(divide='ignore',invalid='ignore'):
        f = factors*b/denominator[..., None]
        c = centered(f)
        size = np.sqrt(np.mean(np.sum(c*c,axis=-1),axis=-1))/radius
        h = c/size[..., None, None]
    valid &= np.isfinite(h).all(axis=(-2,-1)) & (size > 0)
    if not valid.all(): raise ValueError('keystone shape outside its domain')
    if not derivatives: return h, f.mean(axis=-2), size
    df_eta_x = f*np.stack((tx*tx,-tx*tx),axis=-1)[..., None, :]
    df_eta_y = f*np.stack((-ty*ty,ty*ty),axis=-1)[..., None, :]
    df_beta_x = -f*(vx/denominator)[..., None]
    df_beta_y = -f*(vy/denominator)[..., None]
    df = np.stack((df_eta_x,df_eta_y,df_beta_x,df_beta_y),axis=-1)
    dc = df-df.mean(axis=-3,keepdims=True)
    dlogsize = np.sum(c[..., None]*dc,axis=(-3,-2))/np.sum(c*c,axis=(-2,-1))[..., None]
    dh = dc/size[..., None,None,None]-h[..., None]*dlogsize[..., None,None,:]
    return h, f.mean(axis=-2), size, dh


def profile_magnification(observed_centered, predicted_shape):
    """Single positive scalar per frame minimizing original-coordinate SSE."""
    x, h = np.broadcast_arrays(observed_centered,predicted_shape)
    m = np.sum(x*h,axis=(-2,-1))/np.sum(h*h,axis=(-2,-1))
    if not np.isfinite(m).all() or not (m > 0).all():
        raise ValueError('nonpositive profiled magnification')
    return m


def profile_objective(coefficients, theta_deg, reference, observed, frame_weights,gaze_units=(10.,10.)):
    h,_,_,dh = shape_model(theta_deg,reference,coefficients,derivatives=True,gaze_units=gaze_units)
    m = profile_magnification(observed,h)
    residual = observed-m[..., None,None]*h
    cost = np.sum(frame_weights*np.sum(residual*residual,axis=(-2,-1)))
    gradient = -2*np.sum(frame_weights[..., None]*m[..., None]*
                         np.sum(residual[..., None]*dh,axis=(-3,-2)),axis=0)
    return float(cost),gradient


def recover_p1(observed, theta_deg, reference, coefficients, magnification,gaze_units=(10.,10.)):
    """Recover corresponding reference points using the fitted origin gauge.

    Keystone is nonlinear: add its predicted centroid before inversion.
    Do not directly invert a centroid-subtracted triangle.
    """
    theta_deg=np.asarray(theta_deg,dtype=float);magnification=np.asarray(magnification,dtype=float)
    h,mu,size = shape_model(theta_deg,reference,coefficients,gaze_units=gaze_units)
    z = centered(observed)*size[..., None,None]/magnification[..., None,None]+mu[..., None,:]
    radius = np.sqrt(np.mean(np.sum(reference*reference,axis=-1)))
    eta_x,eta_y,beta_x,beta_y = coefficients; t=theta_deg/np.asarray(gaze_units)
    logstretch=eta_x*t[...,0]**2-eta_y*t[...,1]**2
    sx,sy=np.exp(logstretch),np.exp(-logstretch)
    qx,qy=beta_y*t[...,1]/radius,beta_x*t[...,0]/radius
    w=z/np.stack((sx,sy),axis=-1)[...,None,:]
    denominator = 1-qx[...,None]*w[...,0]-qy[...,None]*w[...,1]
    valid = np.all(denominator>1e-8,axis=-1) & (magnification>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        result=w/denominator[...,None]
    valid &= np.isfinite(result).all(axis=(-2,-1))
    return np.where(valid[..., None,None],result,np.nan),valid
