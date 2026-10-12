"""Fresh barrel Stage03 -> keystone Stage04 -> joint Stage05 science.

Only Stage01 gaze and Stage02 P1 are external inputs. No legacy P4 calibration
or optimum is referenced. Geometry is D(K(B)), inverse barrel before K.
"""
import numpy as np
from .capturewise_reverse import forward as keystone_only,host,equal_group_weights
from .capturewise_keystone_first import forward,inverse
from .capturewise_barrel_anchors import solve_small,KeystoneFirstStateProblem
from .capturewise_states import A_REF,DEMAND
from .p1_joint import template


def accommodation_targets(kappa):
    kappa=np.asarray(kappa,dtype=float)
    if kappa.shape!=(4,5):raise ValueError('Fresh20 relative radial coefficients required')
    demand=np.broadcast_to((DEMAND-A_REF)[:,None],kappa.shape)
    slope=float(np.sum(demand*kappa)/np.sum(demand**2))
    if not np.isfinite(slope) or abs(slope)<1e-12:raise ValueError('Uninformative fresh Stage03 radial slope')
    targets=A_REF+kappa/slope
    if not np.isfinite(targets).all() or np.any(targets<0)|np.any(targets>6):raise ValueError('Fresh20 Amean targets outside declared state domain; no clipping/substitution')
    return slope,targets


class KeystoneCalibration:
    """K-only corrected-data initializer then full original-pixel fit."""
    def __init__(self,data,g,barrel_recovered,kappa_targets,xp=np):
        self.xp=xp;self.data=data;self.n=len(g);c=np.asarray(data['capture'],int)-1;f=np.asarray(data['fixation'],int)
        self.capture=c;self.c=xp.asarray(c);self.theta=xp.asarray(data['theta0']);self.g=xp.asarray(g);self.X=xp.asarray(data['X4']);self.Y=np.asarray(barrel_recovered)
        self.corrected=xp.asarray(self.Y-self.Y.mean(axis=1,keepdims=True));self.kappa=xp.asarray(np.asarray(kappa_targets)[c,f]);self.w=xp.asarray(equal_group_weights(data['group'],data['capture']))
        self.corrected_zero_means=np.array([self.Y[(c==j)&(f==2)].mean(axis=0) for j in range(4)])
        self.B_initial=self.corrected_zero_means[0].copy();self.centroid=self.B_initial.mean(axis=0);self.centered=self.B_initial-self.centroid;self.energy=float(np.sum(self.centered**2))
        self.R4=np.sqrt(np.mean(np.sum(self.corrected_zero_means**2,axis=2),axis=1));self.R=xp.asarray(self.R4)[self.c]
        if not np.isfinite(self.R4).all() or np.any(self.R4<=0):raise ValueError('Nondegenerate fresh corrected native radii required')
        self.p0=np.zeros(16);self.scales=np.r_[np.full(12,.05),10.,np.full(3,.1)]
        self.bounds=np.array([[-.25,.25],[-.2,.2],[-.2,.2]]*4+[[-120.,120.]]+[[-.75,.75]]*3)
        self.rejections=0

    def geometry(self,p):
        B,D=template(self.centered,p[13:],True);singular=np.linalg.svd(B,compute_uv=False)
        return B+self.centroid,D,bool(singular[-1]>singular[0]*1e-10 and np.isfinite(B).all())

    def evaluate(self,p,original_pixels=True,details=False):
        xp=self.xp;B,dB,valid=self.geometry(p);a=xp.asarray(p[:12].reshape(4,3))[self.c]
        if original_pixels:ev=forward(B,self.theta,a,p[12],self.R,self.g,self.kappa,xp,True);obs=self.X;gg=self.g
        else:ev=keystone_only(B,self.theta,a,p[12],self.R,1.,xp=xp,derivatives=True);obs=self.corrected;gg=xp.ones(self.n)
        ev['valid'] &= valid
        if not bool(xp.all(ev['valid'])):
            self.rejections+=1
            return dict(loss=float('inf'),valid=host(ev['valid'],xp)) if details else (float('inf'),np.zeros(16))
        residual=ev['prediction']-obs;loss=float(host(xp.einsum('n,nja,nja->',self.w,residual,residual),xp))
        row=2*xp.einsum('n,nja,njap->np',self.w,residual,ev['dalpha'])
        grad=np.zeros(16)
        for j in range(4):grad[3*j:3*j+3]=host(row[self.c==j].sum(axis=0),xp)
        grad[12]=float(host(-2*xp.einsum('n,nja,nja->',self.w,residual,ev['dtheta']),xp))
        raw=xp.einsum('njab,jbp->njap',ev['point_jacobian'],xp.asarray(dB));centered=gg[:,None,None,None]*(raw-raw.mean(axis=1,keepdims=True))
        grad[13:]=host(2*xp.einsum('n,nja,njap->p',self.w,residual,centered),xp)
        if details:return dict(loss=loss,prediction=host(ev['prediction'],xp),mu=host(ev['mu'],xp),valid=host(ev['valid'],xp),template=B,denominator=host(ev['denominator'],xp),radial_derivative=host(ev['radial_derivative'],xp),gradient=grad)
        return loss,grad

    def fit(self,starts=3,maxiter=700,seed=20261025):
        seedbest,seedstarts=solve_small(lambda p:self.evaluate(p,False),self.p0,self.scales,self.bounds,starts,maxiter,seed)
        best,allstarts=solve_small(self.evaluate,seedbest['params'],self.scales,self.bounds,starts,maxiter,seed+1)
        detail=self.evaluate(best['params'],True,True)
        return dict(best=best,starts=allstarts,corrected_initializer_best=seedbest,corrected_initializer_starts=seedstarts,**detail)


class BarrelFirstStateProblem(KeystoneFirstStateProblem):
    """Fresh Stage04 K/reference, fresh Stage03 slope and20 Amean targets."""
    def __init__(self,data,p1,p4,barrel,*,xp=np,g_override=None,theta_anchor_override=None,vertices4=None,sigma_A=.25):
        k=np.asarray(barrel['kappa_targets']);slope,targets=accommodation_targets(k)
        # Base constructor gets entirely fresh inputs. Mean capture targets are
        # needed only for its initialization; actual anchors are all20 cells.
        p3=dict(alpha=p4['alpha'],offset=p4['offset'],R4=p4['R4'])
        radial=dict(B4_ref=p4['B4_ref'],kappa=k.mean(axis=1))
        super().__init__(data,p1,p3,radial,xp=xp,g_override=g_override,theta_anchor_override=theta_anchor_override,vertices4=vertices4,sigma_A=sigma_A,targets=targets.mean(axis=1),learned_template=True,kappa_targets=k,barrel_anchor=False)
        if abs(self.s0-slope)>1e-12*abs(slope):raise ValueError('Fresh equalcapture/gaze slope convention mismatch')
        self.fixation_targets=targets
        self.targets=xp.asarray([targets[self.capture[np.flatnonzero(self.group_np==i)[0]],int(data['fixation'][np.flatnonzero(self.group_np==i)[0]])] for i in range(self.G)])
