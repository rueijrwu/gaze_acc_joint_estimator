"""One float64 DM0 objective, bounded dynamic states and coupled GN/Newton.

CuPy and NumPy execute the same equations. Per-frame Jacobians are chunked;
only 2x2 state blocks, state/global cross blocks and a small global system
are retained. Full fixation means remain coupled through a Woodbury solve.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space
from .optics import Parameters


@dataclass(frozen=True)
class JointSpec:
    b1: np.ndarray
    b4: np.ndarray
    omega1: float
    omega4: float
    aref: float
    theta_bounds: tuple = (-20.,20.)
    a_bounds: tuple = (0.,6.)
    template_sigma: float = .10

    def __post_init__(self):
        for name in ('b1','b4'):
            value=np.array(getattr(self,name),dtype=float,copy=True)
            if value.shape!=(3,2) or not np.isfinite(value).all():raise ValueError('finite native templates required')
            if np.linalg.norm(value.mean(axis=0))>1e-8:raise ValueError('fixed centroid origin required')
            value.setflags(write=False);object.__setattr__(self,name,value)
        if not np.isfinite([self.omega1,self.omega4,self.aref,self.template_sigma]).all() or self.template_sigma<=0:
            raise ValueError('finite references and positive template prior scale required')

    @property
    def r1(self):return float(np.sqrt(np.mean(np.sum(self.b1**2,axis=1))))
    @property
    def r4(self):return float(np.sqrt(np.mean(np.sum(self.b4**2,axis=1))))
    @property
    def template_basis(self):
        constraints=np.vstack((np.tile([1.,0.],3),np.tile([0.,1.],3),self.b4.ravel()))
        return null_space(constraints).reshape(3,2,3)

    def bounds(self):
        x1=max(abs(np.asarray(self.theta_bounds)-self.omega1))/10
        x4=max(abs(np.asarray(self.theta_bounds)-self.omega4))/10
        y1=max(abs(self.b1[:,1]))/self.r1
        # Conservative y envelope over the declared constrained template box.
        y4=(max(abs(self.b4[:,1]))+np.sqrt(3)*self.r4*np.sqrt(3)*.10)/self.r4
        amplitude=.5/max(abs(np.asarray(self.a_bounds)-self.aref))
        extent=np.r_[.5/x1**2,.5/(x1*y1),10*amplitude,[.10]*3,
                     .5/x4**2,.5/x4**2,.5/(x4*1.5*y4),[np.inf]*10]
        return -extent,extent

    def pack(self,record):
        k1,k4=record['k1_native'],record['k4_native']
        if abs(k1[0]+k1[1])>1e-12:raise ValueError('P1 trace-free gauge required')
        return np.r_[100*k1[0],10*self.r1*k1[2],10*record['m1_per_D'],[0.]*3,
                     100*k4[0],100*k4[1],10*self.r4*k4[2],
                     np.asarray(record['center_coefficients_reference_px']).ravel()/self.r4]

    def parameters(self,p):
        p=np.asarray(p,dtype=float)
        u=p[3:6];b=(self.b4+np.sqrt(3)*self.r4*np.einsum('ijp,p->ij',self.template_basis,u))/np.sqrt(1+u@u)
        return Parameters(self.b1,b,self.omega1,self.omega4,(p[0]/100,-p[0]/100,p[1]/(10*self.r1)),
                          (p[6]/100,p[7]/100,p[8]/(10*self.r4)),self.aref,p[2]/10,
                          np.vstack((p[9:].reshape(5,2)*self.r4,np.zeros((1,2)))),self.theta_bounds,self.a_bounds)


class JointDM0:
    def __init__(self,spec,observed,exposure,covariance,targets,xp=np,chunk_size=32768):
        self.spec,self.xp,self.chunk_size=spec,xp,int(chunk_size)
        y=np.asarray(observed,dtype=float);e=np.asarray(exposure,dtype=int)
        if y.shape!=(len(e),10) or not np.isfinite(y).all() or len(e)==0:
            raise ValueError('finite complete relative observations required')
        groups,inverse,counts=np.unique(e,return_inverse=True,return_counts=True)
        if not np.array_equal(inverse,np.sort(inverse)):
            raise ValueError('exposure-contiguous rows required for exact group reduction')
        r=np.asarray(covariance,dtype=float);np.linalg.cholesky(r)
        if r.shape!=(10,10) or not np.allclose(r,r.T,rtol=1e-12,atol=1e-12):raise ValueError('full symmetric raw metric required')
        target=np.asarray(targets,dtype=float)
        if target.shape!=(len(groups),2) or not np.isfinite(target).all():raise ValueError('one mean target per full exposure required')
        self.n,self.k=len(e),len(counts)
        self.y=xp.asarray(y);self.exposure=xp.asarray(inverse);self.counts=xp.asarray(counts)
        self.ends=xp.asarray(np.cumsum(counts)-1);self.target=xp.asarray(target)
        self.weights=1/(self.k*self.counts[self.exposure])
        self.precision=xp.asarray(np.linalg.solve(r,np.eye(10)))
        self.r11inv=xp.asarray(np.linalg.solve(r[:4,:4],np.eye(4)))
        self.edge_weighted=self.y[:,:4]@self.r11inv
        self.b1=xp.asarray(spec.b1);self.b4=xp.asarray(spec.b4);self.template_basis=xp.asarray(spec.template_basis)
        lo,hi=spec.bounds();self.lower,self.upper=xp.asarray(lo),xp.asarray(hi)
        self.global_scales=xp.asarray(np.where(np.isfinite(hi),hi-lo,1.))
        self.state_lower=xp.asarray([spec.theta_bounds[0],spec.a_bounds[0]])
        self.state_upper=xp.asarray([spec.theta_bounds[1],spec.a_bounds[1]])
        self.state_scales=self.state_upper-self.state_lower
        self.mean_scales=xp.asarray([.10,.25])
        self.prior=xp.zeros(19);self.prior[3:6]=1/spec.template_sigma**2;self.prior[17:19]=1.
        self.evaluations=0

    def host(self,value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def scalar(self,value):return float(value)

    def group_sum(self,value):
        xp=self.xp;prefix=xp.cumsum(value,axis=0)
        ends=prefix[self.ends]
        return ends-xp.concatenate((xp.zeros_like(ends[:1]),ends[:-1]),axis=0)

    def batch(self,x,p,sl,global_derivatives=True):
        xp,s=self.xp,self.spec
        theta,a=x[:,0],x[:,1];n=len(x)
        xi1,xi4=theta-s.omega1,theta-s.omega4
        alpha1,beta1,gamma1=p[0]/100,-p[0]/100,p[1]/(10*s.r1)
        d1=1+gamma1*xi1[:,None]*self.b1[:,1]
        factors1=xp.stack((1+alpha1*xi1**2,1+beta1*xi1**2),axis=1)
        f1=self.b1[None]*factors1[:,None,:]/d1[:,:,None]
        df1t=2*xi1[:,None,None]*self.b1[None]*xp.stack((alpha1,beta1))[None,None,:]/d1[:,:,None]-f1*(gamma1*self.b1[:,1]/d1)[:,:,None]
        mu1,mu1t=f1.mean(axis=1),df1t.mean(axis=1)
        edge=(f1[:,1:]-f1[:,:1]).reshape(n,4)
        det=(df1t[:,1:]-df1t[:,:1]).reshape(n,4)
        wa=edge@self.r11inv;energy=xp.sum(edge*wa,axis=1)
        g=xp.sum(edge*self.edge_weighted[sl],axis=1)/energy
        profile=(self.edge_weighted[sl]-2*g[:,None]*wa)/energy[:,None]
        dgt=xp.sum(det*profile,axis=1)
        u=p[3:6];norm=xp.sqrt(1+u@u)
        b4=(self.b4+np.sqrt(3)*s.r4*xp.einsum('ijp,p->ij',self.template_basis,u))/norm
        m1=p[2]/10;M=1+m1*(a-s.aref);baseline=M[:,None,None]*b4[None]
        alpha4,beta4,gamma4=p[6]/100,p[7]/100,p[8]/(10*s.r4)
        d4=1+gamma4*xi4[:,None]*baseline[:,:,1]
        factors4=xp.stack((1+alpha4*xi4**2,1+beta4*xi4**2),axis=1)
        f4=baseline*factors4[:,None,:]/d4[:,:,None]
        def baseline_derivative(dz):
            return factors4[:,None,:,None]*dz/d4[:,:,None,None]-f4[:,:,:,None]*(gamma4*xi4[:,None]/d4)[:,:,None,None]*dz[:,:,1:2,:]
        df4t=2*xi4[:,None,None]*baseline*xp.stack((alpha4,beta4))[None,None,:]/d4[:,:,None]-f4*(gamma4*baseline[:,:,1]/d4)[:,:,None]
        df4a=baseline_derivative(xp.broadcast_to(m1*b4[None,:,:,None],(n,3,2,1)))[...,0]
        t,ar=theta/10,a-s.aref
        phi=xp.stack((xp.ones_like(t),ar,t,ar*t,t*t),axis=1)
        center=p[9:].reshape(5,2)*s.r4
        d=phi@center
        dt=(center[2]+ar[:,None]*center[3]+2*t[:,None]*center[4])/10
        da=center[1]+t[:,None]*center[3]
        f=xp.concatenate((edge,(d[:,None,:]+f4-mu1[:,None,:]).reshape(n,6)),axis=1)
        prediction=g[:,None]*f
        js=xp.empty((n,10,2))
        js[:,:,0]=dgt[:,None]*f+g[:,None]*xp.concatenate((det,(dt[:,None,:]+df4t-mu1t[:,None,:]).reshape(n,6)),axis=1)
        js[:,:,1]=g[:,None]*xp.concatenate((xp.zeros((n,4)),(da[:,None,:]+df4a).reshape(n,6)),axis=1)
        jp=None
        if global_derivatives:
            jp=xp.zeros((n,10,19))
            df1p=xp.empty((n,3,2,2))
            df1p[:,:,:,0]=xi1[:,None,None]**2*self.b1[None]*xp.asarray([1.,-1.])[None,None,:]/d1[:,:,None]/100
            df1p[:,:,:,1]=-f1*(xi1[:,None]*self.b1[:,1]/d1)[:,:,None]/(10*s.r1)
            dep=(df1p[:,1:]-df1p[:,:1]).reshape(n,4,2)
            dgp=xp.einsum('nip,ni->np',dep,profile)
            jp[:,:,:2]=f[:,:,None]*dgp[:,None,:]+g[:,None,None]*xp.concatenate((dep,xp.broadcast_to(-df1p.mean(axis=1)[:,None,:,:],(n,3,2,2)).reshape(n,6,2)),axis=1)
            dm=(ar[:,None,None]*b4[None]/10)[...,None]
            jp[:,4:,2]=g[:,None]*baseline_derivative(dm)[...,0].reshape(n,6)
            db=np.sqrt(3)*s.r4*self.template_basis/norm-b4[:,:,None]*u[None,None,:]/norm**2
            jp[:,4:,3:6]=g[:,None,None]*baseline_derivative(M[:,None,None,None]*db[None]).reshape(n,6,3)
            df4k=xp.zeros((n,3,2,3))
            df4k[:,:,0,0]=xi4[:,None]**2*baseline[:,:,0]/d4/100
            df4k[:,:,1,1]=xi4[:,None]**2*baseline[:,:,1]/d4/100
            df4k[:,:,:,2]=-f4*(xi4[:,None]*baseline[:,:,1]/d4)[:,:,None]/(10*s.r4)
            jp[:,4:,6:9]=g[:,None,None]*df4k.reshape(n,6,3)
            dc=(phi[:,:,None,None]*xp.eye(2)[None,None,:,:]*s.r4).transpose(0,2,1,3).reshape(n,2,10)
            jp[:,4:,9:]=g[:,None,None]*xp.tile(dc,(1,3,1))
        valid=(xp.all(d1>1e-8,axis=1)&xp.all(d4>1e-8,axis=1)&xp.all(factors1>0,axis=1)&xp.all(factors4>0,axis=1)&(M>0)&(g>0)&xp.isfinite(prediction).all(axis=1))
        return prediction,js,jp,valid,{'g':g,'F1':f1,'F4':f4,'mu1':mu1,'mu4':f4.mean(axis=1),'M':M,'D':d}

    def evaluate(self,x,p,hessian=False,global_gradient=True):
        xp=self.xp;self.evaluations+=1
        gx=xp.zeros((self.n,2));gp=xp.zeros(19);point=xp.asarray(0.)
        if hessian:
            hx=xp.empty((self.n,2,2));cross=xp.empty((self.n,2,19));hg=xp.zeros((19,19))
        for start in range(0,self.n,self.chunk_size):
            sl=slice(start,min(start+self.chunk_size,self.n));w=self.weights[sl]
            predicted,js,jp,valid,_=self.batch(x[sl],p,sl,global_gradient or hessian)
            if not bool(xp.all(valid)):raise ValueError('inadmissible optical/P1 scale proposal')
            r=self.y[sl]-predicted;wr=r@self.precision
            point+=.5*xp.sum(w*xp.sum(r*wr,axis=1))
            gx[sl]=-xp.einsum('n,nis,ni->ns',w,js,wr)
            if global_gradient or hessian:gp-=xp.einsum('n,nip,ni->p',w,jp,wr)
            if hessian:
                wjs=xp.einsum('ij,njs->nis',self.precision,js)*w[:,None,None]
                wjp=xp.einsum('ij,njp->nip',self.precision,jp)*w[:,None,None]
                hx[sl]=xp.einsum('nis,niq->nsq',js,wjs)
                cross[sl]=xp.einsum('nis,nip->nsp',js,wjp)
                hg+=jp.reshape(-1,19).T@wjp.reshape(-1,19)
        means=self.group_sum(x)/self.counts[:,None];deviation=means-self.target
        anchors=.5*xp.sum(deviation**2/self.mean_scales**2,axis=0)/self.k
        data_gx=gx.copy()
        gx+=deviation[self.exposure]/(self.k*self.counts[self.exposure,None]*self.mean_scales**2)
        prior_cost=.5*xp.sum(self.prior*p*p);gp+=self.prior*p
        result={'cost':self.scalar(point+xp.sum(anchors)+prior_cost),'point':self.scalar(point),
                'theta_anchor':self.scalar(anchors[0]),'A_anchor':self.scalar(anchors[1]),
                'regularization':self.scalar(prior_cost),'temporal':0.,'gx':gx,'gp':gp,'data_gx':data_gx,
                'means':means,'deviation':deviation}
        if hessian:
            result.update(hx=hx,cross=cross,hg=hg+xp.diag(self.prior))
        return result

    def active(self,value,gradient,lower,upper):
        xp=self.xp
        return ((value<=lower+1e-9)&(gradient>0))|((value>=upper-1e-9)&(gradient<0))

    def stationarity(self,x,p,outcome):
        xp=self.xp
        sx=xp.where(self.active(x,outcome['gx'],self.state_lower,self.state_upper),0.,outcome['gx']*self.state_scales)
        sp=xp.where(self.active(p,outcome['gp'],self.lower,self.upper),0.,outcome['gp']*self.global_scales)
        return {'state_projected_gradient_inf':self.scalar(xp.max(xp.abs(sx))),
                'global_projected_gradient_inf':self.scalar(xp.max(xp.abs(sp)))}

    def coupled_inverse(self,hx,active,regularization=0.):
        xp=self.xp;free=~active
        h=hx.copy();diagonal=xp.diagonal(h,axis1=1,axis2=2)
        h[:,0,0]+=regularization*xp.maximum(xp.abs(diagonal[:,0]),1e-12)
        h[:,1,1]+=regularization*xp.maximum(xp.abs(diagonal[:,1]),1e-12)
        h[:,0,1]=xp.where(free[:,0]&free[:,1],h[:,0,1],0.)
        h[:,1,0]=h[:,0,1]
        h[:,0,0]=xp.where(free[:,0],h[:,0,0],1.)
        h[:,1,1]=xp.where(free[:,1],h[:,1,1],1.)
        if bool(xp.any(h[:,0,0]<=0)) or bool(xp.any(h[:,0,0]*h[:,1,1]-h[:,0,1]**2<=0)):
            raise ValueError('nonpositive free-state curvature')
        determinant=h[:,0,0]*h[:,1,1]-h[:,0,1]**2
        inverse=xp.empty_like(h)
        inverse[:,0,0]=h[:,1,1]/determinant;inverse[:,1,1]=h[:,0,0]/determinant
        inverse[:,0,1]=inverse[:,1,0]=-h[:,0,1]/determinant
        inverse*=free[:,:,None]*free[:,None,:]
        sums=self.group_sum(inverse)
        cinv=xp.zeros((self.k,2,2))
        cinv[:,0,0]=self.k*self.counts**2*self.mean_scales[0]**2
        cinv[:,1,1]=self.k*self.counts**2*self.mean_scales[1]**2
        reduced=xp.linalg.inv(cinv+sums)
        def apply(rhs):
            matrix=rhs.ndim==3
            q=xp.einsum('nij,njp->nip',inverse,rhs) if matrix else xp.einsum('nij,nj->ni',inverse,rhs)
            grouped=self.group_sum(q)
            correction=xp.einsum('eij,ejp->eip',reduced,grouped) if matrix else xp.einsum('eij,ej->ei',reduced,grouped)
            return q-(xp.einsum('nij,njp->nip',inverse,correction[self.exposure]) if matrix else xp.einsum('nij,nj->ni',inverse,correction[self.exposure]))
        return apply

    def observed_hessian(self,x,p):
        """FD of analytic gradients, retaining analytic full-mean coupling."""
        xp=self.xp;hx=xp.empty((self.n,2,2));cross=xp.empty((self.n,2,19));hg=xp.empty((19,19))
        for j,step in enumerate((1e-4,1e-5)):
            plus=x.copy();minus=x.copy()
            plus[:,j]=xp.minimum(x[:,j]+step,self.state_upper[j]);minus[:,j]=xp.maximum(x[:,j]-step,self.state_lower[j])
            denominator=plus[:,j]-minus[:,j]
            a=self.evaluate(plus,p,global_gradient=False);b=self.evaluate(minus,p,global_gradient=False)
            hx[:,:,j]=(a['data_gx']-b['data_gx'])/denominator[:,None]
        hx=(hx+hx.transpose(0,2,1))/2
        for j in range(19):
            plus=p.copy();minus=p.copy();step=1e-5
            plus[j]=xp.minimum(p[j]+step,self.upper[j]);minus[j]=xp.maximum(p[j]-step,self.lower[j])
            denominator=plus[j]-minus[j]
            a=self.evaluate(x,plus);b=self.evaluate(x,minus)
            cross[:,:,j]=(a['gx']-b['gx'])/denominator
            hg[:,j]=(a['gp']-b['gp'])/denominator
        return {'hx':hx,'cross':cross,'hg':(hg+hg.T)/2}

    def direction(self,x,p,o,kind='joint',regularization=1e-6):
        xp=self.xp;dx=xp.zeros_like(x);dp=xp.zeros_like(p)
        if kind in ('A','theta','joint'):
            active=self.active(x,o['gx'],self.state_lower,self.state_upper)
            if kind=='A':active[:,0]=True
            if kind=='theta':active[:,1]=True
            apply=self.coupled_inverse(o['hx'],active,regularization)
            inv_gradient=apply(o['gx'])
            if kind!='joint':return -inv_gradient,dp
            free=~self.active(p,o['gp'],self.lower,self.upper)
            b=o['cross'];z=apply(b)
            schur=o['hg']-xp.einsum('nsp,nsq->pq',b,z)
            rhs=-o['gp']+xp.einsum('nsp,ns->p',b,inv_gradient)
            schur=(schur+schur.T)/2
            schur+=xp.diag(regularization*xp.maximum(xp.abs(xp.diag(o['hg'])),1e-12))
            indices=xp.flatnonzero(free);sub=schur[xp.ix_(indices,indices)]
            # Reject indefinite reduced systems instead of following a saddle.
            if not bool(xp.all(xp.linalg.eigvalsh(sub)>0)):
                raise ValueError('nonpositive reduced proposal curvature')
            xp.linalg.cholesky(sub)
            dp[indices]=xp.linalg.solve(sub,rhs[indices]);dx=-inv_gradient-xp.einsum('nsp,p->ns',z,dp)
        else:
            blocks={'P1':range(2),'baseline':range(2,6),'K4':range(6,9),'D':range(9,19)}
            selected=xp.asarray(list(blocks[kind]));outward=self.active(p,o['gp'],self.lower,self.upper)
            selected=selected[~outward[selected]]
            h=o['hg'][xp.ix_(selected,selected)]
            if kind!='D':h+=xp.diag(regularization*xp.maximum(xp.abs(xp.diag(h)),1e-12))
            dp[selected]=xp.linalg.solve(h,-o['gp'][selected])
        return dx,dp

    def update(self,x,p,kind='joint',observed=False):
        xp=self.xp;o=self.evaluate(x,p,hessian=True)
        before_norm=self.stationarity(x,p,o)
        projected_x=xp.where(self.active(x,o['gx'],self.state_lower,self.state_upper),0.,o['gx']*self.state_scales)
        projected_p=xp.where(self.active(p,o['gp'],self.lower,self.upper),0.,o['gp']*self.global_scales)
        if kind in ('A','theta'):
            block_norm=self.scalar(xp.max(xp.abs(projected_x[:,1 if kind=='A' else 0])))
        elif kind=='joint':block_norm=max(before_norm.values())
        else:
            blocks={'P1':slice(0,2),'baseline':slice(2,6),'K4':slice(6,9),'D':slice(9,19)}
            block_norm=self.scalar(xp.max(xp.abs(projected_p[blocks[kind]])))
        if block_norm<=1e-8:
            return x,p,{'block':kind,'observed_curvature':observed,'accepted':False,'reason':'block_stationary',
                        'cost_before':o['cost'],'gradient_before':before_norm,'block_projected_gradient_inf':block_norm}
        if observed:o.update(self.observed_hessian(x,p))
        accepted=False;used=None
        for attempt in range(7):
            lm=(0. if observed and attempt==0 else 10.**(attempt-6))
            try:dx,dp=self.direction(x,p,o,kind,lm)
            except (ValueError,np.linalg.LinAlgError):continue
            # A trust cap is a numerical proposal control, not a state prior.
            cap=max(1.,self.scalar(xp.max(xp.abs(dx[:,0])))/2.,self.scalar(xp.max(xp.abs(dx[:,1]))))
            dx/=cap;dp/=cap
            for damping in range(14):
                step=.5**damping
                qx=xp.clip(x+step*dx,self.state_lower,self.state_upper)
                qp=xp.clip(p+step*dp,self.lower,self.upper)
                try:candidate=self.evaluate(qx,qp)
                except ValueError:continue
                allowance=64*np.finfo(float).eps*max(1.,o['cost'])
                norms=self.stationarity(qx,qp,candidate);oldnorm=self.stationarity(x,p,o)
                improves=max(norms.values())<max(oldnorm.values())
                if candidate['cost']<o['cost'] or (candidate['cost']<=o['cost']+allowance and improves):
                    x,p=qx,qp;accepted=True;used={'lm':lm,'damping':damping,'trust_cap':cap,'cost_after':candidate['cost'],**norms};break
            if accepted:break
        return x,p,{'block':kind,'observed_curvature':observed,'accepted':accepted,'cost_before':o['cost'],
                    'outcome':used,'gradient_before':before_norm}

    def certificate(self,x,p):
        xp=self.xp;o=self.evaluate(x,p,hessian=True);norms=self.stationarity(x,p,o)
        active=self.active(x,o['gx'],self.state_lower,self.state_upper)
        free_global=~self.active(p,o['gp'],self.lower,self.upper)
        exact=self.observed_hessian(x,p)
        record={**norms,'stationarity_threshold':1e-6,'free_global_parameters':int(self.scalar(xp.sum(free_global))),
                'states_at_bounds':self.host(xp.sum((x<=self.state_lower+1e-9)|(x>=self.state_upper-1e-9),axis=0)).tolist(),
                'globals_at_bounds':self.host(xp.flatnonzero((p<=self.lower+1e-9)|(p>=self.upper-1e-9))).tolist()}
        local=xp.linalg.eigvalsh(o['hx']/self.weights[:,None,None]);rank=(local[:,0]>local[:,1]*1e-10)
        record['state_data_rank2_count']=int(self.scalar(xp.sum(rank)))
        try:
            apply=self.coupled_inverse(exact['hx'],active)
            z=apply(exact['cross'])
            schur=exact['hg']-xp.einsum('nsp,nsq->pq',exact['cross'],z);schur=(schur+schur.T)/2
            indices=xp.flatnonzero(free_global);sub=schur[xp.ix_(indices,indices)]
            diagonal=xp.diag(sub)
            if bool(xp.any(diagonal<=0)):raise ValueError('nonpositive profiled shared curvature')
            normal=sub/xp.sqrt(diagonal[:,None]*diagonal[None,:])
            eigen=self.host(xp.linalg.eigvalsh(normal))
            record['observed_profile_curvature_eigenvalues']=eigen.tolist()
            record['observed_profile_rank']=int(np.sum(eigen>max(abs(eigen))*1e-10))
            record['observed_profile_positive']=bool(eigen.min()>max(abs(eigen))*1e-10)
            record['free_state_observed_curvature_positive']=True
        except (ValueError,np.linalg.LinAlgError) as error:
            record.update(observed_profile_rank=0,observed_profile_positive=False,
                          free_state_observed_curvature_positive=False,curvature_error=str(error))
        # Bounds conservatively guarantee positive optical scales throughout
        # the full shifted visual/A rectangle; verify actual native model too.
        from .p1 import P1Model,domain_margins as p1_domain
        from .accommodation import DM0Shape,domain_margins as p4_domain
        params=self.spec.parameters(self.host(p))
        d1=p1_domain(P1Model(params.b1,params.omega1,params.k1,params.theta_bounds))
        d4=p4_domain(DM0Shape(params.b4,params.omega4,params.aref,params.m1,params.k4,params.a_bounds,params.theta_bounds))
        record['domain']={'P1':d1,'P4':d4}
        record['fit_certified']=bool(max(norms.values())<=1e-6 and record['observed_profile_positive']
            and record['free_state_observed_curvature_positive'] and record['state_data_rank2_count']==self.n
            and d1['valid_domain'] and d4['valid_domain'])
        return record

    def snapshot(self,x,p):
        names=['g','F1','F4','mu1','mu4','M','D'];parts={key:[] for key in names};parts['prediction']=[]
        for start in range(0,self.n,self.chunk_size):
            sl=slice(start,min(start+self.chunk_size,self.n));prediction,_,_,valid,aux=self.batch(x[sl],p,sl,False)
            if not bool(self.xp.all(valid)):raise ValueError('invalid snapshot')
            for key in names:parts[key].append(self.host(aux[key]))
            parts['prediction'].append(self.host(prediction))
        return {key:np.concatenate(values) for key,values in parts.items()}


def fit_joint(objective,x0,p0,max_outer=12,joint_steps=4,polish_steps=6,checkpoint=None):
    xp=objective.xp;x=xp.asarray(x0).copy();p=xp.asarray(p0).copy();history=[]
    for outer in range(max_outer):
        for kind in ('A','theta','P1','baseline','K4','D'):
            repeats=2 if kind in ('A','theta') else 1
            for _ in range(repeats):
                x,p,event=objective.update(x,p,kind);event['outer']=outer;history.append(event)
        for _ in range(joint_steps):
            x,p,event=objective.update(x,p,'joint');event['outer']=outer;history.append(event)
        current=objective.evaluate(x,p);norms=objective.stationarity(x,p,current)
        if checkpoint is not None:checkpoint(outer,x,p,current,norms)
        if max(norms.values())<=1e-6:break
    # Same-objective Newton polish uses measured observed curvature, never a
    # weaker certificate or an increased optical capacity.
    for iteration in range(polish_steps):
        current=objective.evaluate(x,p);norms=objective.stationarity(x,p,current)
        if max(norms.values())<=1e-6:break
        x,p,event=objective.update(x,p,'joint',observed=True)
        event.update(polish=iteration);history.append(event)
        if not event['accepted']:break
    final=objective.evaluate(x,p);certificate=objective.certificate(x,p)
    return x,p,final,certificate,history
