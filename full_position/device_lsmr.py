"""Device-resident LSMR vectors for the profiled optimizer.

LSMR recurrences and stopping rules follow SciPy (SCIPY_LICENSE.txt). Only scalar
norms cross the device boundary per iteration; the large Krylov vectors stay on
the GPU. SciPy's outer trust-region strategy and its fit certificates remain.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.optimize._lsq.trf as trf
from scipy.sparse.linalg._isolve.lsqr import _sym_ortho

_ORIGINAL = (trf.right_multiplied_operator,trf.regularized_lsq_operator,trf.lsmr)


def operator(shape, matvec, rmatvec, xp):
    op = LinearOperator(shape,matvec=lambda x:xp.asnumpy(matvec(xp.asarray(x).ravel())),
                        rmatvec=lambda x:xp.asnumpy(rmatvec(xp.asarray(x).ravel())),dtype=float)
    op.device_matvec,op.device_rmatvec,op.array_backend = matvec,rmatvec,xp
    return op


def right(J,d):
    if not hasattr(J,"device_matvec"):
        return _ORIGINAL[0](J,d)
    xp=J.array_backend; d=xp.asarray(d)
    return operator(J.shape,lambda v:J.device_matvec(v*d),lambda w:d*J.device_rmatvec(w),xp)


def regularized(J,diag):
    if not hasattr(J,"device_matvec"):
        return _ORIGINAL[1](J,diag)
    xp=J.array_backend; diag=xp.asarray(diag); m,n=J.shape
    return operator((m+n,n),lambda v:xp.concatenate((J.device_matvec(v),diag*v)),
                    lambda w:J.device_rmatvec(w[:m])+diag*w[m:],xp)


def lsmr(A,b,damp=0.,atol=1e-6,btol=1e-6,conlim=1e8,maxiter=None,show=False,x0=None):
    if not hasattr(A,"device_matvec"):
        return _ORIGINAL[2](A,b,damp,atol,btol,conlim,maxiter,show,x0)
    xp=A.array_backend
    norm=lambda v:float(xp.linalg.norm(v))
    b=xp.asarray(b,dtype=xp.float64).ravel().copy()
    m,n=A.shape
    maxiter=min(m,n) if maxiter is None else maxiter
    x=xp.zeros(n) if x0 is None else xp.asarray(x0,dtype=xp.float64).ravel().copy()
    u=b.copy() if x0 is None else b-A.device_matvec(x)
    normb=norm(b); beta=norm(u)
    if beta>0:
        u/=beta; v=A.device_rmatvec(u); alpha=norm(v)
    else:
        v=xp.zeros(n); alpha=0.
    if alpha>0: v/=alpha
    zetabar=alpha*beta; alphabar=alpha
    rho=rhobar=cbar=1.; sbar=0.
    h=v.copy(); hbar=xp.zeros(n)
    betadd=beta; betad=0.; rhodold=1.
    tautildeold=thetatilde=zeta=d=0.
    normA2=alpha*alpha; maxrbar=0.; minrbar=1e100
    normA=np.sqrt(normA2); condA=1.; normx=0.; istop=0; itn=0
    ctol=1/conlim if conlim>0 else 0.
    normr=beta; normar=alpha*beta
    if normb==0: x.fill(0)
    if normar!=0 and normb!=0:
        for itn in range(1,maxiter+1):
            u*=-alpha; u+=A.device_matvec(v); beta=norm(u)
            if beta>0:
                u/=beta; v*=-beta; v+=A.device_rmatvec(u); alpha=norm(v)
                if alpha>0: v/=alpha
            chat,shat,alphahat=_sym_ortho(alphabar,damp)
            rhoold=rho; c,s,rho=_sym_ortho(alphahat,beta)
            thetanew=s*alpha; alphabar=c*alpha
            rhobarold=rhobar; zetaold=zeta
            thetabar=sbar*rho; rhotemp=cbar*rho
            cbar,sbar,rhobar=_sym_ortho(cbar*rho,thetanew)
            zeta=cbar*zetabar; zetabar=-sbar*zetabar
            hbar*=-(thetabar*rho/(rhoold*rhobarold)); hbar+=h
            x+=zeta/(rho*rhobar)*hbar
            h*=-(thetanew/rho); h+=v
            betaacute=chat*betadd; betacheck=-shat*betadd
            betahat=c*betaacute; betadd=-s*betaacute
            thetatildeold=thetatilde
            ctildeold,stildeold,rhotildeold=_sym_ortho(rhodold,thetabar)
            thetatilde=stildeold*rhobar; rhodold=ctildeold*rhobar
            betad=-stildeold*betad+ctildeold*betahat
            tautildeold=(zetaold-thetatildeold*tautildeold)/rhotildeold
            taud=(zeta-thetatilde*tautildeold)/rhodold
            d+=betacheck*betacheck
            normr=np.sqrt(d+(betad-taud)**2+betadd*betadd)
            normA2+=beta*beta; normA=np.sqrt(normA2); normA2+=alpha*alpha
            maxrbar=max(maxrbar,rhobarold)
            if itn>1: minrbar=min(minrbar,rhobarold)
            condA=max(maxrbar,rhotemp)/min(minrbar,rhotemp)
            normar=abs(zetabar); normx=norm(x)
            test1=normr/normb
            test2=normar/(normA*normr) if normA*normr!=0 else np.inf
            test3=1/condA
            t1=test1/(1+normA*normx/normb)
            rtol=btol+atol*normA*normx/normb
            if itn>=maxiter: istop=7
            if 1+test3<=1: istop=6
            if 1+test2<=1: istop=5
            if 1+t1<=1: istop=4
            if test3<=ctol: istop=3
            if test2<=atol: istop=2
            if test1<=rtol: istop=1
            if istop: break
    if show:
        print(f"GPU LSMR: iterations={itn}, stop={istop}, residual={normr:.6g}")
    return xp.asnumpy(x),istop,itn,normr,normar,normA,condA,normx


def enable():
    trf.right_multiplied_operator=right
    trf.regularized_lsq_operator=regularized
    trf.lsmr=lsmr


def disable():
    trf.right_multiplied_operator,trf.regularized_lsq_operator,trf.lsmr=_ORIGINAL
