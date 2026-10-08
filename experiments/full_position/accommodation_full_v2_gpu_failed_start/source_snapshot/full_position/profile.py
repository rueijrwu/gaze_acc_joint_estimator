"""Independent fixed-accommodation polynomial branch audit; not global proof."""
from __future__ import annotations
import numpy as np
from .noise import whitening
from .invert import invert


def coordinate_polynomials(model, r, accommodation):
    """Ascending t coefficients; exact for the conditional27/37 bases."""
    if model.channels != 6 and getattr(getattr(model, "position", None), "channels", None) != 6:
        raise ValueError('Polynomial profile supports coordinate models only')
    t = np.array([-1., -1/3, 1/3, 1.])
    values = model.predict(np.column_stack((10*t,np.full(4,accommodation))), r)
    return np.linalg.solve(np.polynomial.polynomial.polyvander(t,3), values)


def gaze_candidates(model, r, y, cov, indices, accommodation):
    coefficients = coordinate_polynomials(model, r, accommodation)[:, indices].copy()
    coefficients[0] -= y
    residual = coefficients@whitening(cov).T
    cost = np.zeros(7)
    for j in range(len(indices)):
        term = np.polynomial.polynomial.polymul(residual[:,j],residual[:,j])
        cost[:len(term)] += term
    derivative = np.polynomial.polynomial.polyder(cost)
    roots = np.polynomial.polynomial.polyroots(derivative)
    real = [float(root.real) for root in roots if abs(root.imag) < 1e-7*(1+abs(root.real)) and -2 <= root.real <= 2]
    candidates = []
    for t in sorted(set([-2.,2.]+real)):
        x = np.array([10*t,accommodation])
        e = whitening(cov)@(model.predict(x,r)[indices]-y)
        candidates.append(dict(state=x.tolist(),cost=float(e@e)))
    return candidates


def profile_inverse(model,r,y,cov,indices=None,grid=65):
    if grid < 3:
        raise ValueError('Profile grid requires at least three accommodation values')
    indices = np.arange(model.channels) if indices is None else np.asarray(indices,int)
    accommodation = np.linspace(0,6,grid)
    levels = [gaze_candidates(model,r,y,cov,indices,a) for a in accommodation]
    seeds = []
    for i,level in enumerate(levels):
        for candidate in level:
            neighbours = [min(levels[j],key=lambda c: abs(c['state'][0]-candidate['state'][0]))
                          for j in [i-1,i+1] if 0 <= j < len(levels)]
            if i in [0,len(levels)-1] or all(candidate['cost'] <= n['cost'] for n in neighbours):
                seeds.append(candidate['state'])
    # Refine likely basin centres on a denser A grid, still without withheld data.
    for state in list(seeds):
        width=6/(grid-1)
        for a in np.linspace(max(0,state[1]-width),min(6,state[1]+width),9):
            candidates=gaze_candidates(model,r,y,cov,indices,a)
            nearest=min(candidates,key=lambda c: abs(c['state'][0]-state[0]))
            seeds.append(nearest['state'])
    seeds=np.unique(np.round(seeds,12),axis=0)
    result=invert(model,r,y,cov,indices,starts=seeds)
    result['profile_audit']=dict(accommodation_grid=grid,refinement_points=9,
        seed_count=len(seeds),finite_grid_not_global_completeness=True)
    return result
