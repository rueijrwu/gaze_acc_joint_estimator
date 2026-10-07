"""Frozen-response measurement ablations; no calibration or withheld leakage."""
from __future__ import annotations
import numpy as np
from .geometry import area_gradient, signed_area
from .invert import invert
from .noise import marginal


AREA_HESSIAN = np.zeros((6,6))
for x, y, value in [(0,3,.5),(0,5,-.5),(2,5,.5),(2,1,-.5),(4,1,.5),(4,3,-.5)]:
    AREA_HESSIAN[x,y] = AREA_HESSIAN[y,x] = value


def summary_transform(v):
    v = np.asarray(v).reshape(3,2)
    G = np.zeros((2,6))
    G[0,::2] = 1/3
    G[1] = area_gradient(v).ravel()
    return np.array([v[:,0].mean(),abs(signed_area(v))]), G


class DerivedSummaryModel:
    """Same frozen six-coordinate response reduced to centroid x/area."""
    channels = 2
    def __init__(self, position):
        self.position = position

    def predict(self,x,r,derivatives=False):
        v,J = self.position.predict(x,r,True)
        value,G = summary_transform(v)
        return (value,G@J) if derivatives else value

    def state_hessian(self,x,r):
        v,J = self.position.predict(x,r,True)
        _,G = summary_transform(v)
        H = np.einsum('kc,cab->kab',G,self.position.state_hessian(x,r))
        H[1] += J.T@(np.sign(signed_area(v.reshape(3,2)))*AREA_HESSIAN)@J
        return H


def same_response_summaries(model,r,v,reference_v,cov):
    """All-three state diagnostic ONLY; never used for withheld prediction.

    The covariance transform is frozen at the training-only pilot reference,
    rather than computed from the observed P4 values.
    """
    y,_ = summary_transform(v)
    _,G = summary_transform(reference_v)
    result = invert(DerivedSummaryModel(model),r,y,G@cov@G.T)
    result['measurement_policy'] = 'all_three derived centroid/area; not a withheld-point baseline'
    return result


def retained_x_prediction(model,ctx,held,v,cov):
    kept = np.array([2*j for j in range(3) if j != held])
    result = invert(model,ctx.r,np.asarray(v)[kept],marginal(cov,kept),kept)
    result['held_point'] = held
    if result['available']:
        result['testable'] = bool(result['rank'] == 2 and not result['ambiguous'])
        result['predictions'] = [dict(state=b['state'],pixel=(ctx.c+ctx.ell*
            model.predict(np.array(b['state']),ctx.r).reshape(3,2)[held]).tolist()) for b in result['plausible_branches']]
    return result
