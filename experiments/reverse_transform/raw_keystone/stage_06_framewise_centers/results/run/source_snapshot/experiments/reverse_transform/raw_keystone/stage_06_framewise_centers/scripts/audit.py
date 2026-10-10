"""CPU replay of sequential center calibration and fixed-gaze accommodation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import run

def optical(p, states, pattern):
    b, k = p[f'reference_p{pattern}'], p['k1' if pattern == 1 else 'k4']
    t = states[:, :2]/p['gaze_units_deg']
    r = np.sqrt(np.mean(np.sum(b*b, axis=1)))
    kap = p['slope']*(states[:, 2]-p['reference_A']) if pattern == 4 else np.zeros(len(states))
    exponent = k[:, 0]*t[:, 0]**2-k[:, 1]*t[:, 1]**2
    f = np.empty((len(states), 3, 2))
    denominators = np.empty((len(states), 3))
    for j in range(3):
        point = b[j][None]*(1+kap*np.dot(b[j], b[j]))[:, None]
        denominator = 1+(k[:, 2]*t[:, 0]*point[:, 1]+k[:, 3]*t[:, 1]*point[:, 0])/r
        f[:, j, 0] = point[:, 0]*np.exp(exponent)/denominator
        f[:, j, 1] = point[:, 1]*np.exp(-exponent)/denominator
        denominators[:, j] = denominator
    return f, denominators



def oracle(p,s):
    f1,d1=optical(p,s,1)
    f4,d4=optical(p,s,4)
    mu1,mu4=f1.mean(axis=1),f4.mean(axis=1)
    c1,c4=p['p1'].mean(axis=1),p['p4'].mean(axis=1)
    h1,h4=f1-mu1[:,None],f4-mu4[:,None]
    m=np.sum((p['p1']-c1[:,None])*h1,axis=(1,2))/np.sum(h1*h1,axis=(1,2))
    C1,C4=c1-m[:,None]*mu1,c4-m[:,None]*mu4
    assert (d1>1e-8).all() and (d4>1e-8).all() and (m>0).all()
    return dict(m=m,centers=[C1,C4],mu=[mu1,mu4],f=[f1,f4],
        pred=[m[:,None,None]*h1,m[:,None,None]*h4],D=(C4-C1)/m[:,None])


def calibration(p,D):
    result=np.empty_like(D)
    for c in range(4):
        mask=p['capture_index']==c
        e=p['exposure'][mask]%5
        means=np.array([D[mask][e==f].mean(axis=0) for f in range(5)])
        z=means[2,0]
        scale=np.ptp(means[:,0])/2
        x=means[:,0]-z
        # Native-coordinate fit, independently from normalized fit in run.py.
        a,b=np.linalg.lstsq(np.column_stack((x/scale,(x/scale)**2)),[-10,-5,0,5,10],rcond=None)[0]
        dx=D[mask,0]-z
        result[mask,0]=a*dx/scale+b*(dx/scale)**2
        result[mask,1]=a/scale*(D[mask,1]-means[2,1])
    return result


def cost(p,s,m):
    o=oracle(p,s)
    f=o['f'][1]
    pred=m[:,None,None]*(f-f.mean(axis=1,keepdims=True))
    err=p['p4']-p['p4'].mean(axis=1,keepdims=True)-pred
    return np.sum(err*err,axis=(1,2))/3


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    out=parser.parse_args().results.resolve()
    prov=json.loads((out/'provenance.json').read_text())
    for group in ('source_sha256','parent_sha256'):
        for path,value in prov[group].items():
            assert run.digest(run.ROOT/path)==value,path
            if group=='source_sha256':
                assert run.digest(out/'source_snapshot'/path)==value,path
    p=run.load_inputs()
    saved=run.npz(out/'frames.npz')
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='COMPLETE',summary['status']
    assert np.array_equal(saved['initial_states'],p['initial_states'])
    for key in ('p1','p4','population_index','capture_index','exposure','row','source_frame','reference_p1','reference_p4','gaze_units_deg','k1','k4','expected_A_D'):
        assert np.array_equal(saved[key],p[key]),key
    checks={}
    previous=p['initial_states']
    for rec in summary['cycles']:
        cycle=run.npz(out/f"cycle_{rec['cycle']:02d}.npz")
        assert np.array_equal(cycle['input_states'],previous)
        o=oracle(p,previous)
        proposal=calibration(p,o['D'])
        assert np.max(abs(o['D']-cycle['input_corrected_separation_xy']))<1e-8
        assert np.max(abs(proposal-cycle['polynomial_gaze_xy']))<1e-8
        gaze=previous[:,:2]+rec['gaze_update_fraction']*(proposal-previous[:,:2])
        assert np.max(abs(gaze-cycle['updated_gaze_xy']))<1e-8
        states=cycle['output_states']
        assert np.array_equal(states[:,:2],cycle['updated_gaze_xy'])
        oo=oracle(p,states)
        assert np.max(abs(oo['m']-cycle['updated_magnification']))<1e-10
        model=run.model_A(p,gaze,oo['m'])
        lo,hi=model.feasible_bounds()
        assert np.array_equal(lo,cycle['lower_A_D']) and np.array_equal(hi,cycle['upper_A_D'])
        assert (states[:,2]>=lo).all() and (states[:,2]<=hi).all()
        cert,_=model.certificate(states[:,2],lo,hi)
        assert cert['stationary'],cert
        # Finite difference raw vertices; common P1 scale held fixed.
        h=1e-4
        plus,minus=states.copy(),states.copy()
        plus[:,2]+=h; minus[:,2]-=h
        grad=(cost(p,plus,oo['m'])-cost(p,minus,oo['m']))/(2*h)
        e=p['exposure']; counts=np.bincount(e,minlength=20)
        means=np.array([states[e==f,2].mean() for f in range(20)])
        expected=np.array([p['expected_A_D'][e==f][0] for f in range(20)])
        q=len(e)/(20*counts[e])
        independent=q*(grad+32*(means-expected)[e])
        value,analytic=model.evaluate(states[:,2])
        error=float(np.max(abs(independent-analytic)/(1+abs(analytic))))
        assert error<1e-4,error
        objective=np.dot(q,cost(p,states,oo['m']))/len(e)+16*np.mean((means-expected)**2)
        assert abs(objective-value/len(e))<1e-8
        checks[f"cycle_{rec['cycle']}"]=dict(certificate=cert,gradient_relative_error_max=error,objective_equal_fixation_px2=objective)
        previous=states
    assert np.array_equal(previous,saved['refined_states'])
    for label in ('initial','refined'):
        states=saved[f'{label}_states']; o=oracle(p,states)
        assert np.max(abs(o['m']-saved[f'{label}_magnification']))<1e-10
        for j,n in enumerate((1,4)):
            assert np.max(abs(o['centers'][j]-saved[f'{label}_center_p{n}_xy']))<1e-8
            assert np.max(abs(o['pred'][j]-saved[f'{label}_prediction_p{n}']))<1e-8
            assert np.max(abs(o['centers'][j]+o['m'][:,None]*o['mu'][j]-p[f'p{n}'].mean(axis=1)))<1e-8
        assert np.max(abs(o['D']-saved[f'{label}_corrected_separation_xy']))<1e-8
        shift=np.column_stack((37*np.sin(np.arange(len(states))),23*np.cos(np.arange(len(states)))))
        moved=dict(p,p1=p['p1']+shift[:,None],p4=p['p4']+shift[:,None])
        mo=oracle(moved,states)
        assert np.max(abs(mo['D']-o['D']))<1e-8
        origins=[400+shift,700+shift]
        synthetic=dict(p,**{f'p{n}':origins[j][:,None]+o['m'][:,None,None]*o['f'][j] for j,n in enumerate((1,4))})
        so=oracle(synthetic,states)
        closure=max(float(np.max(abs(so['centers'][j]-origins[j]))) for j in range(2))
        assert closure<1e-8
        b1,v1,b4,v4=run.inverse(synthetic,states,so['m'])
        assert v1.all() and v4.all()
        inverse_error=max(float(np.max(abs(b1-p['reference_p1']))),float(np.max(abs(b4-p['reference_p4']))))
        assert inverse_error<1e-8
        b1,v1,b4,v4=run.inverse(p,states,o['m'])
        for n,b,v in ((1,b1,v1),(4,b4,v4)):
            assert np.array_equal(v,saved[f'{label}_inverse_valid_p{n}'])
            assert np.max(abs(b[v]-saved[f'{label}_recovered_p{n}'][v]))<1e-8
        checks[label]=dict(synthetic_center_closure_px=closure,synthetic_inverse_closure_px=inverse_error,
            p1_inverse_failures=int((~v1).sum()),p4_inverse_failures=int((~v4).sum()))
    residual=float(np.max(abs(calibration(p,oracle(p,previous)['D'])-previous[:,:2])))
    assert residual<summary['protocol']['tolerance']
    checks['next_gaze_fixed_point_max_deg']=residual
    run.write(out/'audit.json',dict(status='PASS',independent_CPU=True,checks=checks,
        scope='Raw vertex equations, frame centers, arithmetic fixation-mean polynomial calibration, sequential P1 scale and fixed-gaze A, all-frame A gradients, bounds/stationarity, translation invariance and synthetic closure. Conditional origins, no physical center ground truth.'))
    print(json.dumps(dict(status='PASS',fixed_point_gaze_error_deg=residual)),flush=True)

if __name__=='__main__':
    main()
