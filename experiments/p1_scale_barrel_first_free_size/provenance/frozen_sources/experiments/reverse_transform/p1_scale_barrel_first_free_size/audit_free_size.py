#!/usr/bin/env python3
"""Native derivative, exact size-zero control, inverse and omission audits."""
import sys,argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import FreeSizeProblem,CASES
from distortion_model.barrel_first_pipeline import BarrelFirstStateProblem
from distortion_model.capturewise_reverse import forward as p1forward,host
from distortion_model.capturewise_keystone_first import forward,inverse

def run(xp):
    rng=np.random.default_rng(701);capture=np.concatenate([np.full(f+3,c) for c in range(1,5) for f in range(5)]);fix=np.concatenate([np.full(f+3,f) for c in range(4) for f in range(5)]);group=(capture-1)*5+fix;n=len(capture);theta=np.array([-10,-5,0,5,10])[fix]+rng.normal(0,.1,n);g=1+rng.uniform(-.01,.01,n);B=np.array([[-160.,-110.],[170.,-90.],[-10.,200.]])+[.4,-.6];B1=np.array([[-80.,-60.],[90.,-40.],[-10.,100.]]);alpha=np.tile([.004,.018,-.013],(4,1));k=np.array([0.,-1.8e-7,-1.4e-7,-.9e-7])[:,None]*(1+np.linspace(-.08,.08,5)[None,:]);slope=np.sum(np.array([0.,4-.36036036036036034,3-.36036036036036034,2-.36036036036036034])[:,None]*k)/np.sum(np.broadcast_to(np.array([0.,4-.36036036036036034,3-.36036036036036034,2-.36036036036036034])[:,None],(4,5))**2);A=.36036036036036034+k[capture-1,fix]/slope+.02
    pars=np.array([.003,.01,-.008,1.,0.,0.,0.]);X1=p1forward(B1,theta,pars[:3],pars[3],120.,g)['prediction'];X4=forward(B,theta,alpha[capture-1],-2.,200.,g,slope*(A-.36036036036036034))['prediction'];data=dict(capture=capture,fixation=fix,group=group,theta0=theta,X1=X1,X4=X4,P4_raw=X4+[800.,600.]);p1=dict(primary_B1=B1,primary_params=pars,primary_g=g,frozen_R01_px=120.);p4=dict(B4_ref=B,alpha=alpha,offset=np.asarray(-2.),R4=np.full(4,200.));barrel=dict(kappa_targets=k)

    checks=[];equiv=[];roundtrip=[];leak=[];template_fd=[];matched=[]
    for case,config in CASES.items():
        for vertices in (None,[0,2]):
            problem=FreeSizeProblem(data,p1,p4,barrel,xp=xp,g_override=g,vertices4=vertices,**config)
            fixed=FreeSizeProblem(data,p1,p4,barrel,xp=xp,g_override=g,vertices4=vertices,free_size=False,**config)
            q=problem.q0.copy();q[:12]+=.001;q[14:17]=[.002,-.003,.001];t=theta+.01
            f=fixed.evaluate(xp.asarray(t),xp.asarray(A),q[:17],True,True);e=problem.evaluate(xp.asarray(t),xp.asarray(A),q,True,True)
            equiv.append(dict(case=case,retained=problem.vertices4.tolist(),loss=abs(f['loss']-e['loss']),state_gradient=float(np.max(abs(host(f['gradient']-e['gradient'],xp)))),global_scaled=float(np.max(abs((f['global_gradient']-e['global_gradient'][:17])*problem.scales[:17]))),prediction=float(np.max(abs(host(f['p4']['selected_prediction']-e['p4']['selected_prediction'],xp))))))
            q[17]=np.log(1.03);e=problem.evaluate(xp.asarray(t),xp.asarray(A),q,True,True);errors=[]
            for j in range(18):
                h=problem.scales[j]*1e-5;plus=q.copy();minus=q.copy();plus[j]+=h;minus[j]-=h;num=(problem.evaluate(xp.asarray(t),xp.asarray(A),plus,optical=True)['loss']-problem.evaluate(xp.asarray(t),xp.asarray(A),minus,optical=True)['loss'])/2e-5
                errors.append(abs(num-e['global_gradient'][j]*problem.scales[j]))
            state_errors=[]
            for col,unit in [(0,.5),(1,.25)]:
                direction=rng.normal(size=n)*unit;h=1e-5;tp=t+h*direction if col==0 else t;tm=t-h*direction if col==0 else t;ap=A+h*direction if col==1 else A;am=A-h*direction if col==1 else A
                num=(problem.evaluate(xp.asarray(tp),xp.asarray(ap),q,optical=True)['loss']-problem.evaluate(xp.asarray(tm),xp.asarray(am),q,optical=True)['loss'])/(2*h);state_errors.append(abs(num-float(host(xp.sum(e['gradient'][:,col]*xp.asarray(direction)),xp))))
            matched.append(dict(loss=e['loss'],state_gradient=host(e['gradient'],xp),global_scaled=e['global_gradient']*problem.scales,prediction=host(e['p4']['selected_prediction'],xp)))
            checks.append(dict(case=case,retained=problem.vertices4.tolist(),global_fd_scaled_max=max(errors),state_fd_max=max(state_errors)))
            BB,dBB,v=problem.template_geometry(q,True);BB=host(BB,xp);dBB=host(dBB,xp)
            for j in range(4):
                h=1e-6;plus=q.copy();minus=q.copy();plus[14+j]+=h;minus[14+j]-=h;num=(host(problem.template_geometry(plus),xp)-host(problem.template_geometry(minus),xp))/(2*h);template_fd.append(float(np.max(abs(num-dBB[:,:,j]))))
            ff=forward(BB,t,alpha[capture-1]+.001,q[12],200.,g,q[13]*(A-.36036036036036034));iv=inverse(BB,ff['prediction'],t,alpha[capture-1]+.001,q[12],200.,g,q[13]*(A-.36036036036036034));roundtrip.append(float(np.max(abs(iv['recovered']-BB))))
            if vertices is not None:
                modified=dict(data);modified['P4_raw']=data['P4_raw'].copy();modified['P4_raw'][:,1]+=rng.normal(size=(n,2))*1000
                pp=FreeSizeProblem(modified,p1,p4,barrel,xp=xp,g_override=g,vertices4=vertices,**config);ee=pp.evaluate(xp.asarray(t),xp.asarray(A),q,True,True);leak.append(abs(ee['loss']-e['loss']))
    rank=problem.data_only_rank(t,A,q)
    out=dict(_matched_outputs=matched,exact_size_zero_equivalence=equiv,derivative_checks=checks,template_derivative_max_abs=max(template_fd),roundtrip_max_abs_px=max(roundtrip),held_vertex_loss_leak=max(leak),data_only_rank=rank,size_ratio_bounds=[.5,2.],size_prior='none',Aref=.36036036036036034)
    out['pass']=max(x['loss'] for x in equiv)<1e-12 and max(x['global_scaled'] for x in equiv)<1e-12 and max(x['state_gradient'] for x in equiv)<1e-12 and max(x['prediction'] for x in equiv)<1e-12 and max(x['global_fd_scaled_max'] for x in checks)<2e-5 and max(x['state_fd_max'] for x in checks)<1e-6 and max(template_fd)<1e-6 and max(roundtrip)<1e-9 and max(leak)==0 and rank['dimension']==18
    return out

def actual_controls(family,xp):
    from common import load_stage,result_dir,read_json,hashfile
    from run import checked_stage,subset
    from types import SimpleNamespace
    from model import FIXED_CASES
    records=[]
    for ff in [family,family/'validation_temporal']:
        d=load_stage(1,family=ff);p1=load_stage(2,family=ff);a=SimpleNamespace(family=ff);barrel=checked_stage(3,a,d,p1);p4=checked_stage(4,a,d,p1);mask=np.asarray(d['fit_mask'],bool);dd=subset(d,mask)
        for case,config in CASES.items():
            branch=FIXED_CASES[case];z=load_stage(5,branch=branch,family=ff);summary=read_json(result_dir(5,branch=branch,family=ff)/'summary.json');theta=xp.asarray(z['theta'][mask]);A=xp.asarray(z['A'][mask])
            p=FreeSizeProblem(dd,p1,p4,barrel,xp=xp,g_override=z['g'][mask],**config);fixed=FreeSizeProblem(dd,p1,p4,barrel,xp=xp,g_override=z['g'][mask],free_size=False,**config)
            ev=p.evaluate(theta,A,np.r_[z['q'],0.],True,True);old=fixed.evaluate(theta,A,z['q'],True,True);chosen=next(start for start in summary['starts'] if start['start']==summary['selected_start'])
            records.append(dict(family=str(ff),case=case,rows=int(mask.sum()),loss_zero_difference=abs(ev['loss']-old['loss']),saved_loss_difference=abs(ev['loss']-chosen['loss']),saved_native_prediction_max_abs=float(np.max(abs(host(ev['p4']['prediction'],xp)-z['p4_prediction'][mask]))),shared_g_identical=bool(np.array_equal(z['g'],p1['primary_g'])),fit_mask_identical=bool(np.array_equal(z['fit_mask'],mask))))
    return records

if __name__=='__main__':
    from common import FAMILY
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['numpy','cupy'],default='cupy');ap.add_argument('--family',type=Path,default=FAMILY);ap.add_argument('--out',type=Path,default=Path('/tmp/acc_free_size_audit.json'));args=ap.parse_args();xp=np
    if args.backend=='cupy':import cupy as xp
    result=run(xp);result['immutable_control_equivalence']=actual_controls(args.family,xp);result['pass'] &= all(z['loss_zero_difference']<1e-12 and z['saved_loss_difference']<1e-9 and z['saved_native_prediction_max_abs']<1e-9 and z['shared_g_identical'] and z['fit_mask_identical'] for z in result['immutable_control_equivalence'])
    if xp is not np:
        result['cpu']=run(np);pairs=list(zip(result.pop('_matched_outputs'),result['cpu'].pop('_matched_outputs')));result['cpu_gpu_output_max_abs']={key:max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key])))) for a,b in pairs) for key in ('loss','state_gradient','global_scaled','prediction')};result['pass'] &= max(result['cpu_gpu_output_max_abs'].values())<1e-9;result['cpu_gpu_fd_max_delta']=max(abs(a['global_fd_scaled_max']-b['global_fd_scaled_max']) for a,b in zip(result['derivative_checks'],result['cpu']['derivative_checks']));result['pass'] &= result['cpu']['pass']
    result.pop('_matched_outputs',None)
    def encoder(v):return v.tolist() if isinstance(v,np.ndarray) else bool(v) if isinstance(v,np.bool_) else v
    args.out.write_text(json.dumps(result,indent=2,default=encoder)+'\n');print(json.dumps(result,indent=2,default=encoder))
    if not result['pass']:raise SystemExit('Free size audit failed')
