"""Independent readonly reconstruction of free-size experiment artifacts."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[4];FAMILY=ROOT/'experiments/p1_scale_barrel_first_free_size';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(FAMILY/'scripts'))
from common import load_stage,result_dir,read_json,hashfile
from model import FreeSizeProblem,CASES,FIXED_CASES
from run import subset
AREF=.36036036036036034

def raw(B,theta,alpha,offset,R,k):
    n=len(theta);B=np.broadcast_to(B,(n,3,2));alpha=np.broadcast_to(alpha,(n,3));phi=(np.asarray(theta)-offset)/10;R=np.broadcast_to(R,(n,));k=np.broadcast_to(k,(n,));den=1+phi[:,None]*np.sum(alpha[:,None,1:]*B,axis=2)/R[:,None]
    U=np.exp(alpha[:,0,None,None]*phi[:,None,None]**2*np.array([1.,-1.]))*B/den[:,:,None];r2=np.sum(U*U,axis=2);Z=(1+k[:,None]*r2)[:,:,None]*U
    return Z,dict(den_min=float(den.min()),radial_factor_min=float((1+k[:,None]*r2).min()),radial_derivative_min=float((1+3*k[:,None]*r2).min()))

def metric(res,groups):
    mse=np.mean(np.sum(res*res,axis=-1),axis=-1);return float(np.sqrt(np.mean([mse[groups==group].mean() for group in np.unique(groups)])))
def checksha(summary,path):return summary['arrays_sha256']==hashfile(path/'arrays.npz')
def data(f):return [load_stage(i,family=f) for i in range(1,5)]

def frozen_source_hashes():
    manifest=read_json(FAMILY/'provenance'/'frozen_sources'/'MANIFEST.json')
    return {item['relative_path']:item['sha256'] for item in manifest['files']}

def main(fits_only=False):
    fits=[];held=[];profiles=[];sources={};maxnative=0.;maxinverse=0.;allflags=[]
    for family in [FAMILY,FAMILY/'validation_temporal']:
        d,p1,barrel,p4=data(family);c=d['capture'].astype(int)-1;mask=np.asarray(d['fit_mask'],bool);g=p1['primary_g']
        for case,config in CASES.items():
            path=result_dir(5,branch=case,family=family);z=load_stage(5,branch=case,family=family);s=read_json(path/'summary.json');q=z['q'];B=z['learned_B4'];cent=p4['B4_ref'].mean(0);size=np.exp(q[17]);energy=np.sum((p4['B4_ref']-cent)**2)
            Z,domain=raw(B,z['theta'],q[:12].reshape(4,3)[c],q[12],p4['R4'][c],q[13]*(z['A']-AREF));pred=g[:,None,None]*(Z-Z.mean(1,keepdims=True));native=float(np.max(abs(pred-z['p4_prediction'])));maxnative=max(maxnative,native)
            Zi,_=raw(z['recovered4'],z['theta'],q[:12].reshape(4,3)[c],q[12],p4['R4'][c],q[13]*(z['A']-AREF));roundtrip=g[:,None,None]*(Zi-Z.mean(1,keepdims=True));inverr=float(np.max(abs(roundtrip-d['X4'])));maxinverse=max(maxinverse,inverr)
            expected={str(result_dir(i,family=family)/'arrays.npz'):hashfile(result_dir(i,family=family)/'arrays.npz') for i in range(1,5)}
            flags=dict(arrays_hash=checksha(s,path),input_hashes=s['input_artifact_hashes']==expected,mask_exact=np.array_equal(z['fit_mask'],mask),shared_g_exact=np.array_equal(z['g'],g),radii_exact=np.array_equal(z['R4'],p4['R4']),inverse_all_valid=bool(z['valid4'].all()),native_domain_valid=min(domain.values())>0,centroid_fixed=float(np.max(abs(B.mean(0)-cent)))<1e-10,size_energy_correct=abs(np.sum((B-cent)**2)-size**2*energy)<1e-6)
            scientific={k:v for k,v in s['source_hashes'].items() if k.endswith('.py')};sources[str(family)+'/'+case]=scientific
            frozen=frozen_source_hashes()
            flags['snapshots_valid']=(scientific==frozen and all(hashfile(FAMILY/'provenance'/'frozen_sources'/name)==sha for name,sha in frozen.items()))
            p=FreeSizeProblem(subset(d,mask),p1,p4,barrel,g_override=g[mask],**config);cert=p.certificate(z['theta'][mask],z['A'][mask],q,True);components=p.evaluate(z['theta'][mask],z['A'][mask],q,optical=True)['components'];allflags.extend(flags.values());allflags.extend([native<1e-9,inverr<1e-9,cert['state_stationary'],cert['global_stationary']])
            fixed=load_stage(5,branch=FIXED_CASES[case],family=family);spans=np.array([np.ptp([z['A'][mask&(c==cc)&(d['fixation']==ff)].mean() for ff in range(5)]) for cc in range(4)]);fixedspans=np.array([np.ptp([fixed['A'][mask&(c==cc)&(d['fixation']==ff)].mean() for ff in range(5)]) for cc in range(4)])
            fits.append(dict(family=str(family),case=case,flags=flags,size_ratio=size,P4_forward_rms_reconstructed=metric(pred-d['X4'],d['group']),domain=domain,native_prediction_error=native,inverse_reconstruction_error=inverr,recomputed_certificate=cert,all_start_certificates=[x['certificate'] for x in s['starts']],multistart=s['multistart_spread'],loss_components=components,A_spans=spans,fixed_A_spans=fixedspans))
    science_identical=all(v==next(iter(sources.values())) for v in sources.values());allflags.append(science_identical)
    if fits_only:
        partial=dict(pass_all=bool(all(allflags)),scientific_sources_identical_across_four_fits=science_identical,fits=fits,max_native_prediction_difference=maxnative,max_inverse_native_difference=maxinverse)
        Path('/tmp/acc_free_size_fit_audit.json').write_text(json.dumps(partial,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else bool(v) if isinstance(v,np.bool_) else v)+'\n')
        for record in fits:print(record['family'],record['case'],'size',record['size_ratio'],'cert',record['recomputed_certificate']['state_projected_gradient_characteristic_inf'],record['recomputed_certificate']['global_projected_gradient_scaled_inf'])
        print('FITPASS',partial['pass_all'],'native/inverse',maxnative,maxinverse,'sciencehashmatch',science_identical)
        return
    trainfamily=FAMILY/'validation_temporal';d,p1,barrel,p4=data(trainfamily);test=np.asarray(d['test_mask'],bool);train=np.asarray(d['fit_mask'],bool);dt=subset(d,test);c=dt['capture'].astype(int)-1;fix=dt['fixation'].astype(int);anchor=np.array([d['theta0'][train&(d['group']==group)].mean() for group in dt['group']]);pars=p1['primary_params'];P1,_=raw(p1['primary_B1'],anchor,pars[:3],pars[3],float(p1['frozen_R01_px']),0.);H=P1-P1.mean(1,keepdims=True);expectedg=np.clip(np.sum(H*dt['X1'],axis=(1,2))/np.sum(H*H,axis=(1,2)),.98,1.02)
    for case,config in CASES.items():
        p5=load_stage(5,branch=case,family=trainfamily);path=result_dir(5,family=FAMILY)/'validation'/'temporal'/case;s=read_json(path/'summary.json');z=dict(np.load(path/'arrays.npz'));flags=dict(arrays_hash=checksha(s,path),source_row_exact=np.array_equal(z['source_row'],dt['source_row']),source_frame_exact=np.array_equal(z['source_frame'],dt['source_frame']),anchor_train_only=float(np.max(abs(z['theta_anchor']-anchor)))<1e-12,shared_P1only_g=float(np.max(abs(z['g']-expectedg)))<1e-12,training_hashes=s['training_artifact_hashes']=={str(result_dir(i,branch=case if i==5 else 'primary',family=trainfamily)/'arrays.npz'):hashfile(result_dir(i,branch=case if i==5 else 'primary',family=trainfamily)/'arrays.npz') for i in range(1,6)})
        flags['scientific_source_hashes']= {k:v for k,v in s['source_hashes'].items() if k.endswith('.py')}==next(iter(sources.values()))
        statecerts=[];errorlists=dict(initial=[],fixed=[],joint=[]);reconstruction=[];csvrows=list(csv.DictReader((path/'fixation_metrics.csv').open()));csvdifference=0.
        for heldslot in range(3):
            retained=[v for v in range(3) if v!=heldslot];observed=dt['P4_raw'][:,heldslot]-dt['P4_raw'][:,retained].mean(1)
            for label in errorlists:
                if label=='initial':B=p4['B4_ref'];theta=anchor;k=barrel['kappa_targets'][c,fix];alpha=p4['alpha'][c];off=float(p4['offset'])
                else:
                    q=p5['fixed_q'] if label=='fixed' else p5['q'];B=p4['B4_ref'] if label=='fixed' else p5['learned_B4'];theta=z[f'{label}_theta_vertex{heldslot}'];A=z[f'{label}_A_vertex{heldslot}'];k=q[13]*(A-AREF);alpha=q[:12].reshape(4,3)[c];off=q[12]
                if label!='initial':
                    pp=FreeSizeProblem(dt,p1,p4,barrel,g_override=z['g'],theta_anchor_override=anchor,vertices4=retained,**config);ccert=pp.certificate(theta,A,q,False);statecerts.append(dict(held_vertex=heldslot,model=label,certificate=ccert))
                ZZ,_=raw(B,theta,alpha,off,p4['R4'][c],z['g']*0+k);prediction=z['g'][:,None]*(ZZ[:,heldslot]-ZZ[:,retained].mean(1));error=observed-prediction;diff=float(np.max(abs(error-z[f'{label}_residual_vertex{heldslot}'])));reconstruction.append(diff);errorlists[label].append(error)
                for row in csvrows:
                    if row['model']==label and int(row['held_vertex'])==heldslot:
                        m=(dt['capture']==int(row['capture']))&(dt['fixation']==int(row['fixation']));rms=np.sqrt(np.mean(np.sum(error[m]**2,axis=1)));csvdifference=max(csvdifference,abs(rms-float(row['relative_vertex_rms_px'])))
        fixedpath=result_dir(5,family=FAMILY)/'validation'/'temporal'/FIXED_CASES[case];fixedheld=dict(np.load(fixedpath/'arrays.npz'));fixedsummary=read_json(fixedpath/'summary.json');cell_changes=[]
        for group in np.unique(dt['group']):
            m=dt['group']==group;freeerr=np.stack(errorlists['joint'],1)[m];fixederr=np.stack([fixedheld[f'joint_residual_vertex{v}'] for v in range(3)],1)[m];cell_changes.append(dict(capture=int(dt['capture'][m][0]),fixation=int(dt['fixation'][m][0]),free_rms=float(np.sqrt(np.mean(np.sum(freeerr**2,axis=2)))),fixed_rms=float(np.sqrt(np.mean(np.sum(fixederr**2,axis=2)))),difference=float(np.sqrt(np.mean(np.sum(freeerr**2,axis=2)))-np.sqrt(np.mean(np.sum(fixederr**2,axis=2))))))
        flags['immutable_fixed_held_roster']=np.array_equal(fixedheld['source_row'],z['source_row']) and np.array_equal(fixedheld['source_frame'],z['source_frame'])
        flags['immutable_initial_fixed_scores_match']=max(float(np.max(abs(np.stack(errorlists[label],1)-np.stack([fixedheld[f'{label}_residual_vertex{v}'] for v in range(3)],1)))) for label in ('initial','fixed'))<1e-8
        metrics={label:metric(np.stack(errors,1),dt['group']) for label,errors in errorlists.items()};metricdifference=max(abs(value-s['metrics'][label]['weighted_vertex_rms_px']) for label,value in metrics.items());flags['recomputed_state_certificates']=all(rec['certificate']['state_stationary'] and rec['certificate']['invalid_rows']==0 for rec in statecerts)
        flags.update(retained_native_score_reconstruction=max(reconstruction)<1e-9,csv_scores_reconstructed=csvdifference<1e-9,metrics_reconstructed=metricdifference<1e-9,all_six_state_certificates=all(x['certificate']['state_stationary'] and x['certificate']['invalid_rows']==0 for x in s['state_audits']));allflags.extend(flags.values());held.append(dict(recomputed_state_certificates=statecerts,fixed_control_weighted_rms=fixedsummary['metrics']['joint']['weighted_vertex_rms_px'],capture_fixation_changes=cell_changes,case=case,flags=flags,weighted_rms=metrics,csv_max_difference=csvdifference,native_residual_max_difference=max(reconstruction),test_rows=len(c),scored_vertices=3*len(c)))
    path=result_dir(5,family=FAMILY)/'validation'/'template_size_profile'/'free_size_sigma050_optical2';s=read_json(path/'summary.json');allflags.append(checksha(s,path));allflags.append({k:v for k,v in s['source_hashes'].items() if k.endswith('.py')}==next(iter(sources.values())));z=dict(np.load(path/'arrays.npz'));d,p1,barrel,p4=data(FAMILY);mask=np.asarray(d['fit_mask'],bool)
    for row in s['points']:
        label=row['point'];p=FreeSizeProblem(subset(d,mask),p1,p4,barrel,g_override=p1['primary_g'][mask],**CASES['free_size_sigma050_optical2']);p.bounds[17]=np.log(row['size_ratio']);q=z[label+'_q'];cert=p.certificate(z[label+'_theta'],z[label+'_A'],q,True);ev=p.evaluate(z[label+'_theta'],z[label+'_A'],q,True,True);flags=dict(size_exact=abs(q[17]-np.log(row['size_ratio']))<1e-14,conditional_cert=cert['state_stationary'] and cert['global_stationary'] and cert['invalid_rows']==0,objective=abs(ev['loss']-row['loss'])<1e-9,size_gradient_separate=abs(ev['global_gradient'][17]-row['unconstrained_size_gradient'])<1e-8)
        if label=='initial_size_one':flags['immutable_control_objective']=row['loss_minus_immutable_fixed_control']<=1e-8
        allflags.extend(flags.values());profiles.append(dict(point=label,flags=flags,loss=ev['loss'],loss_components=ev['components'],size_ratio=row['size_ratio'],certificate=cert))
    out=dict(pass_all=bool(all(allflags)),scientific_sources_identical_across_four_fits=science_identical,fits=fits,held=held,profiles=profiles,max_native_prediction_difference=maxnative,max_inverse_native_difference=maxinverse,independent_audit_script_sha256=hashfile(Path(__file__)),audit_boundary='Read-only numpy native D(K) and retained-centroid reconstruction, separately implemented pixel equations; certificate recomputation uses frozen scientific class. No fit sources/results modified.')
    def enc(v):return v.tolist() if isinstance(v,np.ndarray) else bool(v) if isinstance(v,np.bool_) else v
    Path('/tmp/acc_free_size_output_audit.json').write_text(json.dumps(out,indent=2,default=enc)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('fits','profiles')},indent=2,default=enc))
    if not out['pass_all']:raise SystemExit('Output audit failed')
if __name__=='__main__':main('--fits-only' in sys.argv)
