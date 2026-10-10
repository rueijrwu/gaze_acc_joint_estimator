import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import cupy as cp
ROOT=Path('/home/aplab/ACC'); ST=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration'; OUT=ST/'results/preflight_1degree'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation/scripts'))
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
runner=loadmod('g7b_base',ST/'scripts/run_joint.py');g6=loadmod('g6_io',ST/'scripts/run.py')
from distortion_model.joint import JointSpec,JointDM0
from distortion_model.centroid_bound import CentroidBound,BoundedJointDM0
from distortion_model.optics import p1_reference,p4_reference,center_polynomial
from distortion_model.geometry import relative_coordinates
parent=ST/'results/g6_attempt_02';inputs,initial,record=runner.check_parent(parent)
valid=inputs['valid'];pop=inputs['population'];e=inputs['exposure'];y=relative_coordinates(pop['p1'][valid],pop['p4'][valid]);cov=inputs['covariance']['relative']
spec=JointSpec(np.array(record['b1_reference_px']),np.array(record['b4_reference_px']),record['omega1_visual_deg'],record['omega4_visual_deg'],record['Aref_D'])
p0=spec.pack(record);x0=np.column_stack((initial['theta_visual_deg'][valid],initial['accommodation_D'][valid]))
bound=CentroidBound(spec,p0,1.0)
# Actual public optical derivative grid, FD at nodes; finite-difference steps small relative to grid.
def H(p,t,a):
 q=spec.parameters(p);_,mu1,_,v1=p1_reference(t,q);_,mu4,v4=p4_reference(t,a,q)
 return center_polynomial(t,a,q)+mu4-mu1

def dense(p):
 ts=np.linspace(-20,20,81);aa=np.linspace(0,6,25);tt,aaa=np.meshgrid(ts,aa,indexing='ij');h=1e-4
 ht=(H(p,tt+h,aaa)-H(p,tt-h,aaa))/(2*h);ha=(H(p,tt,aaa+h)-H(p,tt,aaa-h))/(2*h)
 min_s=float(np.min(ht[...,0]));max_a=float(np.max(np.abs(ha[...,0])))
 endpoint_shift=H(p,tt,np.full_like(tt,4.))-H(p,tt,np.zeros_like(tt))
 cell_slope=bound.slope_floor;cell_cap=bound.a_slope_cap
 # Every grid node is incident to 1,2, or 4 closed interval boxes; evaluate the actual derivative against all incident certificate boxes.
 slope_box,ac_box=bound.derivatives(p)
 checks=viol=0;worst_s=worst_a=0.
 for i in range(len(ts)):
  for j in range(len(aa)):
   cells=[(ii,jj) for ii in set([max(0,i-1),min(i,len(ts)-2)]) for jj in set([max(0,j-1),min(j,len(aa)-2)])]
   for ii,jj in cells:
    k=ii*(len(aa)-1)+jj
    sv=float(slope_box.lo[k]);alo=float(ac_box.lo[k]);ahi=float(ac_box.hi[k]);checks+=1
    worst_s=max(worst_s,sv-float(ht[i,j,0]));worst_a=max(worst_a,float(abs(ha[i,j,0]))-max(abs(alo),abs(ahi)))
    if ht[i,j,0]<sv-1e-8 or abs(ha[i,j,0])>max(abs(alo),abs(ahi))+1e-8:viol+=1
 return {'theta_nodes':81,'A_nodes':25,'nodes':2025,'incident_interval_containment_checks':checks,'violations':viol,'actual_min_Hx_theta_px_per_deg':min_s,'actual_max_abs_Hx_A_px_per_D':max_a,'max_abs_Hx_0_to_4D_px':float(np.max(np.abs(endpoint_shift[...,0]))),'max_Hx_0_to_4D_px':float(np.max(endpoint_shift[...,0])),'min_Hx_0_to_4D_px':float(np.min(endpoint_shift[...,0])),'authorized_shift_limit_px':bound.slope_floor*1.,'minimum_interval_lower_minus_actual_theta_slope':worst_s,'maximum_actual_abs_A_slope_minus_interval_abs_bound':worst_a}
# Dense all 2025 points + interval box certificate for G6 and prior unbounded G7 snapshots.
prev=ST/'results/g7_attempt_01';g7sum=json.loads((prev/'summary.json').read_text());g7name=g7sum['selected_start'];g7npz=np.load(prev/'selected.npz') if (prev/'selected.npz').exists() else None
# Read selected state/global from fit.json to reconstruct parameters where available.
fit=json.loads((prev/g7name/'fit.json').read_text());p_unbounded=np.array(fit['scaled_globals'])
results={'kind':'mechanical preflight; no test suite or campaign fitting','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'parent_G6_attempt02_checkpoint_sha256':__import__('hashlib').sha256((parent/'checkpoint.json').read_bytes()).hexdigest(),'G6_parameter_hash':__import__('hashlib').sha256(p0.tobytes()).hexdigest(),'bound_record_at_G6':bound.record(p0),'G6_dense':dense(p0),'prior_G7_selected_start':g7name,'prior_G7_unbounded_dense':dense(p_unbounded)}
# Gather both G7 parameter snapshots (common and perturbed), if present.
results['prior_G7_start_dense']={}
for name in ['common','perturbed']:
 f=prev/name/'fit.json'
 if f.exists():results['prior_G7_start_dense'][name]=dense(np.asarray(json.loads(f.read_text())['scaled_globals']))
# Actual two rows per exposure.
ids=np.concatenate([np.flatnonzero(e==k)[:2] for k in range(20)])
targets=np.array([[pop['target_theta_deg'][np.flatnonzero(pop['exposure']==k)[0]],pop['demand_diopters'][np.flatnonzero(pop['exposure']==k)[0]]] for k in range(20)])
xc=x0[ids];yc=y[ids];ee=e[ids]
results['actual_rows']={'rows':len(ids),'per_exposure':[int(np.sum(ee==k)) for k in range(20)],'indices':ids.tolist()}
results['proposal']={};proposal_arrays={}
for label,xpmod in [('cpu',np),('gpu',cp)]:
 obj=BoundedJointDM0(spec,yc,ee,cov,targets,xp=xpmod,chunk_size=13 if label=='cpu' else 7,physical_bound=bound)
 x=xpmod.asarray(xc);p=xpmod.asarray(p0);before=obj.evaluate(x,p)['cost'];start=time.perf_counter();xn,pn,event=obj.update(x,p,'joint');
 if xpmod is cp:cp.cuda.Stream.null.synchronize()
 elapsed=time.perf_counter()-start;after=obj.evaluate(xn,pn)['cost'];dx=obj.host(xn)-xc;dp=obj.host(pn)-p0;slack=bound.values(obj.host(pn));proposal_arrays[label]=(obj.host(xn),obj.host(pn))
 results['proposal'][label]={'accepted':event['accepted'],'event':event,'cost_before':float(before),'cost_after':float(after),'elapsed_seconds':elapsed,'dx_inf':float(np.max(np.abs(dx))),'dp_inf':float(np.max(np.abs(dp))),'minimum_linearized_physical_slack':float(np.min(bound.values(p0)+bound.jacobian(p0)@ (obj.host(pn)-p0))),'minimum_actual_physical_slack':float(slack.min()),'feasible_actual':bool(slack.min()>=-1e-10),'state_global_hash':__import__('hashlib').sha256(np.r_[obj.host(xn).ravel(),obj.host(pn)].tobytes()).hexdigest()}
a=results['proposal']['cpu'];b=results['proposal']['gpu'];results['proposal_comparison']={'dx_max_abs_difference':abs(a['dx_inf']-b['dx_inf']),'dp_max_abs_difference':abs(a['dp_inf']-b['dp_inf']),'cost_after_abs_difference':abs(a['cost_after']-b['cost_after']),'accepted_match':a['accepted']==b['accepted'],'state_global_hash_match':a['state_global_hash']==b['state_global_hash'],'states_max_abs_difference':float(np.max(np.abs(proposal_arrays['cpu'][0]-proposal_arrays['gpu'][0]))),'globals_max_abs_difference':float(np.max(np.abs(proposal_arrays['cpu'][1]-proposal_arrays['gpu'][1])))}
(OUT/'preflight.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps({'bound':results['bound_record_at_G6'],'dense_G6':results['G6_dense'],'prior_G7':results['prior_G7_start_dense'],'proposal':results['proposal_comparison'],'proposal_costs':{k:v['cost_after'] for k,v in results['proposal'].items()}},indent=2))
