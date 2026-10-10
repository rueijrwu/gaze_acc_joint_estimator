"""Inspect the saved adjacent interval-bound branches; perform no optimization."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run import context,write,digest,load_npz


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    attempt=args.attempt.resolve();output=args.output.resolve()
    candidate,_,_,_,_,_,_,bound=context(attempt,'adjacent')
    summary=json.loads((candidate/'summary.json').read_text())
    source=candidate/summary['selected_start']/'solution.npz';p=load_npz(source)['scaled_globals']
    # Declare native alpha4 values before evaluating their bound branches.
    schedule=[float(p[6]/100),-1e-8,-1e-9,-1e-10,0.,1e-10,1e-9,1e-8]
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(Path(__file__),output/'source.py')
    write(output/'config.json',{'kind':'saved-model interval-bound branch inspection; no state fitting or held inference',
        'alpha4_schedule_native':schedule,'all_other_globals_and_states_frozen':True,
        'solution_sha256':digest(source),'checkpoint_sha256':digest(candidate/'checkpoint.json'),
        'script_sha256':digest(Path(__file__)),'same_numeric_physical_bound':True})
    records=[];cells=len(bound.theta.lo)
    for alpha in schedule:
        q=p.copy();q[6]=100*alpha;v=bound.values(q);d=bound.branch_derivatives(q)
        floor=v[:cells];row=int(np.argmin(floor));near=np.flatnonzero(floor<=1e-7)
        records.append({'alpha4_native':alpha,'minimum_actual_normalized_slack':float(v.min()),
            'gaze_floor_minimum':float(floor.min()),'gaze_floor_minimum_row':row,
            'floor_rows_0_and_23':[float(floor[i]) for i in [0,23]],
            'floor_rows_0_and_23_native_alpha4_derivatives':[float(100*d['jacobian'][i,6]) for i in [0,23]],
            'selected_minimum_native_alpha4_derivative':float(100*d['jacobian'][row,6]),
            'selected_minimum_nonsmooth':bool(d['nonsmooth'][row]),
            'near_active_floor_rows':near.tolist(),'nonsmooth_near_active_floor_rows':near[d['nonsmooth'][near]].tolist()})
    if digest(source)!=json.loads((output/'config.json').read_text())['solution_sha256']:
        raise ValueError('saved candidate changed during inspection')
    write(output/'summary.json',{'records':records,'saved_candidate_unchanged':True,'optimization_performed':False,
        'inference_performed':False,'physical_optimum_or_generalized_certificate_claimed':False})


if __name__=='__main__':main()
