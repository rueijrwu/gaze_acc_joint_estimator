"""Preserve every scheduled omission when an uncertified calibration forbids G8."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run import ROOT,write,digest,load_npz,load_module,S5,array_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--candidate',choices=['control','adjacent'],required=True)
    args=parser.parse_args();attempt=args.attempt.resolve();parent=attempt/args.candidate
    summary=json.loads((parent/'summary.json').read_text())
    if summary['fit_certified']:raise ValueError('this ledger is only for uncertified calibrations')
    output=attempt/('g8_'+args.candidate);output.mkdir(parents=True,exist_ok=False)
    pop=load_npz(attempt/'population.npz');runner=load_module('uninferred_g8_allocation',S5/'scripts/run.py');out=runner.allocate(pop)
    counts=[]
    for held in range(3):
        keep=[j for j in range(3) if j!=held]
        retained=pop['p1_available'].all(axis=1)&pop['p4_available'][:,keep].all(axis=1)
        out['retained_input_valid'][:,held]=retained
        out['held_input_valid'][:,held]=pop['p4_available'][:,held]&pop['p1_available'].all(axis=1)
        out['status'][retained,held]='not_run_uncertified_calibration'
        out['reason'][retained,held]='calibration_stationarity_gate'
        out['score_status'][retained,held]='not_inferred'
        counts.append({'held_point':held,'scheduled':len(retained),'retained_available':int(retained.sum()),
            'input_unavailable':int((~retained).sum()),'inference_attempted':0,'scored':0})
    np.savez_compressed(output/'not_run_roster.npz',**out)
    write(output/'summary.json',{'status':'NOT_RUN_UNCERTIFIED_CALIBRATION','crosscheck_complete':False,
        'scheduled_rows':len(pop['row']),'scheduled_slots':3*len(pop['row']),'by_held_point':counts,
        'inference_attempted':0,'scored':0,'primary_metrics':None,'no_optical_ranking':True,
        'model_sha256':digest(parent/'model.json'),'parent_checkpoint_sha256':digest(parent/'checkpoint.json'),
        'population_sha256':digest(attempt/'population.npz'),'roster_array_hash':array_hash(out),
        'script_sha256':digest(Path(__file__))})


if __name__=='__main__':main()
