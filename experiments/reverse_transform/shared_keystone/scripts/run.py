"""Execute the Revision 2 shared calibration stages directly from detections."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

from pipeline import (ROOT,STAGES,host,write,load_inputs,scheduled_manifest,snapshot,
                      initialize,read_npz,warm_start,joint,validation,checkpoint,verify_snapshot)


def fallback_report(out,summary,audit):
    text=['# Shared keystone Revision 2 experiment','',
          f"Run status: **{summary.get('status','UNKNOWN')}**.",'',
          f"Numerical audit status: **{audit.get('status','NOT_RUN')}**.",'',
          'This fits centered triangle shape conditional on empirical gaze anchors. '
          'Accommodation and gaze scales are selected by frozen same-recording regularizers. '
          'A stationary solution is not a global optimum or independent physiological validation.','',
          'All scheduled source identities and unavailable observations are retained in '
          '`scheduled_frames.npz`. Source detections were replayed; archived fitted arrays were not used.','',
          'See `protocol.json`, `provenance.json`, `summary.json`, `frames.npz` and `audit.json` '
          'for settings, immutable source hashes, accepted/rejected proposals and numerical outcomes.','']
    (out/'RESULTS.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase','--stage',type=int,choices=(1,2,3,5),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--no-controls',action='store_true',help='Diagnostic run only; primary Phase III requires controls')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('CuPy GPU required by protocol')
    started=time.perf_counter()
    p,provenance=load_inputs()
    provenance.update(phase=args.phase,source_head=subprocess.check_output(
                      ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      backend='CuPy float64',cupy_version=cp.__version__,numpy_version=np.__version__,
                      command=sys.argv)
    scheduled_manifest(out,p)
    audit=dict(status='NOT_RUN')
    try:
        if args.phase==5:
            summary=validation(p,out,cp,provenance)
            audit=dict(status='VALIDATION_RECORDED',
                       leakage_checks=[r.get('omitted_coordinate_noninterference') for r in summary.get('retained_input_inference',[])],
                       split_is_scheduled_before_completeness=True,
                       training_fit_status=summary.get('training_fit',{}).get('status'))
        elif args.phase==1:
            idx=np.flatnonzero(p['complete_valid'])
            anchors,q,states,summary=initialize(p,idx,out,cp,provenance)
        else:
            if args.parent is None:parser.error('--parent prior stage result directory required')
            prior=read_npz(args.parent/'frames.npz')
            idx=prior['population_index']
            if not np.array_equal(idx,np.flatnonzero(p['complete_valid'])):
                raise ValueError('full-data phase requires exact reviewed complete population')
            if not np.isfinite(prior['A_targets']).all():
                raise ValueError('INITIALIZATION_INVALID: later phases cannot replace invalid targets')
            anchors={k:prior[k] for k in ('B1','B4','theta0','q0','A_targets','W')}
            q,states=prior['q'],prior['states']
            provenance.update(parent_result=str(args.parent.resolve()),
                              parent_frames_sha256=__import__('hashlib').sha256((args.parent/'frames.npz').read_bytes()).hexdigest())
            # Freeze manifest before the first optimization step.
            from pipeline import protocol
            write(out/'protocol.json',protocol(anchors))
            snapshot(out,provenance)
            if args.phase==2:
                q,states,summary=warm_start(p,idx,anchors,q,states,cp)
            else:
                q,states,summary=joint(p,idx,anchors,q,states,cp,out,controls=not args.no_controls)
                if args.no_controls:
                    summary['protocol_completion']='DIAGNOSTIC_ONLY_CONTROLS_OMITTED'
            checkpoint(out,p,idx,anchors,q,states,summary)
        if args.phase!=5 and summary['status']=='INITIALIZATION_INVALID':
            audit=dict(status='INITIALIZATION_INVALID',reason='target validity checks failed; no calibration claimed')
        elif args.phase!=5:
            from audit import audit_run
            audit=audit_run(out,xp=cp,scope={1:'initialization',2:'warm_start',3:'joint'}[args.phase])
        changes=verify_snapshot(out)
        if changes:
            summary['fit_status_before_source_change']=summary['status']
            summary['status']='SOURCE_CHANGED_DURING_RUN'
            summary['source_changes_during_run']=changes
        write(out/'audit.json',audit)
        summary['elapsed_seconds']=time.perf_counter()-started
        write(out/'summary.json',summary)
        try:
            from report import build_report
            build_report(out)
        except ImportError:
            fallback_report(out,summary,audit)
        print(f"Phase {args.phase}: {summary['status']}; audit {audit['status']}; {out}",flush=True)
    except Exception as exc:
        summary=dict(status='RUN_FAILED',error_type=type(exc).__name__,error=str(exc),
                     elapsed_seconds=time.perf_counter()-started,traceback=traceback.format_exc())
        write(out/'summary.json',summary)
        write(out/'audit.json',dict(status='NOT_COMPLETED',reason=str(exc)))
        fallback_report(out,summary,dict(status='NOT_COMPLETED'))
        raise


if __name__=='__main__':main()
