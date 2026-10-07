"""Audit regressions: recorded high-residual candidates and integration contracts."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from full_position.model import PositionModel, SummaryModel, STATE_SCALE
from full_position.geometry import context
from full_position.invert import invert, predict_holdout, objective_derivatives
from full_position.noise import whitening, marginal
from full_position.schema import artifact, load_model, write_json
from full_position.validate import summarize
from full_position.report import generate
from full_position.apply import apply
from full_position.profile import coordinate_polynomials
from full_position.ablation import DerivedSummaryModel
from test_full_position import synthetic_model


FIXTURE = json.loads((Path(__file__).parent/'fixtures/audit_inverse_cases.json').read_text())


class Audit(unittest.TestCase):
    def test_exact_hessians_and_coordinate_polynomials(self):
        p = np.array(FIXTURE['cases'][0]['p'])
        r = context(p).r
        x = np.array([-11., .96])
        for model in [synthetic_model(27), synthetic_model(37), SummaryModel(np.arange(13)/100),
                      DerivedSummaryModel(synthetic_model(37))]:
            finite = approx_derivative(lambda x: model.predict(x,r,True)[1].ravel(),x).reshape(model.channels,2,2)
            np.testing.assert_allclose(model.state_hessian(x,r),finite,atol=1e-9,rtol=1e-6)
        model = PositionModel(37,FIXTURE['coefficients'])
        ix = np.array([0,1,4,5])
        cov = marginal(np.array(FIXTURE['cases'][0]['covariance']),ix)
        y = np.arange(4)/10
        W = whitening(cov)
        z = x/STATE_SCALE
        _, _, H = objective_derivatives(model,r,y,W,ix,z)
        finite = approx_derivative(lambda z: objective_derivatives(model,r,y,W,ix,z)[1],z)
        np.testing.assert_allclose(H,finite,atol=1e-2,rtol=1e-7)
        coef = coordinate_polynomials(model,r,x[1])
        for t in [-2.,-.47,0.,1.8]:
            np.testing.assert_allclose(np.polynomial.polynomial.polyval(t,coef),model.predict([10*t,x[1]],r),atol=1e-12)

    def test_recorded_lower_candidate_is_retained_and_certified(self):
        model = PositionModel(37,FIXTURE['coefficients'])
        case = FIXTURE['cases'][0]
        ctx = context(np.array(case['p']))
        v = ((np.array(case['q'])-ctx.c)/ctx.ell).ravel()
        result = predict_holdout(model,ctx,1,v[[0,1,4,5]],np.array(case['covariance']))
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['cost'],24441.6966455,places=4)
        np.testing.assert_allclose(result['state'],[-11.00017615,.96126556],atol=1e-5)
        self.assertLess(result['stationarity_encoded'],1e-4)
        self.assertEqual(len(result['candidates']),49)
        self.assertTrue(any(c['accepted'] and not c['initial']['accepted_before_polish'] for c in result['candidates']))
        self.assertTrue(all('state' in c and 'cost' in c and 'termination' in c['initial'] for c in result['candidates']))

    def test_second_recorded_full_inverse_and_common_support(self):
        from full_position.reevaluate import compare
        case = FIXTURE['cases'][1]
        ctx = context(np.array(case['p']))
        v = ((np.array(case['q'])-ctx.c)/ctx.ell).ravel()
        result = invert(PositionModel(37,FIXTURE['coefficients']),ctx.r,v,np.array(case['covariance']))
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['cost'],31026.2880739,places=4)
        self.assertLess(result['stationarity_encoded'],1e-4)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for base in ['old','new']:
                for name in ['conditional27','conditional37','two_channel13']:
                    rows=[dict(fixation=0,row=i,capture='test',baseline_valid=True,
                        estimated=(i==1 or name=='conditional27'),theta=float(i),A=2.,
                        nominal_theta=0.,demand=2.,cost=1.,rank=2) for i in range(2)]
                    write_json(folder/base/'gaze_+0'/name/'frames.json',rows)
            comparison=compare(folder/'old',folder/'new')
            common=comparison['gaze/common_state_support']
            self.assertEqual(common['count'],1)
            for summary in common['models'].values():
                self.assertEqual(summary['fixation_means'][0]['rows'],[1])

    def test_bounded_polish_budget_recovery_and_indefinite_candidate(self):
        model = PositionModel(27,np.zeros(27))
        model.beta[2],model.beta[9] = 1.,1.
        model.beta[11],model.beta[23] = 1.,1.
        r = context(np.array(FIXTURE['cases'][0]['p'])).r
        y = model.predict([25.,8.],r)
        result = invert(model,r,y,np.eye(6),starts=np.array([[0.,1.]]),max_nfev=1)
        self.assertTrue(result['available'])
        np.testing.assert_allclose(result['state'],[20.,6.],atol=1e-5)
        self.assertEqual(result['candidates'][0]['initial']['termination'],'budget_exhausted')
        self.assertTrue(result['branches'][0]['certificate']['local_minimum'])
        model.beta[2] = 0.
        model.beta[4] = 1.
        y = model.predict([10.,2.],r)
        result = invert(model,r,y,np.eye(6),starts=np.array([[0.,2.]]))
        self.assertTrue(result['available'])
        self.assertAlmostEqual(abs(result['state'][0]),10.,places=4)

    def test_schema_rejects_incompatible_conventions(self):
        obj = artifact(synthetic_model(37),synthetic_model(27),[0.,2.],np.eye(12),
                       dict(correspondence={}),dict(converged=True))
        changes = dict(normalization='rotated',coordinate_order=['x','y'],coefficient_order='different',
            bounds_physical=[[-10,0],[10,6]],pilot_model='conditional37',pilot_coefficients=[1.],
            coordinate_covariance=np.zeros((12,12)).tolist(),covariance_semantics='observed_P4_weights',
            reference_state=[0.,-1.])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.json'
            for key,value in changes.items():
                changed = copy.deepcopy(obj); changed[key] = value
                write_json(path,changed)
                with self.assertRaises(ValueError,msg=key):
                    load_model(path)
            write_json(path,obj); load_model(path)

    def test_frame_and_fixation_aggregation_and_selected_cost(self):
        rows = [dict(fixation=g,row=i,capture='test',baseline_valid=True,estimated=True,
            theta=theta,A=2.,nominal_theta=0.,demand=2.,cost=1.,rank=2)
            for i,(g,theta) in enumerate([(0,1.),(0,-1.),(1,3.)])]
        summary = summarize(rows,[])
        self.assertAlmostEqual(summary['frame_gaze_anchor_discrepancy_deg']['rms'],np.sqrt(11/3))
        self.assertAlmostEqual(summary['fixation_mean_gaze_anchor_discrepancy_deg']['rms'],np.sqrt(9/2))
        self.assertEqual(summary['fixation_means'][0]['count'],2)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            calibration = dict(selected_start=1,alternatives=[dict(cost=.1,converged=False),dict(cost=4.,converged=True)])
            write_json(folder/'summary.json',{'fold/conditional27':dict(calibration_converged=False,calibration=calibration)})
            write_json(folder/'fold/conditional27/training_states.json',dict(states=[[0.,2.]],groups=[0],nominal_anchors=[[0.,2.]]))
            metrics,_ = generate(folder)
            self.assertEqual(metrics[0]['calibration_cost'],4.)
            self.assertEqual(metrics[0]['best_rejected_start_cost'],.1)

    def test_raw_application_holdout_noninterference_and_diagnostics(self):
        model,pilot = synthetic_model(37),synthetic_model(27)
        p = np.array(FIXTURE['cases'][0]['p'])
        ctx = context(p)
        q = ctx.c+ctx.ell*model.predict([3.,2.],ctx.r).reshape(3,2)
        provenance = dict(correspondence={},anchor_range=[[-10,0],[10,6]],
            conditional_context_support=dict(r_min=(ctx.r-.1).tolist(),r_max=(ctx.r+.1).tolist(),
                                             signed_P1_area_branches=[float(np.sign(ctx.signed_area))]))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path = folder/'model.json'
            write_json(model_path,artifact(model,pilot,[0.,2.],np.eye(12)*.01,provenance,dict(converged=True)))
            write_json(folder/'training_states.json',dict(states=[[0.,1.],[5.,3.]]))
            predictions = []
            for i in range(2):
                changed = q.copy(); changed[1] += i*np.array([500.,-400.])
                payload = dict(meta=dict(pair_index=[2,1,0]),arrays=dict(p1_xy=p[None],p4_xy=changed[::-1][None],
                    p4_found=np.ones((1,3),bool),frame_index=np.array([0]),timestamp_ms=np.array([0.])))
                capture = folder/f'input{i}.pkl'; capture.write_bytes(pickle.dumps(payload))
                apply(model_path,capture,folder/f'out{i}',holdouts=True)
                record = json.loads((folder/f'out{i}/frames.jsonl').read_text())
                predictions.append(record['holdouts'][1]['predictions'])
                for key in ['context_outside_training_extrema','state_outside_empirical_training_range',
                            'state_outside_anchor_range','p1_parity_seen_in_training','state_covariance_physical']:
                    self.assertIn(key,record['diagnostics'])
            self.assertEqual(predictions[0],predictions[1])


if __name__ == '__main__':
    unittest.main()
