"""Stage-three independent reference and dual-zero numerical contracts."""
from dataclasses import replace
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from distortion_model.p4 import (DualZeros, corrected_p4_pattern, p4_balance,
                                 select_shared_reference, near_reference)
from distortion_model.p1 import P1Model, evaluate_p1
from distortion_model.optics import Parameters, p1_reference, p4_reference, keystone


class P4ReferenceContracts(unittest.TestCase):
    def setUp(self):
        self.b = np.array([[-230.,125.],[17.,-247.],[213.,122.]])
        self.zeros = DualZeros(-7.3, 4.2)

    def close(self, a, b):
        assert_allclose(a,b,rtol=1e-10,atol=1e-9)

    def test_dual_zero_conversion_and_visual_roundtrip_framewise(self):
        theta=np.linspace(-20,20,1001)
        x1,x4=self.zeros.angles(theta)
        self.close(x1,x4+self.zeros.delta14)
        self.close(self.zeros.p1_from_p4(x4),x1)
        self.close(self.zeros.visual_from_p4(x4),theta)
        self.close(x1+self.zeros.omega1,theta)
        self.close(x4+self.zeros.omega4,theta)
        self.close(self.zeros.angles(self.zeros.omega4),[self.zeros.delta14,0.])

    def test_native_identities_and_p1_at_p4_zero(self):
        p=Parameters(self.b,self.b*.7,-7.3,4.2,(1e-4,-2e-4,2e-6),
                     (2e-4,1e-4,-1e-6),.36,.03,np.zeros((6,2)))
        f1=p1_reference(p.omega1,p)[0];self.close(f1,p.b1)
        for a in [.36,2.,4.]:
            f4,_,valid=p4_reference(p.omega4,a,p)
            self.assertTrue(valid);self.close(f4,(1+p.m1*(a-p.aref))*p.b4)
        at4=p1_reference(p.omega4,p)[0]
        expected=keystone(p.omega4-p.omega1,p.b1,p.k1)[0]
        self.close(at4,expected);self.assertGreater(np.linalg.norm(at4-p.b1),.01)
        self.assertGreater(np.linalg.norm(p4_reference(0.,2.,p)[0]-(1+p.m1*(2-p.aref))*p.b4),.01)

    def test_reference_reexpression_transforms_operator_and_baseline_together(self):
        coefficients=(1e-4,-2e-4,2e-6)
        def H(x):
            a,b,g=coefficients
            return np.array([[1+a*x*x,0.,0.],[0.,1+b*x*x,0.],[0.,g*x,1.]])
        def apply(h,points):
            homogeneous=np.column_stack((points,np.ones(3)))@h.T
            return homogeneous[:,:2]/homogeneous[:,2:]
        delta=self.zeros.delta14;reference=apply(H(delta),self.b)
        for x4 in [-10.,0.,10.]:
            transfer=H(x4+delta)@np.linalg.inv(H(delta))
            self.close(apply(transfer,reference),keystone(x4+delta,self.b,coefficients)[0])
        self.close(H(delta)@np.linalg.inv(H(delta)),np.eye(3))

    def test_p4_reference_change_does_not_change_p1_scale_at_fixed_visual_gaze(self):
        theta=np.linspace(-10,10,31);p1=P1Model(self.b,-7.3,(1e-4,-2e-4,2e-6))
        edges=p1_reference(theta,p1)[2]*1.08
        original=evaluate_p1(theta,edges,np.eye(4),p1)['g']
        for omega4 in [-12.,0.,7.]:
            zeros=DualZeros(p1.omega1,omega4)
            reconstructed_theta=zeros.visual_from_p4(zeros.angles(theta)[1])
            self.close(evaluate_p1(reconstructed_theta,edges,np.eye(4),p1)['g'],original)

    def test_corrected_pattern_keeps_p4_magnification_and_correspondence(self):
        points=np.stack([self.b+[300.,700.],2*self.b+[700.,200.]])
        corrected,valid=corrected_p4_pattern(points,np.array([1.,1.]))
        self.assertTrue(valid.all());self.close(corrected[1],2*corrected[0])
        restored,_=corrected_p4_pattern(points,np.array([1.,2.]))
        self.close(restored[1],restored[0])
        reversed_shape,_=corrected_p4_pattern(points[:,[2,1,0]],np.ones(2))
        self.close(reversed_shape,corrected[:,[2,1,0]])

    def test_balance_scale_invariance_translation_and_no_affine_alignment(self):
        points=np.stack([self.b+[300.,400.],1.4*self.b+[-100.,200.]])
        s,c,v=p4_balance(points,np.array([.9,1.3]))
        self.assertTrue(v.all());self.close(s[0],s[1]);self.close(c[0],c[1])
        modified=points.copy();modified[:,1,0]+=30
        self.assertGreater(np.linalg.norm(p4_balance(modified,np.ones(2))[0]-s),.01)

    def test_invalid_scale_and_missing_or_collapsed_shape_explicit(self):
        points=np.tile(self.b,(5,1,1));points[3,0,0]=np.nan;points[4]=1
        corrected,valid=corrected_p4_pattern(points,np.array([1.,0.,-1.,1.,1.]))
        assert_array_equal(valid,[True,False,False,False,True])
        assert_array_equal(p4_balance(points,np.array([1.,0.,-1.,1.,1.]))[2],[True,False,False,False,False])
        self.assertTrue(np.isnan(corrected[1:4]).all())
        with self.assertRaises(ValueError):corrected_p4_pattern(points,np.ones(4))
        with self.assertRaises(ValueError):DualZeros(np.nan,0.)

    def test_selection_equal_capture_independent_of_frame_count_and_p1_zero(self):
        records=[]
        for c in range(4):
            for j,t in enumerate([-10.,-5.,0.,5.,10.]):
                records.append({'capture':f'capture_{c+1}', 'nominal_target_deg':t,
                                'exposure':5*c+j,'valid':10*(c+1),'score':{'mean':abs(t-5)/100+.01*c}})
        chosen,profiles=select_shared_reference(records)
        self.assertEqual(chosen['nominal_target_deg'],5.);self.assertEqual(len(profiles),5)
        for r in records:r['valid']*=1000
        self.assertEqual(select_shared_reference(records)[0],chosen)
        records[0]['nominal_target_deg']=-5.
        with self.assertRaises(ValueError):select_shared_reference(records)

    def test_near_reference_window_keeps_per_frame_angles(self):
        theta=np.array([1.,3.,4.2,5.,8.,np.nan]);zeros=DualZeros(-7.3,4.2)
        assert_array_equal(near_reference(theta,zeros),[False,True,True,True,False,False])
        x1,x4=zeros.angles(theta);self.assertGreater(np.ptp(x4[1:4]),1.)
        with self.assertRaises(ValueError):near_reference(theta,zeros,0.)


if __name__=='__main__':unittest.main()
