"""G5 composed P4 gaze/accommodation conditional contracts."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from distortion_model.accommodation import DM0Shape,ShapeProfile,shape_derivatives,from_scaled,fit_shape
from distortion_model.optics import keystone,p1_reference
from distortion_model.p1 import P1Model,evaluate_p1


class P4DeformationContracts(unittest.TestCase):
    def setUp(self):
        self.b=np.array([[162.,-96.],[-7.,197.],[-155.,-101.]])
        self.p=np.array([-.17,.001,-.002,.0038]);self.model=from_scaled(self.b,-10.,.36,self.p,True)
        self.theta=np.linspace(-10,10,51);self.a=np.linspace(.5,4.5,51)

    def test_theta_and_mean_derivatives_of_composed_model(self):
        current=shape_derivatives(self.theta,self.a,self.model);h=1e-4
        plus=shape_derivatives(self.theta+h,self.a,self.model);minus=shape_derivatives(self.theta-h,self.a,self.model)
        assert_allclose(current['dtheta'],(plus['edges']-minus['edges'])/(2*h),rtol=3e-6,atol=1e-5)
        assert_allclose(current['dmu_dtheta'],(plus['mu4']-minus['mu4'])/(2*h),rtol=3e-6,atol=1e-5)
        plus=shape_derivatives(self.theta,self.a+h,self.model);minus=shape_derivatives(self.theta,self.a-h,self.model)
        assert_allclose(current['dmu_dA'],(plus['mu4']-minus['mu4'])/(2*h),rtol=3e-6,atol=1e-5)

    def test_keystone_order_and_centered_prediction_mean(self):
        m=1+self.model.m1*(self.a-.36)
        expected=keystone(self.theta+10,m[:,None,None]*self.b,self.model.k4)[0]
        actual=shape_derivatives(self.theta,self.a,self.model)
        assert_allclose(actual['F4'],expected,rtol=1e-10,atol=1e-9)
        wrong=m[:,None,None]*keystone(self.theta+10,self.b,self.model.k4)[0]
        self.assertGreater(np.linalg.norm(wrong-expected),.01)
        centered=actual['F4']-actual['mu4'][:,None,:]
        assert_allclose(centered.mean(axis=1),0.,atol=1e-9)

    def fixture(self):
        e=np.repeat(np.arange(20),10);d=np.repeat(np.repeat([.36,4.,3.,2.],5),10)
        theta=np.tile(np.repeat([-10.,-5.,0.,5.,10.],10),4)
        a=d+np.tile(.015*np.sin(np.arange(10)*2*np.pi/10),20);g=np.linspace(.98,1.02,200)
        observed=g[:,None]*shape_derivatives(theta,a,self.model)['edges']
        return theta,a,g,e,d,observed

    def test_profile_four_global_gradient_with_dynamic_full_means(self):
        theta,a,g,e,d,observed=self.fixture()
        profile=ShapeProfile(theta,observed,g,e,d,d,self.b,-10.,.36,np.eye(4)*.01,deformation=True)
        p=self.p+np.array([-.01,.0002,-.0001,.0003]);out=profile.evaluate(p);h=1e-6
        for j in range(4):
            step=np.eye(4)[j]*h
            numerical=(profile.evaluate(p+step)['cost']-profile.evaluate(p-step)['cost'])/(2*h)
            assert_allclose(out['gradient'][j],numerical,rtol=3e-6,atol=1e-5)
        self.assertTrue(out['inner_certified'])

    def test_synthetic_full_population_global_and_frame_recovery(self):
        theta,a,g,e,d,observed=self.fixture()
        factory=lambda:ShapeProfile(theta,observed,g,e,d,d,self.b,-10.,.36,np.eye(4)*.01,deformation=True)
        model,states,best,outcomes=fit_shape(factory,[[-.15,0.,0.,0.],[-.20,.002,-.001,.002]])
        self.assertTrue(best['conditional_fit_certified']);self.assertEqual(best['conditional_rank'],4)
        assert_allclose(best['scaled_parameters'],self.p,rtol=1e-4,atol=1e-6)
        assert_allclose(states,a,rtol=1e-4,atol=1e-4)
        self.assertEqual(len(outcomes),2)

    def test_prediction_theta_chain_includes_profiled_P1_scale(self):
        p1=P1Model(np.array([[-230.,125.],[17.,-247.],[213.,122.]]),-7.,(1e-4,-2e-4,2e-6))
        observed=p1_reference(self.theta,p1)[2]*1.04
        def prediction(theta):
            g=evaluate_p1(theta,observed,np.eye(4),p1)['g']
            return g[:,None]*shape_derivatives(theta,self.a,self.model)['edges']
        state=evaluate_p1(self.theta,observed,np.eye(4),p1);p4=shape_derivatives(self.theta,self.a,self.model)
        derivative=state['dg_dtheta'][:,None]*p4['edges']+state['g'][:,None]*p4['dtheta']
        h=1e-4
        assert_allclose(derivative,(prediction(self.theta+h)-prediction(self.theta-h))/(2*h),rtol=3e-6,atol=1e-5)


if __name__=='__main__':unittest.main()
