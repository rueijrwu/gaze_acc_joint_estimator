"""DM0 accommodation shape and full-mean anchor numerical contracts."""
import unittest
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from distortion_model.accommodation import (DM0Shape,shape_derivatives,shape_covariance,scalar_projection,
                                           edges,ShapeProfile,fit_shape,from_scaled,parameter_bounds,domain_margins)
from distortion_model.geometry import relative_covariance


class AccommodationContracts(unittest.TestCase):
    def setUp(self):
        self.b=np.array([[162.,-96.],[-7.,197.],[-155.,-101.]])
        self.model=DM0Shape(self.b,-10.,.36,-.017,(1e-5,-2e-5,2e-6))
        self.theta=np.linspace(-10,10,51);self.a=np.linspace(.36,4.,51)

    def close(self,a,b):assert_allclose(a,b,rtol=1e-10,atol=1e-9)

    def test_shape_marginal_matches_native_correlated_edges(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(12,12));sigma=x@x.T+np.eye(12)
        transform=np.zeros((4,12));transform[:2,6:8]=-np.eye(2);transform[:2,8:10]=np.eye(2)
        transform[2:,6:8]=-np.eye(2);transform[2:,10:12]=np.eye(2)
        self.close(shape_covariance(relative_covariance(sigma)),transform@sigma@transform.T)

    def test_scalar_projection_preserves_remaining_direction_and_magnitude(self):
        observed=np.stack([self.b,1.2*self.b]);observed[1,0,1]+=.5
        rho,residual=scalar_projection(observed,self.b)
        self.close(observed,rho[:,None,None]*self.b+residual)
        self.close(np.sum(residual*self.b,axis=(1,2)),[0.,0.]);self.assertGreater(rho[1],1.19)
        self.assertGreater(np.linalg.norm(residual[1]),.1)

    def test_baseline_identity_and_native_zero_for_all_A(self):
        f=shape_derivatives(np.full(51,-10.),self.a,self.model)
        self.close(f['F4'],(1+self.model.m1*(self.a-.36))[:,None,None]*self.b)
        self.close(shape_derivatives(-10.,.36,self.model)['F4'],self.b)

    def test_A_first_second_and_global_composition_derivatives(self):
        p=np.array([-.17,.001,-.002,.0038]);model=from_scaled(self.b,-10.,.36,p,True)
        f=shape_derivatives(self.theta,self.a,model);h=1e-4
        plus=shape_derivatives(self.theta,self.a+h,model);minus=shape_derivatives(self.theta,self.a-h,model)
        assert_allclose(f['dA'],(plus['edges']-minus['edges'])/(2*h),rtol=3e-6,atol=1e-5)
        assert_allclose(f['dAA'],(plus['dA']-minus['dA'])/(2*h),rtol=3e-6,atol=1e-5)
        for j in range(4):
            step=np.eye(4)[j]*1e-6
            plus=shape_derivatives(self.theta,self.a,from_scaled(self.b,-10.,.36,p+step,True))
            minus=shape_derivatives(self.theta,self.a,from_scaled(self.b,-10.,.36,p-step,True))
            assert_allclose(f['dglobal'][...,j],(plus['edges']-minus['edges'])/2e-6,rtol=3e-6,atol=1e-5)

    def fixture(self,window=False):
        exposure=np.repeat(np.arange(4),30);demand=np.repeat([.36,2.,3.,4.],30)
        a=demand+np.tile(.02*np.sin(np.arange(30)*2*np.pi/30),4)
        theta=np.full(120,-10.);g=np.linspace(.98,1.02,120)
        data=g[:,None]*shape_derivatives(theta,a,DM0Shape(self.b,-10.,.36,-.017))['edges']
        kwargs={'data_indices':np.r_[np.arange(20),np.arange(30,50),np.arange(60,80),np.arange(90,110)]} if window else {}
        return theta,data,g,exposure,demand,demand.copy(),a,kwargs

    def test_full_coupled_state_gradient_and_anchor_objective(self):
        theta,data,g,e,d,initial,a,kwargs=self.fixture(True)
        profile=ShapeProfile(theta,data,g,e,d,initial,self.b,-10.,.36,np.eye(4),**kwargs)
        current=a[profile.indices]+.01;cost,gradient,*_=profile.terms([-.17],current);h=1e-5
        for j in [0,25,50,75]:
            delta=np.eye(len(current))[j]*h
            numerical=(profile.terms([-.17],current+delta)[0]-profile.terms([-.17],current-delta)[0])/(2*h)
            assert_allclose(gradient[j],numerical,rtol=3e-6,atol=1e-5)
        full=initial.copy();full[profile.indices]=current
        means=np.bincount(e,weights=full)/np.bincount(e)
        residual=data[profile.indices]-g[profile.indices,None]*edges(self.b)*(1-.017*(current-.36))[:,None]
        expected=.5*np.sum(profile.weights*np.sum(residual**2,axis=1))+.5/4/.25**2*np.sum((means-[.36,2,3,4])**2)
        self.close(cost,expected)

    def test_profile_envelope_gradient_with_bounded_dynamic_states(self):
        theta,data,g,e,d,initial,a,kwargs=self.fixture()
        profile=ShapeProfile(theta,data,g,e,d,initial,self.b,-10.,.36,np.eye(4)*.01)
        p=np.array([-.19]);o=profile.evaluate(p);h=1e-6
        plus=profile.evaluate(p+h)['cost'];minus=profile.evaluate(p-h)['cost']
        assert_allclose(o['gradient'],[(plus-minus)/(2*h)],rtol=3e-6,atol=1e-5)
        self.assertTrue(o['inner_certified']);self.assertGreater(np.ptp(o['a'][:30]),.02)

    def test_synthetic_slope_and_frame_A_recovery(self):
        theta,data,g,e,d,initial,a,kwargs=self.fixture()
        factory=lambda:ShapeProfile(theta,data,g,e,d,initial,self.b,-10.,.36,np.eye(4)*.01)
        model,states,best,outcomes=fit_shape(factory,[[-.12],[-.25]])
        self.assertTrue(best['conditional_fit_certified']);self.assertEqual(best['conditional_rank'],1)
        assert_allclose(model.m1,-.017,rtol=1e-6,atol=1e-8)
        assert_allclose(states,a,rtol=1e-5,atol=1e-5)
        self.assertEqual(len(outcomes),2)

    def test_near_window_keeps_outside_states_and_full_mean_anchor(self):
        theta,data,g,e,d,initial,a,kwargs=self.fixture(True)
        profile=ShapeProfile(theta,data,g,e,d,initial,self.b,-10.,.36,np.eye(4),**kwargs)
        full=profile.full_states([-.17]);mask=np.ones(len(full),bool);mask[profile.indices]=False
        assert_array_equal(full[mask],initial[mask]);self.assertGreater(np.ptp(full[:20]),.01)

    def test_bounds_cover_shifted_full_domain_and_baseline_states(self):
        lower,upper=parameter_bounds(self.b,-10.,.36,True)
        for p in [lower,upper,(lower+upper)/2]:self.assertTrue(domain_margins(from_scaled(self.b,-10.,.36,p,True))['valid_domain'])
        f=shape_derivatives(np.array([-21.,0.,21.]),np.array([2.,-1.,7.]),self.model)
        self.assertFalse(f['valid'].any())
        self.assertFalse(self.model.b4.flags.writeable)

    def test_zero_slope_has_no_shape_A_information(self):
        theta,data,g,e,d,initial,a,kwargs=self.fixture()
        profile=ShapeProfile(theta,data,g,e,d,initial,self.b,-10.,.36,np.eye(4))
        out=profile.evaluate([0.]);self.close(out['a'],d)
        self.close(shape_derivatives(theta,d,DM0Shape(self.b,-10.,.36,0.))['dA'],np.zeros((120,4)))


if __name__=='__main__':unittest.main()
