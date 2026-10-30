"""Independent invariants for the reviewer-motivated experiment scripts."""
import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def mod(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
S=mod('revision','67_tim_revision_studies.py')
P=mod('forcing','66_tim_revision_paraemt.py')

def test_alias_orientation_retains_derivative_information():
    R=60;t=np.arange(0,4,1/R)
    ratios=[]
    for f in (29,31):
        th=.001*np.cos(2*np.pi*f*t+.3);fr=-.001*f*np.sin(2*np.pi*f*t+.3)
        ch=dict(voltage_magnitude_pu=np.ones(len(t)),voltage_angle_deg=np.rad2deg(th),frequency_hz=60+fr,rocof_hz_per_s=np.gradient(fr,t))
        result=S.features(t,ch,R,np.array([29.,31.]));ratios.append(result['z'])
        assert abs(result['a']-29)<1e-10
    assert ratios[0][2]>.999 and ratios[1][2]<-.999
    assert np.isclose(np.exp(ratios[0][0]),29,rtol=.001)
    assert np.isclose(np.exp(ratios[1][0]),31,rtol=.001)

def test_reported_finite_difference_response_is_fold_only():
    R=60;a=14.7;t=np.arange(1000)/R;y=np.exp(2j*np.pi*a*t)
    d=(y[2:]-y[:-2])*R/2
    assert np.allclose(d/y[1:-1],1j*R*np.sin(2*np.pi*a/R),atol=1e-10)

def test_chirp_uses_integrated_frequency():
    f=29.;beta=.25;h=1e-6;t=1.75;u=t-.5
    phase=lambda x:2*np.pi*((f-1.25*beta)*x+.5*beta*x*x)
    assert abs((phase(u+h)-phase(u-h))/(4*np.pi*h)-f)<1e-7
    p=P.ChirpProfile(f,beta,.02)
    assert p.value(.4)==1
    assert abs(p.value(t)-(1+.02*np.sin(phase(u))))<1e-12

def test_empty_and_nonfinite_responses_do_not_issue_singletons():
    M=dict(catalogue=np.array([29.,31.]),R=60,eligible=True,ix=[0],m=np.array([[1.],[3.]]),s=np.ones((2,1)),q=.1)
    r=dict(truth=29.,a=29.,support=10.,z=np.array([2.,0,0,0,0,0]))
    assert S.decide(r,M)['outcome']=='outside'
    r['z'][0]=np.nan
    assert S.decide(r,M)['outcome']=='insufficient'

def test_empirical_rank_edge_and_disjoint_alias_restriction():
    rows=[]
    for role in ('design','calibration'):
        for f in (29.,31.):
            for k in range(4):rows.append(dict(role=role,truth=f,z=np.array([f,0.,np.sign(30-f),0.,0.,0.])))
    M=S.fit(rows,60,np.array([29.,31.]))
    assert np.isinf(M['q'])  # ceil((8+1)*.9)=9 > 8
    M['q']=.1
    r=dict(truth=31.,a=29.,support=10.,z=np.array([29.,0.,1.,0.,0.,0.]))
    assert S.decide(r,M)['incorrect']
