"""Controlled waveform stress and native ParaEMT waveform replay, 60 frames/s."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,resample_poly
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('study',ROOT/'scripts/67_tim_revision_studies.py'); S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
from pmu_sso.config import SyntheticPMUChainConfig
from pmu_sso.simulation.research_estimators import estimate_research_pmu_channels
OUT=S.OUT;FS=4800;R=60
CHAINS={
 'SDFT-pre':dict(estimator_kind='sliding_dft',frequency_stage='predecimation_phase_derivative',window_cycles=1,window_type='rectangular'),
 'TF-p2':dict(estimator_kind='taylor_fourier',frequency_stage='taylor_phasor_derivative',window_cycles=1,window_type='rectangular',taylor_order=2,taylor_ridge_relative=1e-6),
 'SDFT-reported':dict(estimator_kind='sliding_dft',frequency_stage='predecimation_phase_derivative',window_cycles=1,window_type='rectangular')}

def waveform(f,seed,beta=0.,snr=40):
    rng=np.random.default_rng(seed);t=np.arange(3*FS)/FS
    # Frequency f is defined at the 1.75-s analysis centre.
    psi=2*np.pi*(f*t+.5*beta*(t-1.75)**2)+rng.uniform(-np.pi,np.pi)
    phase=np.deg2rad([.04,.08,.12][seed%3])*np.sin(psi)
    mag=1+[.0005,.0015,.003][seed%3]*np.sin(psi+rng.uniform(-np.pi,np.pi))
    rot=np.deg2rad([0,-120,120]);base=np.sqrt(2)*np.cos(2*np.pi*60*t[:,None]+rot)
    v=np.sqrt(2)*mag[:,None]*np.cos(2*np.pi*60*t[:,None]+phase[:,None]+rot)
    # Noise is scaled to waveform modulation residual, not the nominal carrier.
    sigma=np.sqrt(np.mean((v-base)**2))*10**(-snr/20)
    return v+rng.normal(0,sigma,v.shape)

def replay(v,name,processing='reference',delay=0.):
    if processing=='filtered':
        # Declared synthetic front end, not a certified transformer model.
        v=sosfilt(butter(2,600,fs=FS,output='sos'),v,axis=0)
    cfg=SyntheticPMUChainConfig(chain_id=name,**CHAINS[name])
    est=estimate_research_pmu_channels(v,sample_rate_hz=FS,nominal_frequency_hz=60,report_rate_hz=FS,chain=cfg)
    t=est.timestamp_s;c={n:np.asarray(est.channels[n],float).copy() for n in S.NAMES}
    c[S.NAMES[1]]=np.rad2deg(np.unwrap(np.deg2rad(c[S.NAMES[1]])))
    if processing=='filtered':
        sos=butter(4,.4*R,fs=FS,output='sos')
        for name0 in S.NAMES[:3]:
            # Steady initial value avoids an artificial 60-Hz frequency step.
            x=c[name0];c[name0]=sosfilt(sos,x-x[0])+x[0]
    target=np.arange(0,3,1/R)
    ch={n:np.interp(target-(delay if n==S.NAMES[2] else 0),t,c[n]) for n in S.NAMES[:3]}
    if name=='SDFT-reported':ch[S.NAMES[2]]=60+np.gradient(np.deg2rad(ch[S.NAMES[1]]),target)/(2*np.pi)
    ch[S.NAMES[3]]=np.gradient(ch[S.NAMES[2]],target)
    mask=(target>=.75-1e-12)&(target<=2.75+1e-12)
    return S.features(target[mask],{n:v[mask] for n,v in ch.items()},R)

def controlled(pilot=False):
    rows=[];n=2 if pilot else 12
    protocol=OUT/('waveform_pilot_protocol.json' if pilot else 'waveform_protocol.json')
    if not protocol.exists():protocol.write_text(json.dumps(dict(catalogue=S.F.tolist(),replicates=n,split_seed_offsets=[0,10000,20000],R=R,fs=FS,processing=['reference','filtered'],front_end='causal Butterworth order2 cutoff600Hz',internal_antialias='causal Butterworth order4 cutoff24Hz both angle and frequency',delay_test_s=.002,offsets=[-.5,-.25,-.1,-.05,.05,.1,.25,.5],drifts=[-.5,-.25,-.05,.05,.25,.5],snr=[40,20,0],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameter_ranges='declared synthetic stresses, not commercial PMU specifications'),indent=2))
    for si,role in enumerate(('design','calibration','test')):
        for fi,f in enumerate(S.F):
            for rep in range(n):
                seed=10000*si+100*fi+rep+700
                v=waveform(f,seed)
                for name in CHAINS:
                    for proc in ('reference','filtered'):
                        rows.append(dict(role=role,truth=f,seed=seed,chain=name,processing=proc,**replay(v,name,proc)))
        print('waveform split',role,flush=True)
    models={};design=[];results=[]
    for name in CHAINS:
        for proc in ('reference','filtered'):
            M=S.fit([r for r in rows if r['chain']==name and r['processing']==proc],R)
            models[(name,proc)]=M
            design.append(dict(chain=name,processing=proc,**{k:v for k,v in M.items() if k not in ('m','s','catalogue')}))
            for r in rows:
                if r['role']=='test' and r['chain']==name and r['processing']==proc:
                    for cal in ([proc] if proc=='reference' else [proc,'reference']):
                        results.append(dict(chain=name,processing=proc,calibration=cal,kind='nominal',level=0,seed=r['seed'],truth=r['truth'],**S.decide(r,models[(name,cal)])))
    pd.DataFrame(design).to_csv(OUT/'waveform_design.csv',index=False)
    # All stresses are test-only; models remain fixed.
    tests=[('offset',x) for x in (-.5,-.25,-.1,-.05,.05,.1,.25,.5)]+[('drift',x) for x in (-.5,-.25,-.05,.05,.25,.5)]+[('noise',x) for x in (20,0)]+[('delay',.002)]
    for kind,x in (tests[:1] if pilot else tests):
        for fi,f in enumerate(S.F):
            for rep in range(4):
                seed=20000+100*fi+rep+700
                v=waveform(f+(x if kind=='offset' else 0),seed,beta=x if kind=='drift' else 0,snr=x if kind=='noise' else 40)
                for name in CHAINS:
                    for proc in ('reference','filtered'):
                        rr=dict(truth=f+(x if kind=='offset' else 0),**replay(v,name,proc,delay=x if kind=='delay' else 0))
                        # Drift is not a stationary catalogue truth even if its midpoint equals f.
                        if kind=='drift':rr['truth']=float('nan')
                        d=S.decide(rr,models[(name,proc)])
                        results.append(dict(chain=name,processing=proc,calibration=proc,kind=kind,level=x,seed=seed,truth=rr['truth'],nominal=f,**d))
        print('waveform stress',kind,x,flush=True)
    df=pd.DataFrame(results);df.to_csv(OUT/'waveform_decisions.csv',index=False)
    S.summary(df,['chain','processing','calibration','kind','level'],OUT/'waveform_summary.csv')
    # Models are fitted deterministically; no pickle output is distributed.
    print('waveform study complete',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);controlled(a.pilot)
if __name__=='__main__':main()
