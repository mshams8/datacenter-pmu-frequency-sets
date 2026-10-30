"""Native saved abc replay and new ParaEMT stress evaluation; no carrier synthesis."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
W=module('waveform_study',ROOT/'scripts/68_tim_waveform_study.py');S=W.S
P=module('paraemt_study',ROOT/'scripts/66_tim_revision_paraemt.py');ChirpProfile=P.ChirpProfile
P.OUT=ROOT/'data/native'

def extract(path,bus):
    with np.load(path,allow_pickle=False) as d:
        assert int(d['source_bus'])==int(bus)
        t=d['time_s'];v=d['voltage_phasor_pu'];freq=d['pll_frequency_hz'];abc=d['voltage_abc_pu']
    assert np.isfinite(abc).all() and np.isfinite(freq).all()
    return None,t,v,freq,resample_poly(abc[:-1],24,5,axis=0)


def filtered_features(t,v,freq,cycle,R):
    L=round(cycle*1000/60);vv=uniform_filter1d(v.real,L,mode='nearest')+1j*uniform_filter1d(v.imag,L,mode='nearest')
    ff=uniform_filter1d(freq,L,mode='nearest');tt=np.arange(0,3+1e-10,1/R)
    vc=np.interp(tt,t,vv.real)+1j*np.interp(tt,t,vv.imag);fc=np.interp(tt,t,ff)
    ch=dict(voltage_magnitude_pu=abs(vc),voltage_angle_deg=np.angle(vc,deg=True),frequency_hz=fc,rocof_hz_per_s=np.gradient(fc,tt))
    mask=(tt>=.75-1e-12)&(tt<=2.75+1e-12)
    return S.features(tt[mask],{k:a[mask] for k,a in ch.items()},R)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    S.OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(S.COHORT/'case_manifest.csv');rows=[];validation=[]
    protocol=S.OUT/('native_pilot_protocol.json' if args.pilot else 'native_protocol.json')
    if not protocol.exists():protocol.write_text(json.dumps(dict(original_cases=64,new_cases=40,calibration='original nominal source-bus waveform replay; reciprocal network splits 16/16/32',processing=['reference','filtered'],chains=list(W.CHAINS),native_input_rate_hz=1000,resample_up=24,resample_down=5,resampling_claim='rate compatibility only; original Nyquist 500Hz unchanged',new_case_calibration='frozen original records',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    for ci,c in manifest.iloc[:1 if args.pilot else len(manifest)].iterrows():
        path=P.OUT/f'{c.case_id}.npz';r,t,v,freq,wave=extract(path,c.source_bus)
        for name in W.CHAINS:
            for proc in ('reference','filtered'):
                rows.append(dict(case_id=c.case_id,group=c.dependency_group,condition=c.network_condition,alignment=int(c.alignment_index),truth=float(c.target_frequency_hz),chain=name,processing=proc,**W.replay(wave,name,proc)))
        validation.append(dict(case_id=c.case_id,source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),samples=len(t),finite=True,vmin=float(abs(v).min()),vmax=float(abs(v).max())))
        if ci%16==0:print('native nominal',ci,flush=True)
    if args.pilot:
        (S.OUT/'native_pilot.json').write_text(json.dumps([dict(r,z=r['z'].tolist(),amplitude=r['amplitude'].tolist()) for r in rows],indent=2));return
    models={};results=[];design=[]
    for name in W.CHAINS:
        for proc in ('reference','filtered'):
            for condition in sorted(set(r['condition'] for r in rows)):
                training=[dict(r,role='design' if r['alignment']<2 else 'calibration') for r in rows if r['chain']==name and r['processing']==proc and r['condition']!=condition]
                M=S.fit(training,60);models[(name,proc,condition)]=M
                design.append(dict(chain=name,processing=proc,test_condition=condition,**{k:v for k,v in M.items() if k not in ('m','s','catalogue')}))
                for r in rows:
                    if r['chain']==name and r['processing']==proc and r['condition']==condition:
                        results.append(dict(case_id=r['case_id'],group=r['group'],chain=name,processing=proc,condition=condition,kind='nominal_replay',level=0,**S.decide(r,M)))
    pd.DataFrame(design).to_csv(S.OUT/'native_design.csv',index=False)
    original=S.load_records();oldmodels={};newpll=[];convergence=[]
    for row in P.cases():
        path=P.OUT/(row['case_id']+'.npz')
        if not path.exists():raise RuntimeError('New simulation not finished: '+str(path))
        r,t,v,freq,wave=extract(path,row['bus']);kind='offset' if row['offset'] else ('drift' if row['beta'] else 'new_nominal')
        truth=row['nominal']+row['offset'] if row['beta']==0 else float('nan');level=row['offset'] or row['beta']
        for name in W.CHAINS:
            for proc in ('reference','filtered'):
                f=W.replay(wave,name,proc);rr=dict(truth=truth,**f)
                results.append(dict(case_id=row['case_id'],chain=name,processing=proc,condition=row['condition'],kind=kind,level=level,**S.decide(rr,models[(name,proc,row['condition'])])))
        for cyc in (1,2):
            for R in (30,60,120):
                key=(cyc,R,row['condition'])
                if key not in oldmodels:oldmodels[key]=S.train(original,*key)
                rr=dict(truth=truth,**filtered_features(t,v,freq,cyc,R))
                newpll.append(dict(case_id=row['case_id'],cycle=cyc,R=R,kind=kind,level=level,**S.decide(rr,oldmodels[key])))
        validation.append(dict(case_id=row['case_id'],source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),samples=len(t),finite=bool(np.isfinite(v).all() and np.isfinite(freq).all()),vmin=float(abs(v).min()),vmax=float(abs(v).max())))
        # Compare the saved 1-kHz waveform route with a 20-kHz native terminal route.
        if not row['offset'] and not row['beta']:
            with np.load(ROOT/'data/terminal_20khz'/(row['case_id']+'.npz'),allow_pickle=False) as d:
                ht=d['time_s'];hv=d['voltage_abc_pu']
            target=np.arange(60000)/20000;hv=np.column_stack([np.interp(target,ht,hv[:,j]) for j in range(3)])
            high=resample_poly(hv,6,25,axis=0)
            for name in ('SDFT-pre','TF-p2'):
                lowf=W.replay(wave,name,'filtered');highf=W.replay(high,name,'filtered')
                dl=S.decide(dict(truth=truth,**lowf),models[(name,'filtered',row['condition'])]);dh=S.decide(dict(truth=truth,**highf),models[(name,'filtered',row['condition'])])
                convergence.append(dict(case_id=row['case_id'],chain=name,max_feature_difference=float(np.max(abs(lowf['z'][:3]-highf['z'][:3]))),same_outcome=dl['outcome']==dh['outcome'],same_fold=lowf['a']==highf['a']))
        print('native stress',row['case_id'],flush=True)
    df=pd.DataFrame(results);df.to_csv(S.OUT/'native_decisions.csv',index=False);S.summary(df,['chain','processing','kind','level'],S.OUT/'native_summary.csv')
    df=pd.DataFrame(newpll);df.to_csv(S.OUT/'new_paraemt_decisions.csv',index=False);S.summary(df,['cycle','R','kind','level'],S.OUT/'new_paraemt_summary.csv')
    pd.DataFrame(validation).to_csv(S.OUT/'simulation_validation.csv',index=False);pd.DataFrame(convergence).to_csv(S.OUT/'native_rate_comparison.csv',index=False)
    print('native replay complete',flush=True)
if __name__=='__main__':main()
