"""Frozen reviewer-motivated studies; empirical calibration, no coverage claim."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.signal import lfilter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pmu_sso.inference.candidate_scan import complex_amplitude, scan_candidates
OUT=ROOT/'reproduced'
COHORT=ROOT/'data/original'
F=np.array([14.7,29.,31.,45.3])
NAMES=('voltage_magnitude_pu','voltage_angle_deg','frequency_hz','rocof_hz_per_s')
OFFSETS=(-1.5,-1,-.6,.6,1,1.5)
def fold(f,R): return np.minimum(np.remainder(f,R),R-np.remainder(f,R))

@lru_cache(None)
def basis(t_tuple,R,catalogue):
    t=np.array(t_tuple); a=np.unique(np.round(fold(np.array(catalogue),R),9))
    freq=list(a); backgrounds=[]
    for v in a:
        b=[v+d for d in OFFSETS if 1.5<=v+d<=R/2]
        backgrounds.append(list(range(len(freq),len(freq)+len(b))));freq.extend(b)
    X=np.c_[np.ones(len(t)),t]; detrend=np.eye(len(t))-X@np.linalg.pinv(X)
    w=np.hanning(len(t)); B=(2/w.sum())*(np.exp(-2j*np.pi*np.outer(freq,t))*w)@detrend
    return a,B,backgrounds

def features(t,channels,R,catalogue=F):
    Y=np.column_stack([channels[n] for n in NAMES]);Y[:,1]=np.rad2deg(np.unwrap(np.deg2rad(Y[:,1])))
    a,B,bg=basis(tuple(t),R,tuple(catalogue)); C=B@Y
    supports=np.array([np.log(np.maximum(abs(C[i]),1e-15)/np.maximum(np.median(abs(C[ix]),axis=0),1e-15)) for i,ix in enumerate(bg)])
    legacy=np.sort(supports,axis=1)[:,-2:].mean(axis=1)
    family=supports[:,1:3].mean(axis=1)
    i=int(np.argmax(legacy)); j=int(np.argmax(family))
    # Safe zero treatment: caller rejects nonfinite responses before scoring.
    with np.errstate(divide='ignore',invalid='ignore'):
        rho=(180/np.pi)*C[i,2]/C[i,1];rho2=C[i,3]/(2*np.pi*C[i,2])
        z=np.array([np.log(abs(rho)),np.cos(np.angle(rho)),np.sin(np.angle(rho)),np.log(abs(rho2)),np.cos(np.angle(rho2)),np.sin(np.angle(rho2))])
    return dict(a=float(a[i]),a_family=float(a[j]),support=float(min(supports[i,1:3])),z=z,
                amplitude=np.abs(C[i]),legacy_score=float(legacy[i]))

def fit(rows,R,catalogue=F,tau=4.,floor=.02,alpha=.1,combined=False):
    D=[r for r in rows if r['role']=='design']; K=[r for r in rows if r['role']=='calibration']
    cf=np.array(catalogue); pairs=[(i,j) for i in range(len(cf)) for j in range(i+1,len(cf)) if abs(fold(cf[i],R)-fold(cf[j],R))<1e-8]
    options=([0,3],list(range(6))) if combined else ([0],[0,1,2])
    candidates=[]
    for ix in options:
        m=[];s=[]
        for f in cf:
            Z=np.array([r['z'][ix] for r in D if abs(r['truth']-f)<1e-8]); c=np.median(Z,axis=0)
            m.append(c);s.append(np.maximum(1.4826*np.median(abs(Z-c),axis=0),floor))
        m=np.array(m);s=np.array(s)
        sep=min([np.sqrt(np.mean((m[i]-m[j])**2/(s[i]**2+s[j]**2))) for i,j in pairs],default=float('inf'))
        candidates.append((ix,m,s,sep))
    ix,m,s,sep=candidates[0 if candidates[0][3]>=tau else 1]
    scores=[]
    for r in K:
        k=int(np.argmin(abs(cf-r['truth']))); scores.append(float(np.sqrt(np.mean(((r['z'][ix]-m[k])/s[k])**2))))
    rank=math.ceil((len(scores)+1)*(1-alpha));q=float('inf') if rank>len(scores) else float(np.sort(scores)[rank-1])
    return dict(ix=ix,m=m,s=s,sep=sep,q=q,eligible=sep>=tau,catalogue=cf,R=R,
                sep_mag=candidates[0][3],sep_complex=candidates[1][3],n_design=len(D),n_cal=len(K),tau=tau,floor=floor)

def decide(r,M):
    cf=M['catalogue']; idx=np.flatnonzero(abs(fold(cf,M['R'])-r['a'])<1e-8)
    exact=bool(any(abs(cf[idx]-r['truth'])<1e-8)); outcome='unresolved'; pick=float('nan'); forced=float('nan')
    if len(idx)==1: outcome='singleton';pick=float(cf[idx[0]]); forced=pick
    elif not M['eligible']: pass
    elif r['support']<math.log(2) or not np.isfinite(r['z'][M['ix']]).all(): outcome='insufficient'
    else:
        scores=np.sqrt(np.mean(((r['z'][M['ix']]-M['m'][idx])/M['s'][idx])**2,axis=1))
        forced=float(cf[idx[np.argmin(scores)]]); good=idx[scores<=M['q']+1e-12]
        if len(good)==0:outcome='outside'
        elif len(good)==1:outcome='singleton';pick=float(cf[good[0]])
    correct=bool(outcome=='singleton' and abs(pick-r['truth'])<1e-8)
    return dict(outcome=outcome,singleton=outcome=='singleton',correct=correct,
                incorrect=outcome=='singleton' and not correct,fold_correct=exact,pick=pick,forced=forced)

def load_records():
    manifest=pd.read_csv(COHORT/'case_manifest.csv');records=[]
    for ci,c in manifest.iterrows():
        for cyc in (1,2):
            for R in (30,60,120):
                path=COHORT/f"pmu/open_{cyc}cycle/{R}fps/{c.case_id}.npz"
                with np.load(path) as p:
                    t=p['time_s'];mask=(t>=.75-1e-12)&(t<=2.75+1e-12);bi=int(np.flatnonzero(p['bus_ids']==c.source_bus)[0])
                    ch={n:np.asarray(p[n][mask,bi],float) for n in NAMES}
                    t=t[mask]
                records.append(dict(case_id=c.case_id,group=c.dependency_group,condition=c.network_condition,
                    alignment=int(c.alignment_index),truth=float(c.target_frequency_hz),cycle=cyc,R=R,t=t,channels=ch,**features(t,ch,R)))
    return records

def train(records,cyc,R,testcondition,**kw):
    rows=[]
    for r in records:
        if r['cycle']==cyc and r['R']==R and r['condition']!=testcondition:
            rows.append(dict(r,role='design' if r['alignment']<2 else 'calibration'))
    return fit(rows,R,**kw)

def summary(df,keys,path):
    d=df.copy();d['outside']=d.outcome.eq('outside');d['unresolved']=d.outcome.eq('unresolved');d['insufficient']=d.outcome.eq('insufficient')
    agg=d.groupby(keys,dropna=False)[['correct','incorrect','fold_correct','singleton','outside','unresolved','insufficient']].mean()
    agg['n']=d.groupby(keys,dropna=False).size();agg.reset_index().to_csv(path,index=False)

def network_study(pilot=False):
    records=load_records();models={}; designs=[];nominal=[]
    for r in records:
        key=(r['cycle'],r['R'],r['condition'])
        if key not in models:
            models[key]=train(records,*key)
            M=models[key];designs.append(dict(cycle=key[0],test_condition=key[2],**{k:v for k,v in M.items() if k not in ('m','s','catalogue')}))
        nominal.append(dict(case_id=r['case_id'],group=r['group'],cycle=r['cycle'],R=r['R'],method='rho1',**decide(r,models[key])))
    pd.DataFrame(designs).to_csv(OUT/'network_design.csv',index=False)
    pd.DataFrame(nominal).to_csv(OUT/'nominal_decisions.csv',index=False);summary(pd.DataFrame(nominal),['cycle','R'],OUT/'nominal_summary.csv')
    # Numerical cross-check against the archived coefficient and fold implementations.
    maxerr=0.
    for r in records:
        scan=scan_candidates(r['t'],r['channels'],F,r['R']);a=max(scan,key=lambda x:x['support'])['observed_alias_hz']
        assert abs(a-r['a'])<1e-8
        z=(180/np.pi)*complex_amplitude(r['t'],r['channels'][NAMES[2]],a)/complex_amplitude(r['t'],np.rad2deg(np.unwrap(np.deg2rad(r['channels'][NAMES[1]]))),a)
        maxerr=max(maxerr,abs(np.log(abs(z))-r['z'][0]))
    (OUT/'coefficient_equivalence.json').write_text(json.dumps(dict(max_log_ratio_difference=maxerr,records=len(records))))
    noise=[];ns=5 if pilot else 20
    for ri,r in enumerate(records):
        M=models[(r['cycle'],r['R'],r['condition'])];t=r['t'];n=len(t)
        amp=np.array([abs(complex_amplitude(t,r['channels'][name],fold(r['truth'],r['R'])))/np.sqrt(2) for name in NAMES[:3]])
        for kind in ('white','correlated'):
            for seed in range(ns):
                rng=np.random.default_rng(20261030+1000*ri+seed)
                U=rng.normal(size=(n,4))
                if kind=='correlated':U=lfilter([1],[1,-.8],U,axis=0)
                U-=U.mean(axis=0);U/=np.sqrt(np.mean(U**2,axis=0))
                E=np.c_[U[:,0],U[:,1],.5*U[:,1]+np.sqrt(.75)*U[:,2]]
                E/=np.sqrt(np.mean(E**2,axis=0))
                for snr in (40,30,20,10,0):
                    sigma=amp*10**(-snr/20);ch={k:v.copy() for k,v in r['channels'].items()}
                    for j,name in enumerate(NAMES[:3]):ch[name]+=sigma[j]*E[:,j]
                    ch[NAMES[3]]=np.gradient(ch[NAMES[2]],t)
                    z=features(t,ch,r['R']);rr=dict(r,**z);d=decide(rr,M)
                    noise.append(dict(case_id=r['case_id'],group=r['group'],cycle=r['cycle'],R=r['R'],kind=kind,seed=seed,snr=snr,
                        family_fold_correct=abs(z['a_family']-fold(r['truth'],r['R']))<1e-8,angle_noise_deg=sigma[1],frequency_noise_hz=sigma[2],**d))
        if ri%64==0: print('noise records',ri,flush=True)
    df=pd.DataFrame(noise);df.to_csv(OUT/('noise_pilot.csv' if pilot else 'noise_decisions.csv'),index=False)
    summary(df,['cycle','R','kind','snr'],OUT/('noise_pilot_summary.csv' if pilot else 'noise_summary.csv'))
    # Cluster-bootstrap across the eight original dependency groups, never over derived noise trials.
    if not pilot:
        ci=[];rng=np.random.default_rng(1841)
        for keys,g in df.groupby(['cycle','R','kind','snr']):
            cluster=g.groupby('group')[['fold_correct','family_fold_correct','incorrect','correct']].mean().to_numpy()
            bootstrap=cluster[rng.integers(0,len(cluster),(2000,len(cluster)))].mean(axis=1)
            row=dict(zip(['cycle','R','kind','snr'],keys))
            for j,k in enumerate(['fold_correct','family_fold_correct','incorrect','correct']):row[k+'_lo'],row[k+'_hi']=np.quantile(bootstrap[:,j],[.025,.975])
            ci.append(row)
        pd.DataFrame(ci).to_csv(OUT/'noise_cluster_intervals.csv',index=False)
    sens=[]; sd=[]
    for floor in (.005,.01,.02,.05,.1):
        for tau in (1,2,3,4,5,6,8):
            for cyc in (1,2):
                for condition in sorted({r['condition'] for r in records}):
                    M=train(records,cyc,60,condition,tau=tau,floor=floor)
                    sd.append(dict(tau=tau,floor=floor,cycle=cyc,condition=condition,sep=M['sep'],q=M['q'],selected=str(M['ix']),eligible=M['eligible']))
                    for r in records:
                        if r['cycle']==cyc and r['R']==60 and r['condition']==condition:
                            sens.append(dict(tau=tau,floor=floor,cycle=cyc,case_id=r['case_id'],**decide(r,M)))
    pd.DataFrame(sd).to_csv(OUT/'threshold_design.csv',index=False)
    summary(pd.DataFrame(sens),['tau','floor','cycle'],OUT/'threshold_summary.csv')
    # Keep prospective event detection outside the method; disclose noise-only outputs.
    null=[]
    for r in records:
        rng=np.random.default_rng(19);ch={name:np.mean(r['channels'][name])+rng.normal(size=len(r['t']))*np.std(r['channels'][name])*.1 for name in NAMES[:3]}
        ch[NAMES[3]]=np.gradient(ch[NAMES[2]],r['t']);rr=dict(r,truth=float('nan'),**features(r['t'],ch,r['R']))
        null.append(dict(cycle=r['cycle'],R=r['R'],**decide(rr,models[(r['cycle'],r['R'],r['condition'])])))
    summary(pd.DataFrame(null),['cycle','R'],OUT/'noise_only_summary.csv')
    print('network studies complete',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/('signal_pilot_protocol.json' if args.pilot else 'signal_protocol.json')
    if not protocol.exists():protocol.write_text(json.dumps(dict(seed=20261030,seeds=5 if args.pilot else 20,snr=[40,30,20,10,0],noise=['white','AR1_0.8'],channel_noise_correlation=.5,scales=[.005,.01,.02,.05,.1],tau=[1,2,3,4,5,6,8],calibration='original unperturbed records, reciprocal conditions',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),created_before_results=True),indent=2))
    network_study(args.pilot)
if __name__=='__main__':main()
