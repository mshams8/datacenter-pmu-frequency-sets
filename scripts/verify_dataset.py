"""Verify archive hashes, declared counts and safe numerical arrays."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];D=R/'data'
for line in (D/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split('  ',1)
    assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
expected={'native':104,'terminal_20khz':8,'original/pmu':384}
for name,n in expected.items():
    files=list((D/name).rglob('*.npz'));assert len(files)==n,(name,len(files))
    for p in files:
        with np.load(p,allow_pickle=False) as a:
            for key in a.files:assert np.isfinite(a[key]).all(),(p,key)
            assert np.all(np.diff(a['time_s'])>0),p
assert len(pd.read_csv(D/'native_manifest.csv'))==104
assert len(pd.read_csv(D/'reference/noise_decisions.csv'))==76800
print(json.dumps({'hashes':'pass','finite_arrays':'pass','counts':expected},indent=2))
