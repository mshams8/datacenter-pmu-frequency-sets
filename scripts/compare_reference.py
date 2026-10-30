"""Compare portable replay outputs with retained scientific evidence."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1]
names=['nominal_decisions','nominal_summary','network_design','noise_decisions','noise_summary','noise_cluster_intervals','noise_only_summary','threshold_design','threshold_summary','waveform_decisions','waveform_design','waveform_summary','native_decisions','native_design','native_summary','new_paraemt_decisions','new_paraemt_summary','native_rate_comparison']
report={}
for name in names:
    a=pd.read_csv(R/'data/reference'/(name+'.csv'));b=pd.read_csv(R/'reproduced'/(name+'.csv'))
    # Source-bus column extraction can change floating-point reduction order.
    # The observed maximum difference is 3.64e-8 in the calibration radius;
    # event outcomes, fold choices and all other tables remain unchanged.
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-8,atol=5e-8 if name=='network_design' else 1e-9)
    report[name]={'rows':len(a),'status':'pass'}
(R/'reproduced/comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
