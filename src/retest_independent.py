"""Secondary grouped holdout for retest prioritization (review round 6; post hoc; not fully independent, see docs/PREREG_GDSC.md round 7). The model and rules are frozen as in src/retest_priority.py
(trained on exploration nulls from one hash-half of cell lines) and evaluated on exploration-split nulls with a replicate from the other half:
plates and cell lines not used to train the model or choose rules, but sharing the tau fit, response-rate features, and compound labels. Writes results/gdsc/retest_independent.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from retest_priority import FEATURES,half,curve

def main():
    ex=pd.read_csv('results/gdsc/null_features_explore.csv.gz',dtype={'BARCODE':str})
    tr=ex[half(ex.COSMIC_ID)];ev=ex[~half(ex.COSMIC_ID)&ex.alt_effect.notna()].reset_index(drop=True)
    gbm=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.,random_state=0).fit(tr[FEATURES].to_numpy(float),tr.H)
    rng=np.random.default_rng(0)
    sc={'random':rng.random(len(ev)),'lowest_zprime':-ev.zprime.to_numpy(),'weakest_null':ev.effect.to_numpy(),'compound_history':ev.drug_resp.fillna(0).to_numpy(),
        'learned_gbm':gbm.predict_proba(ev[FEATURES].to_numpy(float))[:,1]}
    out={'n_nulls_with_replicate':int(len(ev)),'effective_share':float(ev.H.mean()),'n_cell_lines':int(ev.COSMIC_ID.nunique())}
    cells=ev.COSMIC_ID.to_numpy();uc=np.unique(cells);idx={x:np.flatnonzero(cells==x) for x in uc}
    for k,v in sc.items():
        c=curve(ev,v,(.1,.2,1.));b=[curve(ev.iloc[i].reset_index(drop=True),v[i],(.1,))[0]['recovered_share'] for i in (np.concatenate([idx[x] for x in rng.choice(uc,len(uc))]) for _ in range(500))]
        out[k]={'recovered_10':c[0]['recovered_share'],'recovered_10_ci':[float(x) for x in np.percentile(b,[2.5,97.5])],'recovered_20':c[1]['recovered_share'],'recovered_all':c[2]['recovered_share'],'precision_10':c[0]['retest_precision']}
    Path('results/gdsc/retest_independent.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
if __name__=='__main__':main()
