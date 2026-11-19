"""Stream GDSC1 raw plate data into per-plate control statistics and per-(plate, drug slot) readings.

Definitions follow docs/PREREG_GDSC.md. Writes data/processed/gdsc_plates.csv.gz and gdsc_units.csv.gz.
"""
import argparse,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd

COLS=['BARCODE','COSMIC_ID','CELL_LINE_NAME','DRUGSET_ID','ASSAY','TAG','DRUG_ID','INTENSITY']
LIB=re.compile(r'^(L\d+)-D(\d+)-S$')

def split_of(barcode):
    """Pre-registered split: residues 0-2 of sha256(barcode) mod 10 are exploration, 3-9 test."""
    return 'explore' if int(hashlib.sha256(str(barcode).encode()).hexdigest(),16)%10<3 else 'test'

def accumulate(path,chunksize):
    """Per (BARCODE, TAG, DRUG_ID): count, sum, and sum of squares of INTENSITY, plus plate metadata."""
    sums=[];meta={}
    for ch in pd.read_csv(path,usecols=COLS,chunksize=chunksize,dtype={'BARCODE':str,'TAG':str,'DRUGSET_ID':str,'ASSAY':str,'CELL_LINE_NAME':str}):
        ch=ch.dropna(subset=['INTENSITY'])
        for b,g in ch.drop_duplicates('BARCODE').set_index('BARCODE')[['COSMIC_ID','CELL_LINE_NAME','DRUGSET_ID','ASSAY']].iterrows():meta.setdefault(b,g.to_dict())
        ch['DRUG_ID']=ch['DRUG_ID'].fillna(-1).astype(int);ch['I2']=ch['INTENSITY']**2
        sums.append(ch.groupby(['BARCODE','TAG','DRUG_ID']).agg(n=('INTENSITY','size'),s=('INTENSITY','sum'),s2=('I2','sum')))
    agg=pd.concat(sums).groupby(level=[0,1,2]).sum().reset_index()
    return agg,pd.DataFrame.from_dict(meta,orient='index').rename_axis('BARCODE').reset_index()

def stats(g):
    m=g.s/g.n;var=np.maximum(g.s2/g.n-m**2,0)*g.n/np.maximum(g.n-1,1)
    return m,np.sqrt(var)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',default='data/raw/gdsc/GDSC1_public_raw_data_24Jul22.csv')
    ap.add_argument('--out',default='data/processed');ap.add_argument('--chunksize',type=int,default=2_000_000);args=ap.parse_args()
    agg,meta=accumulate(args.raw,args.chunksize)
    ctl=agg.groupby(['BARCODE','TAG'])[['n','s','s2']].sum().reset_index()
    ctl['mean'],ctl['sd']=stats(ctl)
    wide=ctl.pivot(index='BARCODE',columns='TAG',values=['n','mean','sd'])
    get=lambda k,t:wide[(k,t)] if (k,t) in wide.columns else pd.Series(np.nan,index=wide.index)
    p=pd.DataFrame(index=wide.index)
    use_nc1=get('n','NC-1').fillna(0)>=4
    for k in ('n','mean','sd'):p[f'nc_{k}']=np.where(use_nc1,get(k,'NC-1'),get(k,'NC-0'));p[f'b_{k}']=get(k,'B')
    p['nc_tag']=np.where(use_nc1,'NC-1','NC-0')
    p['zprime']=1-3*(p.nc_sd+p.b_sd)/(p.nc_mean-p.b_mean).abs()
    pc=ctl[ctl.TAG.str.startswith('PC')].groupby('BARCODE')[['n','s']].sum()
    p['pc_n']=pc.n.reindex(p.index).fillna(0);p['pc_mean']=(pc.s/pc.n).reindex(p.index)
    p=p.reset_index().merge(meta,on='BARCODE',how='left')
    p['eligible']=(p.nc_n>=4)&(p.b_n>=4)&(p.nc_mean>p.b_mean)
    p['split']=p.BARCODE.map(split_of)
    # Units: single-agent library slots; effect = 1 - viability averaged over the two highest doses (D1, D2).
    lib=agg[agg.TAG.str.match(LIB)].copy()
    lib[['slot','dose']]=lib.TAG.str.extract(LIB);lib['dose']=lib.dose.astype(int)
    top=lib[lib.dose<=2].groupby(['BARCODE','slot','DRUG_ID'])[['n','s']].sum().reset_index()
    u=top.merge(p[['BARCODE','nc_mean','b_mean','eligible','split','COSMIC_ID','CELL_LINE_NAME','DRUGSET_ID','zprime']],on='BARCODE')
    u['viability']=(u.s/u.n-u.b_mean)/(u.nc_mean-u.b_mean);u['effect']=1-u.viability
    u=u.drop(columns=['s','nc_mean','b_mean']).rename(columns={'n':'n_wells'})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    p.to_csv(out/'gdsc_plates.csv.gz',index=False);u.to_csv(out/'gdsc_units.csv.gz',index=False)
    print(f'{len(p)} plates ({p.eligible.sum()} eligible), {len(u)} units -> {out}')
if __name__=='__main__':main()
