import os, numpy as np, pandas as pd
from scipy import stats, optimize
D=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultados', 'eficiencias_dmu_ano_todas.csv')); OUTS=['km_ge230','km_lt230','mva','equip','manobra']
def fit(d):
    y=np.log(d.pmso.values); X=np.c_[np.ones(len(d)),np.log1p(d[OUTS].values)]
    def nll(th):
        b=th[:-2]; sv,su=np.exp(th[-2:]); e=y-X@b; s=np.hypot(sv,su)
        return -np.sum(np.log(2)-np.log(s)+stats.norm.logpdf(e/s)+stats.norm.logcdf(e*(su/sv)/s))
    b0=np.linalg.lstsq(X,y,rcond=None)[0]; r=optimize.minimize(nll,np.r_[b0,np.log(.3),np.log(.5)],method='BFGS')
    return r.x[1:6].sum()
g=D.dmu.unique(); rng=np.random.default_rng(1); S=[]
for b in range(300):
    pick=rng.choice(g,len(g),replace=True); d=pd.concat([D[D.dmu==k] for k in pick]); S.append(fit(d))
S=np.array(S); print('soma', round(fit(D),3), 'IC cluster', np.percentile(S,[2.5,97.5]).round(3), 'se', S.std().round(3))
pd.Series({'sfa_soma':fit(D),'lo':np.percentile(S,2.5),'hi':np.percentile(S,97.5)}).to_csv('resultados/sfa_soma_bootstrap_cluster.csv')
d21=D[D.ano==2021]; print('grupos <300 km em 2021', (d21.km<300).sum(), 'de', len(d21))
print('amostra pmso 2017/2021', D[D.ano==2017].pmso.sum()/1e6, D[D.ano==2021].pmso.sum()/1e6, 'km', D[D.ano==2017].km.sum(), D[D.ano==2021].km.sum())
