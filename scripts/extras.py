import sys, os, numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from dea import dea_mult, dea_env
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultados'); D=pd.read_csv(f'{R}/eficiencias_dmu_ano_todas.csv')
OUTS=['km_ge230','km_lt230','mva','equip','manobra']
out={}
# 1) SFA: soma das elasticidades com EP (Hessiana numérica via BFGS inv)
y=np.log(D.pmso.values); X=np.c_[np.ones(len(D)),np.log1p(D[OUTS].values)]
def nll(th):
    b=th[:-2]; sv,su=np.exp(th[-2:]); e=y-X@b; s=np.hypot(sv,su)
    return -np.sum(np.log(2)-np.log(s)+stats.norm.logpdf(e/s)+stats.norm.logcdf(e*(su/sv)/s))
b0=np.linalg.lstsq(X,y,rcond=None)[0]; r=optimize.minimize(nll,np.r_[b0,np.log(.3),np.log(.5)],method='BFGS')
import numdifftools as nd
H=nd.Hessian(nll)(r.x); V=np.linalg.inv(H); g=np.r_[0,np.ones(5),0,0]
s=g@r.x; se=np.sqrt(g@V@g); out.update(sfa_soma=s,sfa_soma_se=se,sfa_soma_lo=s-1.96*se,sfa_soma_hi=s+1.96*se)
# OLS soma
import statsmodels.api as sm
m=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(D.dmu)[0]}); s2=m.params[1:].sum(); se2=np.sqrt(np.ones(5)@m.cov_params()[1:,1:]@np.ones(5))
out.update(ols_soma=s2,ols_soma_lo=s2-1.96*se2,ols_soma_hi=s2+1.96*se2)
# 2) participação das 4 estatais legadas do grupo Eletrobras no PMSO 2021 e nos produtos
leg=['AG394','AG48','AG372','AG97']; d21=D[D.ano==2021]
out['share_pmso_legadas']=d21[d21.dmu.isin(leg)].pmso.sum()/d21.pmso.sum()
out['share_km_legadas']=d21[d21.dmu.isin(leg)].km.sum()/d21.km.sum()
out['custo_por_km_legadas']=d21[d21.dmu.isin(leg)].pmso.sum()/d21[d21.dmu.isin(leg)].km.sum()
out['custo_por_km_outras']=d21[~d21.dmu.isin(leg)].pmso.sum()/d21[~d21.dmu.isin(leg)].km.sum()
# 3) sensibilidade: sem as legadas, custo reconhecido por especificação
S=D[~D.dmu.isin(leg)].reset_index(drop=True); Xs=S.pmso.values; Ys=S[OUTS].values.astype(float)
K = 1.2766  # IPCA jan/2017 -> dez/2021: limites monetários da NT 160/2017 levados a preços de dez/2021
AR = [('u', 0, 'v', 0, 2.5 * K, 8.5 * K), ('u', 1, 'u', 0, 0.20, 0.75), ('u', 2, 'v', 0, 0.4 * K, 4.0 * K),
      ('u', 4, 'v', 0, 15 * K, 70 * K), ('u', 3, 'u', 4, 1, 10)]
rows=[]
for nm,rts,ar in [('CRS','crs',None),('VRS','vrs',None),('NDRS','ndrs',None),('NDRS_AR','ndrs',AR)]:
    S['e']=dea_mult(Xs,Ys,rts,ar)[0]; E=S.groupby('dmu').e.mean(); alvo=E[E>=.5].mean()
    p21=S[S.ano==2021].set_index('dmu').pmso; rec=(p21*E.reindex(p21.index)/alvo).sum()
    rows.append(dict(spec=nm,media=E.mean(),alvo=alvo,var_pct=100*(rec/p21.sum()-1)))
sens=pd.DataFrame(rows); sens.to_csv(f'{R}/sens_sem_legadas.csv',index=False); print(sens.round(3))
# 4) descritivas
desc=D[['pmso','km_ge230','km_lt230','mva','equip','manobra','h_int','idade','fogo','frac_amz','km_por_se']].describe().T[['mean','50%','min','max']]
desc.to_csv(f'{R}/descritivas.csv'); print(desc.round(1))
# 5) efeito por ano da média de eficiência e média PMSO real
print(D.groupby('ano')[['ef_NDRS_AR','ef_VRS_bc','pmso']].mean().round(3))
out['n']=len(D); out['dmus']=D.dmu.nunique(); out['agentes_2021']=int(D[D.ano==2021].n_agentes.sum())
pd.Series(out).to_csv(f'{R}/extras.csv'); print(pd.Series(out).round(3))
