"""Segundo passo do teste de Simar & Wilson (2002): rejeitado CRS, testa H0: NIRS (retornos não crescentes)
contra VRS. Estatística S2 = média(theta_NIRS)/média(theta_VRS); pseudo-amostras geradas da fronteira NIRS.
Também classifica a natureza de escala de cada unidade (determinística): CRS se theta_CRS = theta_VRS;
DRS se theta_CRS < theta_VRS e theta_NIRS = theta_VRS; IRS caso contrário.
Uso: python rts_test_nirs.py [B]. Saídas: resultados/rts_test_SW2002_nirs.csv, resultados/escala_por_unidade.csv
"""
import os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
from dea import dea_env

R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
X = D['pmso'].values[:, None]; Y = D[OUTS].values.astype(float); n = len(D)
th_c = dea_env(X, Y, 'crs'); th_v = dea_env(X, Y, 'vrs'); th_n = dea_env(X, Y, 'nirs')
tol = 1e-6
esc = np.where(np.abs(th_c - th_v) < tol, 'CRS', np.where(np.abs(th_n - th_v) < tol, 'DRS', 'IRS'))
D['escala'] = esc; D['scale_eff'] = th_c / th_v
leg = ['AG394', 'AG48', 'AG372', 'AG97']
D['grupo'] = np.where(D.dmu.isin(leg), 'legadas', np.where(D.pmso < D.pmso.median(), 'metade menor', 'metade maior (exc. legadas)'))
D[['dmu', 'nome', 'ano', 'pmso', 'escala', 'scale_eff', 'grupo']].to_csv(os.path.join(R, 'escala_por_unidade.csv'), index=False)
print(pd.crosstab(D.grupo, D.escala)); print(D.groupby('grupo').scale_eff.mean().round(3))
S_obs = th_n.mean() / th_v.mean()
d = 1 / th_n; refl = np.r_[d, 2 - d]
sd = refl.std(ddof=1); iqr = np.subtract(*np.percentile(refl, [75, 25]))
h = 0.9 * min(sd, iqr / 1.34) * (2 * n) ** (-1 / 5)
def one(b):
    rng = np.random.default_rng(5000 + b)
    beta = rng.choice(d, n, replace=True); eps = rng.standard_normal(n); bm = beta.mean()
    dstar = bm + (beta + h * eps - bm) / np.sqrt(1 + h ** 2 / np.var(d))
    dstar = np.where(dstar < 1, 2 - dstar, dstar)
    Xs = X * (th_n / (1 / dstar))[:, None]
    return dea_env(X, Y, 'nirs', Xref=Xs, Yref=Y).mean() / dea_env(X, Y, 'vrs', Xref=Xs, Yref=Y).mean()
t0 = time.time()
S_star = np.array(Parallel(n_jobs=-1)(delayed(one)(b) for b in range(B)))
p = np.mean(S_star <= S_obs)
out = dict(B=B, h=h, S_obs=S_obs, S_star_mean=S_star.mean(), S_star_q025=np.percentile(S_star, 2.5),
           S_star_q05=np.percentile(S_star, 5), p_value=p, mean_NIRS=th_n.mean(), mean_VRS=th_v.mean(),
           n_IRS=int((esc == 'IRS').sum()), n_DRS=int((esc == 'DRS').sum()), n_CRS=int((esc == 'CRS').sum()))
pd.Series(out).to_csv(os.path.join(R, 'rts_test_SW2002_nirs.csv')); print(pd.Series(out).round(4), 'tempo', round(time.time() - t0))
