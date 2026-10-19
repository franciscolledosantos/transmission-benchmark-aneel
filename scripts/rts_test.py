"""Teste bootstrap de retornos de escala de Simar & Wilson (2002, EJOR 139, 115-132) e
CRS corrigido de viés (bootstrap homogêneo suavizado de Simar & Wilson, 1998, sob CRS).

H0: tecnologia CRS. Estatística: S = média(theta_CRS)/média(theta_VRS) (Farrell, orientação insumo), S em (0, 1].
Sob H0, S é próximo de 1; valores pequenos rejeitam. Pseudo-amostras geradas da fronteira CRS (bootstrap
homogêneo com kernel refletido e banda de Silverman); p-valor = proporção de S* <= S observado.
O mesmo laço fornece o CRS corrigido de viés e seu IC 95%, e o custo reconhecido correspondente.
Uso: python rts_test.py [B]
"""
import os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
from dea import dea_env

R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
X = D['pmso'].values[:, None]; Y = D[OUTS].values.astype(float)
n = len(D)
t0 = time.time()
th_c = dea_env(X, Y, 'crs'); th_v = dea_env(X, Y, 'vrs')
S_obs = th_c.mean() / th_v.mean()
print('S observado', round(S_obs, 4), 'médias CRS/VRS', th_c.mean().round(4), th_v.mean().round(4), 'tempo', round(time.time() - t0, 1))
# banda de Silverman sobre distâncias de Shephard CRS refletidas
d = 1 / th_c
refl = np.r_[d, 2 - d]
sd = refl.std(ddof=1); iqr = np.subtract(*np.percentile(refl, [75, 25]))
h = 0.9 * min(sd, iqr / 1.34) * (2 * n) ** (-1 / 5)

def one(b):
    rng = np.random.default_rng(1000 + b)
    beta = rng.choice(d, n, replace=True)
    eps = rng.standard_normal(n)
    bm = beta.mean()
    dstar = bm + (beta + h * eps - bm) / np.sqrt(1 + h ** 2 / np.var(d))
    dstar = np.where(dstar < 1, 2 - dstar, dstar)
    Xs = X * (th_c / (1 / dstar))[:, None]          # pseudo-insumos na tecnologia CRS
    tc = dea_env(X, Y, 'crs', Xref=Xs, Yref=Y)       # theta* CRS (original contra pseudo-fronteira)
    tv = dea_env(X, Y, 'vrs', Xref=Xs, Yref=Y)
    return tc, tv

res = Parallel(n_jobs=-1)(delayed(one)(b) for b in range(B))
TC = np.array([r[0] for r in res]); TV = np.array([r[1] for r in res])
S_star = TC.mean(1) / TV.mean(1)
p = np.mean(S_star <= S_obs)
print('B', B, 'h', round(h, 4), 'S* média', S_star.mean().round(4), 'quantis 2.5/5/50', np.percentile(S_star, [2.5, 5, 50]).round(4), 'p-valor', round(p, 4), 'tempo', round(time.time() - t0, 1))
# CRS corrigido de viés (distâncias de Shephard)
Dc, Ds = 1 / th_c, 1 / TC
bias = Ds.mean(0) - Dc
dbc = Dc - bias
q = np.percentile(Ds - Dc, [97.5, 2.5], axis=0)
D['ef_CRS_bc'] = 1 / dbc; D['ef_CRS_lo'] = 1 / (Dc - q[1]); D['ef_CRS_hi'] = 1 / (Dc - q[0])
D[['dmu', 'nome', 'ano', 'ef_CRS', 'ef_CRS_bc', 'ef_CRS_lo', 'ef_CRS_hi']].to_csv(os.path.join(R, 'crs_bootstrap.csv'), index=False)
# custo reconhecido com CRS corrigido de viés (normalização da NT 126/2018)
E = D.groupby('dmu').agg(e=('ef_CRS_bc', 'mean'), e0=('ef_CRS', 'mean'))
p21 = D[D.ano == 2021].set_index('dmu')['pmso']
alvo = E.loc[E.e >= 0.5, 'e'].mean(); rec = (p21 * E.e.reindex(p21.index) / alvo).sum()
var = 100 * (rec / p21.sum() - 1)
leg = ['AG394', 'AG48', 'AG372', 'AG97']
out = dict(B=B, h=h, S_obs=S_obs, S_star_mean=S_star.mean(), S_star_q025=np.percentile(S_star, 2.5),
           S_star_q05=np.percentile(S_star, 5), p_value=p, mean_CRS=th_c.mean(), mean_CRS_bc=D['ef_CRS_bc'].mean(),
           mean_VRS=th_v.mean(), n_ge05_CRS_bc=int((E.e >= 0.5).sum()), alvo_CRS_bc=alvo, var_pct_CRS_bc=var,
           legadas_CRS=E.loc[leg, 'e0'].mean(), legadas_CRS_bc=E.loc[leg, 'e'].mean(),
           largura_mediana_IC_2021=float((D.loc[D.ano == 2021, 'ef_CRS_hi'] - D.loc[D.ano == 2021, 'ef_CRS_lo']).median()))
pd.Series(out).to_csv(os.path.join(R, 'rts_test_SW2002.csv'))
print(pd.Series(out).round(4))
print(D[D.dmu.isin(leg) & (D.ano == 2021)][['nome', 'ef_CRS', 'ef_CRS_bc', 'ef_CRS_lo', 'ef_CRS_hi']].round(3))
