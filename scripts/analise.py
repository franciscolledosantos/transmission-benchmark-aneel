"""Benchmarking dos custos de O&M das transmissoras (2017-2021).
A) DEA (CRS, VRS, NDRS, NDRS com restrições de pesos da NT 160/2017) em fronteira conjunta 2017-2021
B) Bootstrap de Simar-Wilson (VRS)
C) Normalização regulatória (média das eficiências >= 50%, NT 126/2018) e custo reconhecido
D) Normas de custo paramétricas e de ML com validação deixando a DMU de fora (GroupKFold)
E) Segundo estágio: fogo, Amazônia, idade, dispersão, qualidade
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from dea import dea_env, dea_mult, bootstrap_sw

R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
D = pd.read_csv(os.path.join(R, 'painel_dmu_ano.csv'))
# amostra: custo positivo, algum produto, e exclui o primeiro ano de operação (ano parcial)
prim = D.groupby('dmu')['ano'].transform('min')
D = D[(D['pmso'] > 0) & ((D['km'] > 0) | (D['mva'] > 0)) & ((D['ano'] == prim) & (D['ano'] == 2017) | (D['ano'] > prim))].copy()
D = D.reset_index(drop=True)
print('amostra', len(D), 'DMU-ano', D['dmu'].nunique(), 'DMUs')
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
X = D['pmso'].values; Y = D[OUTS].values.astype(float)
# restrições de pesos (NT 160/2017, Tabela 3), insumo em R$ mil:
#  u_km230/v 2,5-8,5; u_km<230/u_km230 0,20-0,75; u_mva/v 0,4-4,0; u_manobra/v 15-70; u_equip/u_manobra 1-10
K = 1.2766  # IPCA jan/2017 -> dez/2021: limites monetários da NT 160/2017 levados a preços de dez/2021
AR = [('u', 0, 'v', 0, 2.5 * K, 8.5 * K), ('u', 1, 'u', 0, 0.20, 0.75), ('u', 2, 'v', 0, 0.4 * K, 4.0 * K),
      ('u', 4, 'v', 0, 15 * K, 70 * K), ('u', 3, 'u', 4, 1, 10)]
res = {}
for nm, rts, ar in [('CRS', 'crs', None), ('VRS', 'vrs', None), ('NDRS', 'ndrs', None), ('NDRS_AR', 'ndrs', AR), ('CRS_AR', 'crs', AR)]:
    e, W = dea_mult(X, Y, rts, ar)
    D['ef_' + nm] = e
    res[nm] = e
    print(nm, 'média', np.nanmean(e).round(3), 'eficientes', int((e > 0.9999).sum()))
# super-eficiência (VRS e NDRS_AR não têm solução para todas; usa CRS) para diagnóstico de atípicos
sup = np.full(len(D), np.nan)
for o in range(len(D)):
    m = np.arange(len(D)) != o
    sup[o] = dea_env(X[:, None], Y, 'crs', Xref=X[m][:, None], Yref=Y[m], idx=[o])[o]
D['super_CRS'] = sup
# B) bootstrap VRS
bs = bootstrap_sw(X, Y, 'vrs', B=1000, seed=42, n_jobs=-1)
D['ef_VRS_bc'] = bs['theta_bc']; D['ef_VRS_lo'] = bs['lo']; D['ef_VRS_hi'] = bs['hi']
print('bootstrap h', round(bs['h'], 4), 'média bc', D['ef_VRS_bc'].mean().round(3),
      'largura média IC', (D['ef_VRS_hi'] - D['ef_VRS_lo']).mean().round(3))
D.to_csv(os.path.join(R, 'eficiencias_dmu_ano.csv'), index=False)

# C) por DMU (média dos anos), normalização e custo reconhecido em 2021
specs = ['CRS', 'VRS', 'NDRS', 'NDRS_AR', 'CRS_AR', 'VRS_bc']
E = D.groupby('dmu').agg(nome=('nome', 'first'), anos=('ano', 'nunique'), pmso_med=('pmso', 'mean'),
                         **{s: ('ef_' + s, 'mean') for s in specs}).reset_index()
last = D[D['ano'] == 2021].set_index('dmu')['pmso']
E['pmso_2021'] = E['dmu'].map(last)
rec = {}
for s in specs:
    alvo = E.loc[E[s] >= 0.5, s].mean()
    E['idx_' + s] = E[s] / alvo
    E['rec_' + s] = E['pmso_2021'] * E['idx_' + s]
    rec[s] = dict(alvo=alvo, custo_2021=E['pmso_2021'].sum(), reconhecido=E['rec_' + s].sum())
E['rec_min'] = E[['rec_' + s for s in specs[:5]]].min(axis=1); E['rec_max'] = E[['rec_' + s for s in specs[:5]]].max(axis=1)
E['amplitude_%'] = 100 * (E['rec_max'] - E['rec_min']) / E['pmso_2021']
E.to_csv(os.path.join(R, 'eficiencias_dmu.csv'), index=False)
rec = pd.DataFrame(rec).T; rec['var_%'] = 100 * (rec['reconhecido'] / rec['custo_2021'] - 1)
rec.to_csv(os.path.join(R, 'custo_reconhecido.csv'))
print(rec.round(3))
sp = E[specs].corr(method='spearman'); sp.to_csv(os.path.join(R, 'spearman_specs.csv')); print(sp.round(2))
print('amplitude mediana do custo reconhecido entre especificações (% do PMSO 2021):', E['amplitude_%'].median().round(1))
