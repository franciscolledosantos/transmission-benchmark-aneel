"""Sensibilidade: PMSO incluindo condenações trabalhistas (coluna 'Cond. Trabalhistas' da aba PMSO Consolidado),
deflacionadas pelo IPCA como Pessoal. Reestima os cinco DEA e a SFA e compara eficiências e custo reconhecido.
Saídas: resultados/sens_cond_trab.csv (por especificação) e resultados/sens_cond_trab_dmu.csv (por agrupamento).
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, os.path.dirname(__file__))
from dea import dea_mult

B = os.path.join(os.path.dirname(__file__), '..', 'dados')
F = os.path.join(B, 'ts14_f1', 'BASES DE DADOS', 'BASE DE DADOS - 2017 a 2021 - ABERTURA TS.xlsx')
R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
h = pd.read_excel(F, 'AGRUPAMENTO - Holdings', header=None, skiprows=1); h[0] = h[0].ffill()
grp = dict(zip(h[1].astype(int), h[0]))
pm = pd.read_excel(F, 'PMSO Consolidado', header=None, skiprows=2).iloc[:, 10:18]
pm.columns = ['IdAgente', 'conc', 'ano', 'pessoal', 'materiais', 'servicos', 'outros', 'cond_trab']
pm = pm[pd.to_numeric(pm['IdAgente'], errors='coerce').notna()].copy()
pm['IdAgente'] = pm['IdAgente'].astype(int); pm['ano'] = pm['ano'].astype(int)
pm['cond_trab'] = pd.to_numeric(pm['cond_trab'], errors='coerce').fillna(0)
pm['pmso_nom4'] = pm[['pessoal', 'materiais', 'servicos', 'outros']].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
ix = pd.read_excel(F, 'IPCA-IGPM', header=None, skiprows=1, names=['mes', 'igpm', 'ipca'])
ix['mes'] = pd.to_datetime(ix['mes'], errors='coerce'); ix = ix.dropna().set_index('mes')
ref = ix.loc['2021-12-01']; anual = ix.groupby(ix.index.year).mean()
pm['f_ipca'] = pm['ano'].map(ref['ipca'] / anual['ipca'])
pm['cond_real'] = pm['cond_trab'] * pm['f_ipca']
pm['dmu'] = pm['IdAgente'].map(grp).fillna(pm['IdAgente'].map(lambda i: f'AG{i}'))
C = pm.groupby(['dmu', 'ano']).agg(cond_real=('cond_real', 'sum'), cond_nom=('cond_trab', 'sum'), pmso_nom4=('pmso_nom4', 'sum')).reset_index()
print('condenações nominais 2017-2021 (R$ mil):', round(pm.cond_trab.sum()), 'agente-anos com condenação:', int((pm.cond_trab > 0).sum()))
print('participação no PMSO nominal 2021:', round(100 * pm[pm.ano == 2021].cond_trab.sum() / pm[pm.ano == 2021].pmso_nom4.sum(), 2), '%')
D = D.merge(C, on=['dmu', 'ano'], how='left'); D['cond_real'] = D['cond_real'].fillna(0)
D['pmso_ct'] = D['pmso'] + D['cond_real']
D['share_ct'] = D['cond_real'] / D['pmso_ct']
leg = ['AG394', 'AG48', 'AG372', 'AG97']
print(D.groupby('dmu').apply(lambda d: d.cond_real.sum() / d.pmso_ct.sum()).sort_values(ascending=False).head(10).round(3))

K = 1.2766
AR = [('u', 0, 'v', 0, 2.5 * K, 8.5 * K), ('u', 1, 'u', 0, 0.20, 0.75), ('u', 2, 'v', 0, 0.4 * K, 4.0 * K),
      ('u', 4, 'v', 0, 15 * K, 70 * K), ('u', 3, 'u', 4, 1, 10)]
Y = D[OUTS].values.astype(float)

def sfa_eff(x):
    y = np.log(x); X = np.c_[np.ones(len(D)), np.log1p(D[OUTS].values)]
    def nll(th):
        b = th[:-2]; sv, su = np.exp(th[-2:]); e = y - X @ b; s = np.hypot(sv, su)
        return -np.sum(np.log(2) - np.log(s) + stats.norm.logpdf(e / s) + stats.norm.logcdf(e * (su / sv) / s))
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]; r = optimize.minimize(nll, np.r_[b0, np.log(.3), np.log(.5)], method='BFGS')
    b = r.x[:-2]; sv, su = np.exp(r.x[-2:]); e = y - X @ b; s2 = sv ** 2 + su ** 2
    mu = e * su ** 2 / s2; ss = su * sv / np.sqrt(s2)
    return np.exp(-(mu + ss * stats.norm.pdf(mu / ss) / stats.norm.cdf(mu / ss))), b[1:].sum()

rows = []; per = {}
for var, col in [('líquido (artigo)', 'pmso'), ('incluindo condenações', 'pmso_ct')]:
    X = D[col].values
    for spec, rts, ar in [('CRS', 'crs', None), ('VRS', 'vrs', None), ('NDRS', 'ndrs', None), ('NDRS_AR', 'ndrs', AR), ('CRS_AR', 'crs', AR)]:
        D['e'] = dea_mult(X, Y, rts, ar)[0]
        E = D.groupby('dmu').e.mean(); alvo = E[E >= .5].mean()
        p21 = D[D.ano == 2021].set_index('dmu')[col]; rec = (p21 * E.reindex(p21.index) / alvo).sum()
        rows.append(dict(variante=var, spec=spec, media_dmu=E.mean(), legadas=E.reindex(leg).mean(), alvo=alvo,
                         var_pct=100 * (rec / p21.sum() - 1), furnas=E['AG394'], chesf=E['AG48'], eletronorte=E['AG372'],
                         cgt=E['AG97'], ceee_t=E['AG5785'], cemig=E['AG4951'], copel=E['COPEL_HOL']))
        per[(var, spec)] = E
    ef, soma = sfa_eff(X); D['e'] = ef; E = D.groupby('dmu').e.mean()
    rows.append(dict(variante=var, spec='SFA', media_dmu=E.mean(), legadas=E.reindex(leg).mean(), alvo=np.nan, var_pct=np.nan,
                     furnas=E['AG394'], chesf=E['AG48'], eletronorte=E['AG372'], cgt=E['AG97'], ceee_t=E['AG5785'],
                     cemig=E['AG4951'], copel=E['COPEL_HOL']))
    print(var, 'SFA soma elasticidades', round(soma, 3))
    per[(var, 'SFA')] = E
S = pd.DataFrame(rows); S.to_csv(os.path.join(R, 'sens_cond_trab.csv'), index=False)
print(S.round(3).to_string())
T = pd.DataFrame({f'{v}|{s}': per[(v, s)] for (v, s) in per})
T['share_cond_2017_21'] = D.groupby('dmu').apply(lambda d: d.cond_real.sum() / d.pmso_ct.sum())
T['nome'] = D.groupby('dmu').nome.first()
T.to_csv(os.path.join(R, 'sens_cond_trab_dmu.csv'))
dif = T[[c for c in T.columns if c.startswith('incluindo')]].values - T[[c for c in T.columns if c.startswith('líquido')]].values
print('maior queda de escore por agrupamento (CRS):', T.assign(d=T['incluindo condenações|CRS'] - T['líquido (artigo)|CRS']).sort_values('d')[['nome', 'd', 'share_cond_2017_21']].head(6).round(3))
