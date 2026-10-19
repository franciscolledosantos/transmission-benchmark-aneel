"""Análises da revisão de 17/10/2026:
1) Interpretação do LightGBM monotônico: contribuições SHAP (TreeSHAP nativo do LightGBM) e elasticidade de escala
   implícita (aumento proporcional de todos os produtos em 10%), para legadas e demais; idem translog-ridge.
2) Números menores: painel balanceado 2017 x 2021, mediana por agrupamento, NDRS+WR com limites a preços de 2017,
   agrupamentos só de subestação, n de agrupamentos >= 0,5 por especificação, R2 do segundo estágio sem efeitos de ano,
   sensibilidade do segundo estágio (indisponibilidade zero -> ausente; Amazônia pela UF dos módulos).
Saídas: resultados/ml_interpretacao.csv, resultados/ml_shap_importancia.csv, resultados/extras_revisao.csv,
        resultados/segundo_estagio_sens.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from dea import dea_mult
R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
B = os.path.join(os.path.dirname(__file__), '..', 'dados')
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano_todas.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
ENV = ['idade', 'fogo', 'frac_amz0', 'log_km_se', 'log_hint']
L = ['l_' + c for c in OUTS]
leg = ['AG394', 'AG48', 'AG372', 'AG97']
y = D['lnC'].values
out = {}

# 1) ML
par = dict(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8, verbose=-1)
m = lgb.LGBMRegressor(monotone_constraints=[1] * 5, **par).fit(D[L].values, y)
me = lgb.LGBMRegressor(monotone_constraints=[1] * 5 + [0] * 5, **par).fit(D[L + ENV].values, y)
def shap_imp(model, cols):
    c = model.predict(D[cols].values, pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(c).mean(0), index=cols)
imp = pd.DataFrame({'LightGBM monotone': shap_imp(m, L), 'LightGBM monotone + env': shap_imp(me, L + ENV)})
imp['share_env_model_%'] = 100 * imp['LightGBM monotone + env'] / imp['LightGBM monotone + env'].sum()
imp.to_csv(os.path.join(R, 'ml_shap_importancia.csv')); print(imp.round(3))
# translog-ridge (no conjunto completo)
TL = L + [f'{a}*{b}' for i, a in enumerate(L) for b in L[i:]]
def design(df):
    M = pd.DataFrame(index=df.index)
    for c in TL:
        if '*' in c:
            a, b = c.split('*'); M[c] = df[a] * df[b]
        else:
            M[c] = df[c]
    return M.values
tr = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(design(D), y)
k = np.log(1.10)
def scaled(df, f):
    d = df.copy()
    for c in OUTS:
        d['l_' + c] = np.log1p(d[c] * f)
    return d
up, dn = scaled(D, 1.10), scaled(D, 1 / 1.10)
D['el_lgb'] = (m.predict(up[L].values) - m.predict(dn[L].values)) / (2 * k)
D['el_tl'] = (tr.predict(design(up)) - tr.predict(design(dn))) / (2 * k)
# elasticidade de escala em Cobb-Douglas = soma das elasticidades (constante)
D['grupo'] = np.where(D.dmu.isin(leg), 'legacy', np.where(D.pmso < D.pmso.median(), 'smaller half', 'larger half (excl. legacy)'))
I = D.groupby('grupo')[['el_lgb', 'el_tl']].median()
I.loc['all'] = D[['el_lgb', 'el_tl']].median()
# por quintil de custo
D['q'] = pd.qcut(D.pmso, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
I2 = D.groupby('q')[['el_lgb', 'el_tl']].median()
I = pd.concat([I, I2]); I.to_csv(os.path.join(R, 'ml_interpretacao.csv')); print(I.round(3))
# curva de dependência: custo previsto para um agrupamento "mediano" escalado de 0,1x a 30x
base = D[OUTS].median()
grid = np.array([0.1, 0.3, 1, 3, 10, 30])
pdp = []
for f in grid:
    row = pd.DataFrame([{('l_' + c): np.log1p(base[c] * f) for c in OUTS}])
    pdp.append(dict(fator=f, lgb_lnC=m.predict(row[L].values)[0]))
pdp = pd.DataFrame(pdp); pdp['elasticidade_arco'] = pdp.lgb_lnC.diff() / np.log(pdp.fator).diff()
pdp.to_csv(os.path.join(R, 'ml_curva_escala.csv'), index=False); print(pdp.round(3))
# previsão fora da amostra para as legadas (média do resíduo)
for nm in ['LightGBM monotone', 'Translog (ridge)', 'Cobb-Douglas (OLS)']:
    out['res_medio_legadas_' + nm] = D.loc[D.dmu.isin(leg), 'res_' + nm].mean()

# 2) números menores
P = D.copy()
b17 = set(P[P.ano == 2017].dmu) & set(P[P.ano == 2021].dmu)
s17 = P[(P.ano == 2017) & P.dmu.isin(b17)]; s21 = P[(P.ano == 2021) & P.dmu.isin(b17)]
out.update(bal_n=len(b17), bal_pmso17=s17.pmso.sum() / 1e6, bal_pmso21=s21.pmso.sum() / 1e6, bal_var_pmso_pct=100 * (s21.pmso.sum() / s17.pmso.sum() - 1),
           bal_km17=s17.km.sum(), bal_km21=s21.km.sum(), bal_var_km_pct=100 * (s21.km.sum() / s17.km.sum() - 1),
           n_dmu_2017=P[P.ano == 2017].dmu.nunique(), n_dmu_2021=P[P.ano == 2021].dmu.nunique(),
           all_pmso17=P[P.ano == 2017].pmso.sum() / 1e6, all_pmso21=P[P.ano == 2021].pmso.sum() / 1e6,
           all_km17=P[P.ano == 2017].km.sum(), all_km21=P[P.ano == 2021].km.sum(),
           mediana_obs_pmso=P.pmso.median(), mediana_media_dmu_pmso=P.groupby('dmu').pmso.mean().median(),
           max_obs_pmso=P.pmso.max())
so = P.groupby('dmu').km.max() == 0
out.update(n_dmu_so_subestacao=int(so.sum()), n_obs_so_subestacao=int(P.dmu.isin(so[so].index).sum()),
           share_custo_so_subestacao=P[P.dmu.isin(so[so].index)].pmso.sum() / P.pmso.sum())
E = pd.read_csv(os.path.join(R, 'eficiencias_dmu.csv'))
for s in ['CRS', 'VRS', 'NDRS', 'NDRS_AR', 'CRS_AR', 'VRS_bc']:
    out['n_ge05_' + s] = int((E[s] >= 0.5).sum())
# NDRS+WR com limites a preços de jan/2017 (sem indexação)
X = P.pmso.values; Y = P[OUTS].values.astype(float)
for nmK, K in [('K1', 1.0), ('K1277', 1.2766)]:
    AR = [('u', 0, 'v', 0, 2.5 * K, 8.5 * K), ('u', 1, 'u', 0, 0.20, 0.75), ('u', 2, 'v', 0, 0.4 * K, 4.0 * K),
          ('u', 4, 'v', 0, 15 * K, 70 * K), ('u', 3, 'u', 4, 1, 10)]
    P['e'] = dea_mult(X, Y, 'ndrs', AR)[0]; Em = P.groupby('dmu').e.mean(); alvo = Em[Em >= .5].mean()
    p21 = P[P.ano == 2021].set_index('dmu').pmso
    out['NDRS_AR_var_pct_' + nmK] = 100 * ((p21 * Em.reindex(p21.index) / alvo).sum() / p21.sum() - 1)
    out['NDRS_AR_media_' + nmK] = P.e.mean()

# segundo estágio: R2 sem efeitos de ano e R2 parcial; sensibilidade (zeros de indisponibilidade, Amazônia por UF dos módulos)
A = pd.read_parquet(os.path.join(B, 'ativos_2017_2021.parquet'))
AMZ = {'AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO', 'MT', 'MA'}
A['amz'] = A['uf'].isin(AMZ)
fam = A.groupby(['dmu', 'ano']).amz.mean().rename('frac_amz_mod').reset_index()
P = P.merge(fam, on=['dmu', 'ano'], how='left')
P['frac_amz_alt'] = np.where(P.km > 0, P.frac_amz0, P.frac_amz_mod)
# registros de indisponibilidade
F = os.path.join(B, 'ts14_f1', 'BASES DE DADOS', 'BASE DE DADOS - 2017 a 2021 - ABERTURA TS.xlsx')
h = pd.read_excel(F, 'AGRUPAMENTO - Holdings', header=None, skiprows=1); h[0] = h[0].ffill(); grp = dict(zip(h[1].astype(int), h[0]))
q = pd.read_excel(F, 'INDISPONILIDADE', header=None, skiprows=1).iloc[:, :6]
q.columns = ['IdAgente', 'emp', 'contrato', 'ano', 'holding', 'h_int']
q = q[pd.to_numeric(q['IdAgente'], errors='coerce').notna()].copy(); q['IdAgente'] = q.IdAgente.astype(int); q['ano'] = q.ano.astype(int)
q['dmu'] = q['IdAgente'].map(grp).fillna(q['IdAgente'].map(lambda i: f'AG{i}'))
reg = set(zip(q.dmu, q.ano))
P['tem_registro'] = [(a, b) in reg for a, b in zip(P.dmu, P.ano)]
out['n_hint_zero'] = int((P.h_int == 0).sum()); out['n_sem_registro_indisp'] = int((~P.tem_registro).sum())
out['n_so_subestacao_com_amz'] = int(((P.km == 0) & (P.frac_amz_mod > 0)).sum())
P['log_hint_na'] = np.where(P.tem_registro, P.log_hint, np.nan)
rows = []
for dep in ['ef_NDRS_AR', 'ef_VRS_bc', 'ef_SFA', 'ef_ML']:
    for var, f, amzv, hv in [('original', 'C(ano)', 'frac_amz0', 'log_hint'), ('sem efeitos de ano', None, 'frac_amz0', 'log_hint'),
                             ('Amazônia por UF dos módulos; sem registro de indisponibilidade = ausente', 'C(ano)', 'frac_amz_alt', 'log_hint_na')]:
        d = P.dropna(subset=[dep, 'idade', 'fogo', amzv, 'log_km_se', hv]).copy(); d['y2'] = np.log(d[dep])
        form = f'y2 ~ idade + fogo + {amzv} + log_km_se + {hv}' + (f' + {f}' if f else '')
        mm = smf.ols(form, d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['dmu'])[0]})
        r0 = smf.ols('y2 ~ C(ano)', d).fit().rsquared if f else 0.0
        for v in ['idade', 'fogo', amzv, 'log_km_se', hv]:
            rows.append(dict(dep=dep, variante=var, var=v, coef=mm.params[v], se=mm.bse[v], p=mm.pvalues[v], n=int(mm.nobs),
                             r2=mm.rsquared, r2_ano=r0, r2_parcial_condicoes=(mm.rsquared - r0) / (1 - r0)))
S2 = pd.DataFrame(rows); S2.to_csv(os.path.join(R, 'segundo_estagio_sens.csv'), index=False)
print(S2.groupby(['dep', 'variante'])[['n', 'r2', 'r2_ano', 'r2_parcial_condicoes']].first().round(3))
print(S2[S2.p < 0.1].round(4).to_string())
pd.Series(out).to_csv(os.path.join(R, 'extras_revisao.csv')); print(pd.Series(out).round(3).to_string())
