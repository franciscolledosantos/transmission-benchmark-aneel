"""D) Normas de custo (média e fronteira estocástica) e ML com validação deixando a DMU de fora.
E) Segundo estágio: fatores ambientais e de qualidade (o que a ANEEL testou em 2018: idade, queimadas, adensamento).
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
ENV = ['idade', 'fogo', 'frac_amz0', 'log_km_se', 'log_hint']
D['lnC'] = np.log(D['pmso'])
for c in OUTS:
    D['l_' + c] = np.log1p(D[c])
D['frac_amz0'] = D['frac_amz'].fillna(0)
D['log_km_se'] = np.log1p(D['km_por_se'])
D['log_hint'] = np.log1p(D['h_int'])
L = ['l_' + c for c in OUTS]
TL = L + [f'{a}*{b}' for i, a in enumerate(L) for b in L[i:]]
def design(df, cols):
    M = pd.DataFrame(index=df.index)
    for c in cols:
        if '*' in c:
            a, b = c.split('*'); M[c] = df[a] * df[b]
        else:
            M[c] = df[c]
    return M.values
y = D['lnC'].values; g = D['dmu'].values
gkf = GroupKFold(n_splits=D['dmu'].nunique())
models = {
    'Cobb-Douglas (OLS)': lambda: RidgeCV(alphas=[1e-6]),
    'Translog (ridge)': lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)),
    'Random forest': lambda: RandomForestRegressor(500, min_samples_leaf=3, random_state=0, n_jobs=-1),
    'LightGBM monotone': lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                                                  monotone_constraints=[1] * 5, verbose=-1),
    'LightGBM monotone + env': lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                                                        monotone_constraints=[1] * 5 + [0] * 5, verbose=-1),
}
cols = {'Cobb-Douglas (OLS)': L, 'Translog (ridge)': TL, 'Random forest': L, 'LightGBM monotone': L,
        'LightGBM monotone + env': L + ENV}
pred = {}
for nm, mk in models.items():
    Xm = design(D, cols[nm]); p = np.zeros(len(D))
    for tr, te in gkf.split(Xm, y, g):
        m = mk().fit(Xm[tr], y[tr]); p[te] = m.predict(Xm[te])
    pred[nm] = p
# SFA normal-meia-normal (custo: e = v + u, u >= 0), Cobb-Douglas
Xc = np.c_[np.ones(len(D)), design(D, L)]
def nll(th):
    b = th[:-2]; sv, su = np.exp(th[-2]), np.exp(th[-1])
    e = y - Xc @ b; s = np.hypot(sv, su); lam = su / sv
    return -np.sum(np.log(2) - np.log(s) + stats.norm.logpdf(e / s) + stats.norm.logcdf(e * lam / s))
b0 = np.linalg.lstsq(Xc, y, rcond=None)[0]
th0 = np.r_[b0, np.log(0.3), np.log(0.5)]
sfa = optimize.minimize(nll, th0, method='BFGS')
b = sfa.x[:-2]; sv, su = np.exp(sfa.x[-2:]); e = y - Xc @ b
s2 = sv ** 2 + su ** 2; mu_s = e * su ** 2 / s2; s_s = su * sv / np.sqrt(s2)
Eu = mu_s + s_s * stats.norm.pdf(mu_s / s_s) / stats.norm.cdf(mu_s / s_s)          # JLMS
D['ef_SFA'] = np.exp(-Eu)
# SFA fora da amostra (deixa a DMU de fora): só a parte de fronteira, para comparar previsão
p_sfa = np.zeros(len(D))
for tr, te in gkf.split(Xc, y, g):
    def nll_tr(th):
        bb = th[:-2]; a, c = np.exp(th[-2]), np.exp(th[-1]); ee = y[tr] - Xc[tr] @ bb; ss = np.hypot(a, c)
        return -np.sum(np.log(2) - np.log(ss) + stats.norm.logpdf(ee / ss) + stats.norm.logcdf(ee * (c / a) / ss))
    r = optimize.minimize(nll_tr, sfa.x, method='BFGS')
    a, c = np.exp(r.x[-2:])
    p_sfa[te] = Xc[te] @ r.x[:-2] + c * np.sqrt(2 / np.pi)                              # média condicional do custo
pred['SFA (Cobb-Douglas)'] = p_sfa
rows = []
for nm, p in pred.items():
    err = y - p
    rows.append(dict(modelo=nm, RMSE_log=np.sqrt(np.mean(err ** 2)), MAE_log=np.mean(np.abs(err)),
                     MdAPE_pct=100 * np.median(np.abs(np.exp(-err) - 1)), R2=1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)))
    D['res_' + nm] = err
CV = pd.DataFrame(rows); CV.to_csv(os.path.join(R, 'normas_custo_cv.csv'), index=False); print(CV.round(3))
# eficiência "de média" do ML: exp(-resíduo) normalizada (custo previsto / custo real)
D['ef_ML'] = np.exp(-D['res_LightGBM monotone'])
D['ef_ML_env'] = np.exp(-D['res_LightGBM monotone + env'])
print('SFA: sigma_v', round(sv, 3), 'sigma_u', round(su, 3), 'lambda', round(su / sv, 2), 'ef média', D['ef_SFA'].mean().round(3))
print('elasticidades SFA', dict(zip(['const'] + L, np.round(b, 3))), 'soma', np.round(b[1:].sum(), 3))
# E) segundo estágio
res2 = []
for dep in ['ef_NDRS_AR', 'ef_VRS_bc', 'ef_SFA', 'ef_ML']:
    d = D.dropna(subset=[dep] + ENV).copy(); d['y2'] = np.log(d[dep])
    f = 'y2 ~ idade + fogo + frac_amz0 + log_km_se + log_hint + C(ano)'
    m = smf.ols(f, d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['dmu'])[0]})
    for v in ENV:
        res2.append(dict(dep=dep, var=v, coef=m.params[v], se=m.bse[v], p=m.pvalues[v], n=int(m.nobs), r2=m.rsquared))
S2 = pd.DataFrame(res2); S2.to_csv(os.path.join(R, 'segundo_estagio.csv'), index=False)
print(S2.round(4).to_string())
D.to_csv(os.path.join(R, 'eficiencias_dmu_ano_todas.csv'), index=False)
E = D.groupby('dmu')[['ef_CRS', 'ef_VRS', 'ef_NDRS', 'ef_NDRS_AR', 'ef_VRS_bc', 'ef_SFA', 'ef_ML', 'ef_ML_env']].mean()
sp = E.corr('spearman'); sp.to_csv(os.path.join(R, 'spearman_todos.csv')); print(sp.round(2))
