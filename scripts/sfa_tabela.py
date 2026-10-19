"""Tabela completa da fronteira de custo estocástica (Cobb-Douglas, normal-meia-normal): coeficientes, EP (Hessiana),
sigma_u, sigma_v, lambda, log-verossimilhança; correlação entre os logaritmos dos produtos; especificações alternativas
para a soma das elasticidades (IC por bootstrap de agrupamentos, 300 réplicas).
Saídas: resultados/sfa_tabela.csv, resultados/sfa_corr_produtos.csv, resultados/sfa_spec_alt.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats, optimize
import numdifftools as nd
R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano_todas.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
y_all = np.log(D.pmso.values)

def fit(X, y):
    def nll(th):
        b = th[:-2]; sv, su = np.exp(th[-2:]); e = y - X @ b; s = np.hypot(sv, su)
        return -np.sum(np.log(2) - np.log(s) + stats.norm.logpdf(e / s) + stats.norm.logcdf(e * (su / sv) / s))
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    r = optimize.minimize(nll, np.r_[b0, np.log(.3), np.log(.5)], method='BFGS')
    return r, nll

# 1) tabela principal
X = np.c_[np.ones(len(D)), np.log1p(D[OUTS].values)]
r, nll = fit(X, y_all)
H = nd.Hessian(nll)(r.x); V = np.linalg.inv(H); se = np.sqrt(np.diag(V))
names = ['const'] + ['ln(1+' + c + ')' for c in OUTS]
rows = [dict(parametro=n, coef=r.x[i], ep=se[i], z=r.x[i] / se[i], p=2 * stats.norm.sf(abs(r.x[i] / se[i]))) for i, n in enumerate(names)]
sv, su = np.exp(r.x[-2:])
# EP de sigma pelo método delta (log-param.)
rows += [dict(parametro='sigma_v', coef=sv, ep=sv * se[-2], z=np.nan, p=np.nan), dict(parametro='sigma_u', coef=su, ep=su * se[-1], z=np.nan, p=np.nan),
         dict(parametro='lambda = sigma_u/sigma_v', coef=su / sv, ep=np.nan, z=np.nan, p=np.nan),
         dict(parametro='gamma = sigma_u2/(sigma_u2+sigma_v2)', coef=su ** 2 / (su ** 2 + sv ** 2), ep=np.nan, z=np.nan, p=np.nan),
         dict(parametro='log-likelihood', coef=-r.fun, ep=np.nan, z=np.nan, p=np.nan), dict(parametro='n', coef=len(D), ep=np.nan, z=np.nan, p=np.nan)]
g = np.r_[0, np.ones(5), 0, 0]; s = g @ r.x; ses = np.sqrt(g @ V @ g)
rows.append(dict(parametro='sum of elasticities (delta method)', coef=s, ep=ses, z=(s - 1) / ses, p=2 * stats.norm.sf(abs((s - 1) / ses))))
T = pd.DataFrame(rows); T.to_csv(os.path.join(R, 'sfa_tabela.csv'), index=False); print(T.round(4).to_string())
# OLS para comparação
import statsmodels.api as sm
m = sm.OLS(y_all, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.dmu)[0]})
print('OLS', np.round(m.params, 3), 'EP cluster', np.round(m.bse, 3))
# 2) correlação entre logaritmos dos produtos
Cc = pd.DataFrame(X[:, 1:], columns=OUTS).corr(); Cc.to_csv(os.path.join(R, 'sfa_corr_produtos.csv')); print(Cc.round(3))
vif = [1 / (1 - sm.OLS(X[:, i], np.delete(X, i, axis=1)).fit().rsquared) for i in range(1, 6)]
print('VIF', dict(zip(OUTS, np.round(vif, 1))))
print('zeros: km_ge230', int((D.km_ge230 == 0).sum()), 'km_lt230', int((D.km_lt230 == 0).sum()), 'mva', int((D.mva == 0).sum()), 'equip', int((D.equip == 0).sum()))

# 3) especificações alternativas da soma de elasticidades (IC por bootstrap de agrupamentos)
D['km'] = D.km_ge230 + D.km_lt230; D['modulos'] = D.equip + D.manobra
specs = {
    'A: five outputs, ln(1+x) [paper]': OUTS,
    'B: three outputs (km, MVA+Mvar, equipment+switching modules)': ['km', 'mva', 'modulos'],
    'C: two outputs (km, switching modules)': ['km', 'manobra'],
    'D: five outputs, ln(x + 1% of the median)': OUTS,
}
rng = np.random.default_rng(7); G = D.dmu.unique()
def design(d, cols, mode):
    if mode == 'D':
        return np.c_[np.ones(len(d)), np.column_stack([np.log(d[c].values + 0.01 * np.median(D.loc[D[c] > 0, c])) for c in cols])]
    return np.c_[np.ones(len(d)), np.log1p(d[cols].values)]
rows = []
for nm, cols in specs.items():
    mode = nm[0]
    Xs = design(D, cols, mode); rr, _ = fit(Xs, y_all); soma = rr.x[1:len(cols) + 1].sum()
    S = []
    for b in range(300):
        pick = rng.choice(G, len(G), replace=True); d = pd.concat([D[D.dmu == k] for k in pick])
        rb, _ = fit(design(d, cols, mode), np.log(d.pmso.values)); S.append(rb.x[1:len(cols) + 1].sum())
    lo, hi = np.percentile(S, [2.5, 97.5])
    el = dict(zip(cols, np.round(rr.x[1:len(cols) + 1], 3)))
    rows.append(dict(spec=nm, soma=soma, lo=lo, hi=hi, elasticidades=el, sigma_u=np.exp(rr.x[-1]), sigma_v=np.exp(rr.x[-2]), loglik=-rr.fun))
    print(nm, round(soma, 3), np.round([lo, hi], 3), el)
A = pd.DataFrame(rows); A.to_csv(os.path.join(R, 'sfa_spec_alt.csv'), index=False)
