"""Conjuntos de referência (pares, lambdas) do DEA CRS, supereficiência e sensibilidade do custo reconhecido
à exclusão das observações que sustentam a fronteira CRS (LTTE 2019, ONTE 2019, CPFL Transmissão 2019).
Saídas: resultados/peers_CRS.csv, resultados/peers_legadas_2021.csv, resultados/sens_excl_peers.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(__file__))
from dea import dea_mult

R = os.path.join(os.path.dirname(__file__), '..', 'resultados')
D = pd.read_csv(os.path.join(R, 'eficiencias_dmu_ano.csv'))
OUTS = ['km_ge230', 'km_lt230', 'mva', 'equip', 'manobra']
X = D['pmso'].values; Y = D[OUTS].values.astype(float); n = len(D)
D['rot'] = D['nome'] + ' ' + D['ano'].astype(str)

def env_lambda(o, rts='crs', mask=None):
    """Envelope CRS/VRS com lambdas (unidade o contra a referência mask)."""
    ref = np.arange(n) if mask is None else np.where(mask)[0]; k = len(ref)
    c = np.r_[1.0, np.zeros(k)]
    A = np.vstack([np.r_[-X[o], X[ref]][None, :], np.c_[np.zeros((Y.shape[1], 1)), -Y[ref].T]])
    b = np.r_[0.0, -Y[o]]
    Aeq = beq = None
    if rts == 'vrs':
        Aeq, beq = np.r_[0.0, np.ones(k)][None, :], [1.0]
    r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)] + [(0, None)] * k, method='highs')
    lam = np.zeros(n); lam[ref] = r.x[1:]
    return r.x[0], lam

# 1) pares sob CRS (fronteira conjunta)
TH = np.zeros(n); LAM = np.zeros((n, n))
for o in range(n):
    TH[o], LAM[o] = env_lambda(o, 'crs')
peer = (LAM > 1e-6)
cnt = peer.sum(0)
P = pd.DataFrame({'dmu': D.dmu, 'nome': D.nome, 'ano': D.ano, 'pmso': D.pmso.round(0), 'km': D.km.round(0),
                  'mva': D.mva, 'equip': D.equip, 'manobra': D.manobra, 'ef_CRS': D.ef_CRS.round(4),
                  'super_CRS': D.super_CRS.round(4), 'n_referenciadas': cnt,
                  'soma_lambda': LAM.sum(0).round(2), 'lambda_max': LAM.max(0).round(2)})
P = P[P.n_referenciadas > 0].sort_values('n_referenciadas', ascending=False)
P.to_csv(os.path.join(R, 'peers_CRS.csv'), index=False)
print(P.head(12).to_string())
# pares sob VRS para as legadas (quantas vezes a unidade é seu próprio par)
LV = np.zeros((n, n)); THV = np.zeros(n)
for o in range(n):
    THV[o], LV[o] = env_lambda(o, 'vrs')
self_peer = np.array([LV[o, o] > 1e-6 for o in range(n)])
leg = ['AG394', 'AG48', 'AG372', 'AG97']
rows = []
for o in np.where(D.dmu.isin(leg) & (D.ano == 2021))[0]:
    pc = [(D.rot[j], round(LAM[o, j], 2)) for j in np.where(LAM[o] > 1e-6)[0]]
    pv = [(D.rot[j], round(LV[o, j], 2)) for j in np.where(LV[o] > 1e-6)[0]]
    rows.append(dict(dmu=D.dmu[o], nome=D.nome[o], ef_CRS=round(TH[o], 3), pares_CRS=pc, soma_lambda_CRS=round(LAM[o].sum(), 1),
                     ef_VRS=round(THV[o], 3), pares_VRS=pv))
pd.DataFrame(rows).to_csv(os.path.join(R, 'peers_legadas_2021.csv'), index=False)
for r in rows: print(r)
print('unidades VRS eficientes que são seu único par:', int(sum((THV > 0.9999) & self_peer & ((LV > 1e-6).sum(1) == 1))))

# 2) sensibilidade: excluir observações da referência (mas mantê-las avaliadas? não: exclusão completa da amostra)
K = 1.2766
AR = [('u', 0, 'v', 0, 2.5 * K, 8.5 * K), ('u', 1, 'u', 0, 0.20, 0.75), ('u', 2, 'v', 0, 0.4 * K, 4.0 * K),
      ('u', 4, 'v', 0, 15 * K, 70 * K), ('u', 3, 'u', 4, 1, 10)]
excl = {'completa': [], 'sem LTTE 2019': ['LTTE 2019'], 'sem LTTE, ONTE e CPFL T 2019': ['LTTE 2019', 'ONTE 2019', 'CPFL TRANSMISSÃO 2019'],
        'sem as 7 CRS-eficientes': list(D.rot[D.ef_CRS > 0.9999])}
rows = []
for nm, ex in excl.items():
    S = D[~D.rot.isin(ex)].reset_index(drop=True); Xs = S.pmso.values; Ys = S[OUTS].values.astype(float)
    for spec, rts, ar in [('CRS', 'crs', None), ('VRS', 'vrs', None), ('NDRS', 'ndrs', None), ('NDRS_AR', 'ndrs', AR), ('CRS_AR', 'crs', AR)]:
        S['e'] = dea_mult(Xs, Ys, rts, ar)[0]; E = S.groupby('dmu').e.mean(); alvo = E[E >= .5].mean()
        p21 = S[S.ano == 2021].set_index('dmu').pmso; idx = E.reindex(p21.index) / alvo
        rec = (p21 * idx).sum(); legm = E.reindex(leg).mean(); outras = E[~E.index.isin(leg)].mean()
        rows.append(dict(amostra=nm, n=len(S), spec=spec, media=E.mean(), legadas=legm, outras=outras, alvo=alvo,
                         n_ge05=int((E >= .5).sum()), var_pct=100 * (rec / p21.sum() - 1),
                         furnas=E.get('AG394', np.nan), chesf=E.get('AG48', np.nan)))
    print(nm, 'excluídas', len(ex))
sens = pd.DataFrame(rows); sens.to_csv(os.path.join(R, 'sens_excl_peers.csv'), index=False)
print(sens.round(3).to_string())
