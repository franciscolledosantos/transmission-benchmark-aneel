"""DEA orientado a insumo com um ou mais insumos: forma do envelope e forma dos multiplicadores
com restrições de pesos (regiões de segurança do tipo razão), retornos CRS/VRS/NDRS/NIRS e
bootstrap suavizado de Simar & Wilson (1998).

Convenção: X (n x m) insumos, Y (n x s) produtos; eficiência theta em (0, 1].
"""
import numpy as np
from scipy.optimize import linprog


def dea_env(X, Y, rts='vrs', Xref=None, Yref=None, idx=None):
    """Forma do envelope: min theta s.a. Xref' lam <= theta x_o, Yref' lam >= y_o, restrição de escala.
    Xref/Yref permitem avaliar unidades contra outra tecnologia (super-eficiência, bootstrap)."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Xr = X if Xref is None else (np.atleast_2d(Xref.T).T if Xref.ndim == 1 else Xref)
    Yr = Y if Yref is None else Yref
    n, m = X.shape; s = Y.shape[1]; k = Xr.shape[0]
    idx = range(n) if idx is None else idx
    out = np.full(n, np.nan)
    for o in idx:
        c = np.r_[1.0, np.zeros(k)]
        A = np.vstack([np.c_[-X[o][:, None], Xr.T], np.c_[np.zeros((s, 1)), -Yr.T]])
        b = np.r_[np.zeros(m), -Y[o]]
        Aeq = beq = None
        if rts == 'vrs':
            Aeq, beq = np.r_[0.0, np.ones(k)][None, :], [1.0]
        elif rts == 'ndrs':      # sum(lam) >= 1
            A = np.vstack([A, np.r_[0.0, -np.ones(k)]]); b = np.r_[b, -1.0]
        elif rts == 'nirs':      # sum(lam) <= 1
            A = np.vstack([A, np.r_[0.0, np.ones(k)]]); b = np.r_[b, 1.0]
        r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)] + [(0, None)] * k, method='highs')
        out[o] = r.x[0] if r.status == 0 else np.nan
    return out


def dea_mult(X, Y, rts='vrs', ar=None):
    """Forma dos multiplicadores (um insumo ou vários), orientada a insumo:
    max u.y_o + w  s.a. v.x_o = 1,  u.y_j - v.x_j + w <= 0,  u, v >= 0.
    w livre (VRS), w = 0 (CRS), w >= 0 (NDRS, equivale a sum(lam) >= 1), w <= 0 (NIRS).
    ar: lista de (tipo_num, i, tipo_den, j, lo, hi) com lo <= peso_num/peso_den <= hi, tipo 'u' ou 'v'.
    Devolve eficiências e os pesos ótimos."""
    X = X[:, None] if X.ndim == 1 else X
    n, m = X.shape; s = Y.shape[1]
    nv = s + m + 1                      # [u (s), v (m), w]
    def pos(t, i):
        return i if t == 'u' else s + i
    Aar, bar = [], []
    for (tn, i, td, j, lo, hi) in (ar or []):
        r1 = np.zeros(nv); r1[pos(tn, i)] = -1; r1[pos(td, j)] = lo; Aar.append(r1); bar.append(0)   # num >= lo*den
        r2 = np.zeros(nv); r2[pos(tn, i)] = 1; r2[pos(td, j)] = -hi; Aar.append(r2); bar.append(0)   # num <= hi*den
    wb = {'vrs': (None, None), 'crs': (0, 0), 'ndrs': (0, None), 'nirs': (None, 0)}[rts]
    eff = np.full(n, np.nan); W = np.full((n, nv), np.nan)
    Abase = np.c_[Y, -X, np.ones(n)]
    for o in range(n):
        c = -np.r_[Y[o], np.zeros(m), 1.0]
        A = np.vstack([Abase] + ([np.array(Aar)] if Aar else [])); b = np.r_[np.zeros(n), bar]
        Aeq = np.r_[np.zeros(s), X[o], 0.0][None, :]
        r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[1.0], bounds=[(0, None)] * (s + m) + [wb], method='highs')
        if r.status == 0:
            eff[o] = -r.fun; W[o] = r.x
    return eff, W


def bootstrap_sw(X, Y, rts='vrs', B=500, seed=0, alpha=0.05, n_jobs=1):
    """Bootstrap suavizado homogêneo de Simar & Wilson (1998) para DEA orientado a insumo.
    Devolve theta, theta corrigido de viés, IC (alpha) e viés."""
    from joblib import Parallel, delayed
    X = X[:, None] if X.ndim == 1 else X
    n = len(X)
    th = dea_env(X, Y, rts)
    # largura de banda: regra de Silverman sobre os escores refletidos em torno de 1
    d = 1 / th
    refl = np.r_[d, 2 - d]
    sd = refl.std(ddof=1); iqr = np.subtract(*np.percentile(refl, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34) * (2 * n) ** (-1 / 5)
    def one(b):
        rng = np.random.default_rng(seed + b)
        beta = rng.choice(d, n, replace=True)
        eps = rng.standard_normal(n)
        bm = beta.mean()
        dstar = bm + (beta + h * eps - bm) / np.sqrt(1 + h ** 2 / np.var(d))
        dstar = np.where(dstar < 1, 2 - dstar, dstar)       # reflexão
        Xs = X * (th / (1 / dstar))[:, None]                 # x* = x * theta_hat / theta*
        return dea_env(X, Y, rts, Xref=Xs, Yref=Y)
    T = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(B)))
    # trabalhar em distâncias de Shephard (1/theta)
    D, Ds = 1 / th, 1 / T
    bias = Ds.mean(0) - D
    dbc = D - bias
    q = np.percentile(Ds - D, [100 * (1 - alpha / 2), 100 * alpha / 2], axis=0)
    lo_d, hi_d = D - q[0], D - q[1]
    return dict(theta=th, theta_bc=1 / dbc, lo=1 / hi_d, hi=1 / lo_d, bias=bias, h=h)
