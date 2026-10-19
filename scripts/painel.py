"""Painel de DMUs (agrupamento operacional x ano, 2017-2021) a partir da base da TS 14/2022 da ANEEL.

Insumo: PMSO (Pessoal, Materiais, Serviços, Outros) em R$ mil de dez/2021
 (Pessoal e Serviços pelo IPCA; Materiais e Outros pelo IGP-M, como na NT 160/2017).
Produtos (famílias da NT 160/2017-SRM/ANEEL):
 km_ge230  extensão de LT >= 230 kV (circuito duplo conta metade)
 km_lt230  extensão de LT < 230 kV
 mva       potência MVA + Mvar (TR, CV, reatores, capacitores, compensadores, filtros)
 equip     número de equipamentos principais (inclui reserva)
 manobra   módulos de manobra (EL + MC + IB)
Qualidade: horas de interrupção (INDISPONIBILIDADE).
Ambiente: idade média dos ativos (LISTA-ATIVOS), fração de km na Amazônia Legal, km por subestação,
 número de UFs atendidas.
"""
import os, re
import numpy as np, pandas as pd

B = os.path.join(os.path.dirname(__file__), '..', 'dados')
F = os.path.join(B, 'ts14_f1', 'BASES DE DADOS', 'BASE DE DADOS - 2017 a 2021 - ABERTURA TS.xlsx')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
os.makedirs(OUT, exist_ok=True)
AMZ = {'AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO', 'MT', 'MA'}
EQUIP = {'TR', 'CV', 'RTL', 'RTB', 'RTS', 'RTN', 'BC', 'BF', 'CR', 'CE', 'CS', 'TT'}
COLS = ['status', 'IdeMdl', 'IdAgente', 'conc', 'contrato', 'nome', 'tipo', 'tipo2', 'kv', 'MVA', 'MvarC', 'MvarI',
        'ExtLT', 'MvarMax', 'banco', 'PR', 'CSCD', 'ExtCS', 'ExtCD']

# agrupamentos operacionais
h = pd.read_excel(F, 'AGRUPAMENTO - Holdings', header=None, skiprows=1)
h[0] = h[0].ffill()
grp = dict(zip(h[1].astype(int), h[0]))

rows, mods = [], []
for ano in range(2017, 2022):
    a = pd.read_excel(F, f'ATIVOS {ano}').iloc[:, :19]
    a.columns = COLS
    a = a[pd.to_numeric(a['IdAgente'], errors='coerce').notna()].copy()
    a['IdAgente'] = a['IdAgente'].astype(int)
    for c in ['kv', 'MVA', 'MvarC', 'MvarI', 'ExtLT', 'MvarMax', 'ExtCS', 'ExtCD']:
        a[c] = pd.to_numeric(a[c], errors='coerce').fillna(0)
    a['ano'] = ano
    a['uf'] = a['nome'].astype(str).str.extract(r'\b([A-Z]{2})\s*$')[0]
    a['km'] = np.where(a['tipo'] == 'LT', a['ExtCS'] + 0.5 * a['ExtCD'], 0.0)
    a['km'] = np.where((a['tipo'] == 'LT') & (a['km'] == 0), a['ExtLT'], a['km'])
    a['se'] = np.where(a['tipo'] == 'MG', a['nome'].astype(str).str.replace(r'^MG [\d,\.]+ kV ', '', regex=True)
                       .str.replace(r' MG\d+', '', regex=True), None)
    mods.append(a)
A = pd.concat(mods, ignore_index=True)
A['dmu'] = A['IdAgente'].map(grp).fillna(A['IdAgente'].map(lambda i: f'AG{i}'))
A.to_parquet(os.path.join(B, 'ativos_2017_2021.parquet'), index=False)

g = A.groupby(['dmu', 'ano'])
P = pd.DataFrame({
    'km_ge230': g.apply(lambda d: d.loc[(d.tipo == 'LT') & (d.kv >= 230), 'km'].sum()),
    'km_lt230': g.apply(lambda d: d.loc[(d.tipo == 'LT') & (d.kv < 230), 'km'].sum()),
    'mva': g.apply(lambda d: d.loc[d.tipo.isin(EQUIP), 'MVA'].sum() + d.loc[d.tipo.isin(EQUIP), 'MvarMax'].sum()),
    'equip': g.apply(lambda d: d.tipo.isin(EQUIP).sum()),
    'manobra': g.apply(lambda d: d.tipo.isin(['EL', 'MC', 'IB']).sum()),
    'n_se': g.apply(lambda d: d['se'].dropna().nunique()),
    'n_modulos': g.size(),
    'n_agentes': g['IdAgente'].nunique(),
    'km_amz': g.apply(lambda d: d.loc[(d.tipo == 'LT') & d.uf.isin(AMZ), 'km'].sum()),
    'n_uf': g.apply(lambda d: d['uf'].dropna().nunique()),
    'kv_max': g['kv'].max(),
}).reset_index()
P['km'] = P['km_ge230'] + P['km_lt230']
# exposição a fogo: densidade de focos (INPE, satélite de referência) da UF, ponderada pelos km de LT
# (ou pelo número de módulos, para DMUs só com subestações)
fo = pd.read_csv(os.path.join(B, 'inpe_focos_uf_2015_2022.csv'))
dens = fo.set_index(['uf', 'ano'])['focos_100km2']
A['wfogo'] = np.where(A['tipo'] == 'LT', A['km'], 0.0)
A['dens'] = [dens.get((u, y), np.nan) for u, y in zip(A['uf'], A['ano'])]
def fogo(d):
    d = d.dropna(subset=['dens'])
    w = d['wfogo'] if d['wfogo'].sum() > 0 else pd.Series(1.0, index=d.index)
    return np.average(d['dens'], weights=w) if w.sum() > 0 else np.nan
P['fogo'] = A.groupby(['dmu', 'ano']).apply(fogo).values
P['frac_amz'] = np.where(P['km'] > 0, P['km_amz'] / P['km'], np.nan)
P['km_por_se'] = P['km'] / P['n_se'].clip(lower=1)

# PMSO consolidado (R$ mil, valores históricos) - colunas da segunda tabela (com condenações trabalhistas)
pm = pd.read_excel(F, 'PMSO Consolidado', header=None, skiprows=2).iloc[:, 10:18]
pm.columns = ['IdAgente', 'conc', 'ano', 'pessoal', 'materiais', 'servicos', 'outros', 'cond_trab']
pm = pm[pd.to_numeric(pm['IdAgente'], errors='coerce').notna()].copy()
for c in ['IdAgente', 'ano']:
    pm[c] = pm[c].astype(int)
for c in ['pessoal', 'materiais', 'servicos', 'outros', 'cond_trab']:
    pm[c] = pd.to_numeric(pm[c], errors='coerce').fillna(0)
ix = pd.read_excel(F, 'IPCA-IGPM', header=None, skiprows=1, names=['mes', 'igpm', 'ipca'])
ix['mes'] = pd.to_datetime(ix['mes'], errors='coerce')
ix = ix.dropna().set_index('mes')
ref = ix.loc['2021-12-01']
anual = ix.groupby(ix.index.year).mean()            # deflator pela média do ano
pm['f_ipca'] = pm['ano'].map(ref['ipca'] / anual['ipca'])
pm['f_igpm'] = pm['ano'].map(ref['igpm'] / anual['igpm'])
pm['pmso'] = (pm['pessoal'] + pm['servicos']) * pm['f_ipca'] + (pm['materiais'] + pm['outros']) * pm['f_igpm']
pm['pmso_nom'] = pm[['pessoal', 'materiais', 'servicos', 'outros']].sum(axis=1)
pm['frac_terceiros'] = pm['servicos'] / pm['pmso_nom'].replace(0, np.nan)
pm['dmu'] = pm['IdAgente'].map(grp).fillna(pm['IdAgente'].map(lambda i: f'AG{i}'))
C = pm.groupby(['dmu', 'ano']).agg(pmso=('pmso', 'sum'), pmso_nom=('pmso_nom', 'sum'), pessoal=('pessoal', 'sum'),
                                   servicos=('servicos', 'sum'), nome=('conc', 'first')).reset_index()
C['frac_terceiros'] = C['servicos'] / C['pmso_nom'].replace(0, np.nan)

# qualidade: horas de interrupção
q = pd.read_excel(F, 'INDISPONILIDADE', header=None, skiprows=1).iloc[:, :6]
q.columns = ['IdAgente', 'emp', 'contrato', 'ano', 'holding', 'h_int']
q = q[pd.to_numeric(q['IdAgente'], errors='coerce').notna()].copy()
q['IdAgente'] = q['IdAgente'].astype(int); q['ano'] = q['ano'].astype(int)
q['h_int'] = pd.to_numeric(q['h_int'], errors='coerce').fillna(0)
q['dmu'] = q['IdAgente'].map(grp).fillna(q['IdAgente'].map(lambda i: f'AG{i}'))
Q = q.groupby(['dmu', 'ano'])['h_int'].sum().rename('h_int').reset_index()

# idade média dos ativos (ponderada pelo AIS), por agente -> DMU
L = pd.read_excel(os.path.join(B, 'extra', 'LISTA-ATIVOS--IDADE-MEDIA-.xlsx'), 'Resumo Geral', header=None, skiprows=3)
L = L[pd.to_numeric(L[0], errors='coerce').notna()].copy()
idade = []
for k, ano in enumerate(range(2017, 2022)):
    d = pd.DataFrame({'IdAgente': L[0].astype(int), 'ais': pd.to_numeric(L[6], errors='coerce'),
                      'idade': pd.to_numeric(L[21 + k], errors='coerce'), 'ano': ano})
    idade.append(d)
I = pd.concat(idade)
I = I[(I['idade'] > 0) & (I['ais'] > 0)]
I['dmu'] = I['IdAgente'].map(grp).fillna(I['IdAgente'].map(lambda i: f'AG{i}'))
I['w'] = I['ais'] * I['idade']
I = I.groupby(['dmu', 'ano']).agg(w=('w', 'sum'), ais=('ais', 'sum')).reset_index()
I['idade'] = I['w'] / I['ais']; I['ais_bi'] = I['ais'] / 1e9

D = P.merge(C, on=['dmu', 'ano'], how='left').merge(Q, on=['dmu', 'ano'], how='left').merge(
    I[['dmu', 'ano', 'idade', 'ais_bi']], on=['dmu', 'ano'], how='left')
D['h_int'] = D['h_int'].fillna(0)
D.to_csv(os.path.join(OUT, 'painel_dmu_ano.csv'), index=False)
print(D.shape, D['dmu'].nunique())
print(D.groupby('ano')[['pmso', 'km', 'mva', 'equip', 'manobra']].sum().round(0))
print('PMSO faltante', D['pmso'].isna().sum(), 'PMSO<=0', (D['pmso'] <= 0).sum(), 'sem km', (D['km'] == 0).sum(),
      'idade faltante', D['idade'].isna().sum())
