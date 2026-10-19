import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultados'); F=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figuras')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
E=pd.read_csv(f'{R}/eficiencias_dmu.csv'); D=pd.read_csv(f'{R}/eficiencias_dmu_ano_todas.csv')
Em=D.groupby('dmu')[['ef_SFA','ef_ML']].mean(); E=E.join(Em,on='dmu')
lab={'AG394':'Furnas','AG48':'Chesf','AG372':'Eletronorte','ISA_HOL':'ISA CTEEP group','AG97':'CGT Eletrosul','TAESA_HOL':'TAESA group','AG5785':'CEEE-T','STATE GRID_HOL':'State Grid group','COPEL_HOL':'Copel group','AG4951':'Cemig GT','EVOLTZ_HOL':'Evoltz group','TBE_HOL':'TBE group','CELEO_HOL':'Celeo group','AG10307':'XRTE','AG8965':'BMTE'}
top=E.sort_values('pmso_2021',ascending=False).head(15).iloc[::-1]
# Fig 1: painel (a) escores de fronteira (<= 1); painel (b) razão de prática média do ML (pode exceder 1)
fig,(ax,ax2)=plt.subplots(1,2,figsize=(7.0,5.0),gridspec_kw={'width_ratios':[3,1.3]},sharey=True)
sp={'CRS':('o','#1b9e77'),'VRS':('s','#d95f02'),'NDRS_AR':('D','#7570b3'),'ef_SFA':('^','#e7298a')}
nm={'CRS':'DEA CRS','VRS':'DEA VRS','NDRS_AR':'DEA NDRS + weight restrictions','ef_SFA':'SFA'}
y=np.arange(len(top))
for k,(m,c) in sp.items(): ax.scatter(top[k],y,marker=m,color=c,s=22,label=nm[k],zorder=3)
for i,(_,r) in enumerate(top.iterrows()):
    vals=[r[k] for k in sp]; ax.plot([min(vals),max(vals)],[i,i],color='0.8',lw=1,zorder=1)
ax.set_yticks(y); ax.set_yticklabels([lab.get(d,d) for d in top.dmu]); ax.set_xlabel('(a) Mean efficiency score, 2017–2021'); ax.set_xlim(0,1.05); ax.axvline(1,color='0.5',ls=':',lw=.8)
ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(0.5,1.12),ncol=2,frameon=False); ax.grid(axis='x',color='0.92')
ax2.scatter(top['ef_ML'],y,marker='v',color='#666666',s=22,zorder=3); ax2.axvline(1,color='0.5',ls=':',lw=.8)
ax2.set_xlabel('(b) ML cost norm:\npredicted / observed cost'); ax2.set_xlim(0,max(1.1,top['ef_ML'].max()*1.08)); ax2.grid(axis='x',color='0.92')
plt.tight_layout(); plt.savefig(f'{F}/Fig1_scores_specs.png',dpi=300); plt.savefig(f'{F}/Fig1_scores_specs.tif',dpi=300); plt.close()
# Fig 2 recognised cost
rc=pd.read_csv(f'{R}/custo_reconhecido.csv',index_col=0)
rt=pd.read_csv(f'{R}/rts_test_SW2002.csv',index_col=0).iloc[:,0]; rc.loc['CRS_bc','var_%']=float(rt['var_pct_CRS_bc'])
order=['VRS','VRS_bc','NDRS','CRS','CRS_bc','NDRS_AR','CRS_AR']; names=['DEA VRS','DEA VRS\n(bias-corr.)','DEA NDRS','DEA CRS','DEA CRS\n(bias-corr.)','DEA NDRS\n+ WR','DEA CRS\n+ WR']
fig,ax=plt.subplots(figsize=(6.3,3.2))
v=rc.loc[order,'var_%']; ax.bar(range(len(v)),v,color=['#d95f02','#d95f02','#7570b3','#1b9e77','#1b9e77','#7570b3','#1b9e77'])
for i,x in enumerate(v): ax.text(i,x+(1.5 if x>0 else -4),f'{x:+.0f}%',ha='center',fontsize=7)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(range(len(v))); ax.set_xticklabels(names,fontsize=7)
ax.set_ylabel('Recognised O&M cost vs actual 2021 (%)'); ax.set_ylim(-70,15)
plt.tight_layout(); plt.savefig(f'{F}/Fig2_recognised_cost.png',dpi=300); plt.savefig(f'{F}/Fig2_recognised_cost.tif',dpi=300); plt.close()
# Fig 3 bootstrap CI 2021
d=D[D.ano==2021].sort_values('ef_VRS_bc').reset_index(drop=True)
fig,ax=plt.subplots(figsize=(6.5,3.2))
ax.errorbar(range(len(d)),d.ef_VRS_bc,yerr=[d.ef_VRS_bc-d.ef_VRS_lo,d.ef_VRS_hi-d.ef_VRS_bc],fmt='o',ms=3,color='#d95f02',ecolor='0.6',lw=.8,label='bias-corrected, 95% CI')
ax.scatter(range(len(d)),d.ef_VRS,s=8,color='k',marker='x',label='conventional DEA VRS',zorder=3)
ax.set_xlabel('Operator (2021, ranked)'); ax.set_ylabel('Efficiency'); ax.legend(frameon=False,fontsize=7); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig(f'{F}/Fig3_bootstrap.png',dpi=300); plt.savefig(f'{F}/Fig3_bootstrap.tif',dpi=300); plt.close()
print('ok', len(d))
