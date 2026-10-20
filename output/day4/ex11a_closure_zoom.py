import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'output' / 'day4'
import numpy as np
import pandas as pd
from scipy.ndimage import label
n=json.load(open('MainAnalysis_template.ipynb'))
for c in n['cells']:
 s=''.join(c['source'])
 if s.startswith('def Selections(frame):'):exec(s)
m=Selections(pd.read_pickle('data/MC_EXT_flattened.pkl'))
o=Selections(pd.read_pickle('data/oscillated_data.pkl'))
edges=np.linspace(0,2,21)
target=np.histogram(o.trk_energy_tot,bins=edges,weights=o.weight)[0]
energy=m.trk_energy_tot.to_numpy();ids=np.searchsorted(edges,energy,side='right')-1
valid=(ids>=0)&(ids<20)
ids=ids[valid];w=m.weight.to_numpy()[valid];e=m.true_E.to_numpy()[valid]
base=np.bincount(ids,weights=w,minlength=20)
dms=np.linspace(.0001,50,2501);amps=np.linspace(0,1,1001)
grid=np.empty((len(dms),len(amps)))
for i,dm in enumerate(dms):
 deficit=np.bincount(ids,weights=w*np.sin(1.27*dm*.47/e)**2,minlength=20)
 pred=base[None,:]-amps[:,None]*deficit[None,:]
 good=base>0
 grid[i]=np.sum(((pred[:,good]-target[None,good])/(.15*pred[:,good]))**2,axis=1)
np.savez(OUT / 'ex11a_closure_scan.npz', dms=dms, amps=amps, grid=grid)
best=np.unravel_index(np.argmin(grid),grid.shape)
print('best',dms[best[0]],amps[best[1]],grid[best])
for level in [1,2.30,4,6.18,9,11.83]:
 labs,n=label(grid<=level)
 mask=labs==labs[best]
 rows,cols=np.where(mask)
 print('level',level,'islands',n,'connected ranges',dms[rows.min()],dms[rows.max()],amps[cols.min()],amps[cols.max()], 'boundary',[bool(mask[0].any()),bool(mask[-1].any()),bool(mask[:,0].any()),bool(mask[:,-1].any())])
print('profile at 50',grid[-1].min(),amps[grid[-1].argmin()])

# Plot the broad scan and a zoom around the recovered solution.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, title in zip(axes, ['Broad scan: additional allowed regions', 'Zoom around the local minimum']):
    fill = ax.contourf(amps, dms, grid, levels=np.linspace(0, 12, 49), cmap='viridis', extend='max')
    cs = ax.contour(amps, dms, grid, levels=[2.30, 6.18, 11.83], colors=['white', 'orange', 'red'], linewidths=1.5)
    ax.clabel(cs, fmt={2.30: '1 sigma', 6.18: '2 sigma', 11.83: '3 sigma'}, fontsize=9, inline=False)
    ax.plot(0.556, 11.12, marker='x', color='magenta', ms=9, mew=2, label='Recovered minimum')
    ax.set(xlabel=r'$\sin^2(2\theta)$', ylabel=r'$\Delta m^2$ (eV$^2$)', title=title)
axes[0].set_xlim(0.2, 0.8)
axes[1].set_xlim(0.38, 0.75)
axes[1].set_ylim(8, 15)
axes[1].legend(loc='upper right', fontsize=8)
fig.colorbar(fill, ax=axes, label=r'$\Delta\chi^2$ (minimum approximately zero)')
fig.savefig(OUT / 'ex11a_closure_zoom.png', dpi=180)
plt.close(fig)

# Refine local interval endpoints by profiling the other parameter.
from scipy.optimize import minimize_scalar,brentq

def objective(dm,a):
 deficit=np.bincount(ids,weights=w*np.sin(1.27*dm*.47/e)**2,minlength=20)
 p=base-a*deficit;g=base>0
 return np.sum(((p[g]-target[g])/(.15*p[g]))**2)

def prof_dm(dm):
 return minimize_scalar(lambda a:objective(dm,a),bounds=(.3,.8),method='bounded',options={'xatol':1e-10}).fun

def prof_a(a):
 return minimize_scalar(lambda dm:objective(dm,a),bounds=(9,13.5),method='bounded',options={'xatol':1e-9}).fun
for level in [1,2.3]:
 roots=[brentq(lambda dm:prof_dm(dm)-level,9,11.12),brentq(lambda dm:prof_dm(dm)-level,11.12,13.5),brentq(lambda a:prof_a(a)-level,.4,.556),brentq(lambda a:prof_a(a)-level,.556,.7)]
 print('Local projected bounds for threshold',level,roots)
for dm in [50,100,200,500,1000]:
 print('High dm',dm,prof_dm(dm))
