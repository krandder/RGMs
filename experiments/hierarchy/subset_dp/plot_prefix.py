"""Draw the exact depth posterior on increasing prefixes of the same trajectory."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
here=Path(__file__).resolve().parent
rows=json.loads((here/'data_sensitivity.json').read_text())
steps=[0,128,512,2048,8192]
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'treefit','svg.fonttype':'none'}):
    fig,ax=plt.subplots(figsize=(6.4,3.45),layout='constrained')
    for seed,(color,mark) in enumerate([('#9a5222','o'),('#195b86','s'),('#39765c','^')]):
        probs=[next(r['depth_probs'][1] for r in rows if r['seed']==seed and r['n']==n) for n in steps]
        ax.plot(range(5),probs,color=color,marker=mark,lw=1.9,label='seed '+str(seed))
    ax.set(yscale='log',ylim=(1e-17,2.4),xlabel='Observed transitions from the same world',ylabel='Posterior mass on depth two')
    ax.set_xticks(range(5),['0 (prior)','128','512','2,048','8,192'])
    ax.yaxis.set_major_locator(FixedLocator([1e-16,1e-12,1e-8,1e-4,1]))
    ax.grid(axis='y',color='#d5dadd'); ax.legend(loc='lower left',ncol=3,frameon=False)
    ax.set_title('Depth-two generators, evaluated by exact inference',loc='left',weight='semibold',pad=10)
    fig.savefig(here/'depth-prefix.svg',metadata={'Date':None})
    svg=here/'depth-prefix.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
