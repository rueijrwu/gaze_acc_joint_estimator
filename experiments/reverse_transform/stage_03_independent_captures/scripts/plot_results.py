#!/usr/bin/env python3
"""Plot saved independent-capture results; performs no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();root=args.results.resolve()
    summary=json.loads((root/'summary.json').read_text())
    post=json.loads((root/'postfit.json').read_text()) if (root/'postfit.json').exists() else None
    with np.load(root/'frames.npz',allow_pickle=False) as z: f={k:z[k] for k in z.files}
    with np.load(root/'population.npz',allow_pickle=False) as z: pop={k:z[k] for k in z.files}
    captures=[int(x['capture']) for x in summary['captures']]
    common=f['common_inverse_valid'];cap=f['capture_index'];exp=f['exposure']
    ref=f['reference_p4'];rref=float(np.sqrt(np.mean(np.sum(ref*ref,axis=1))))
    nominal=np.array([-10.,-5.,0.,5.,10.])
    models=[('P1 M only','before_p4','#4d4d4d','--'),
            ('Keystone','keystone_recovered_p4','#1b9e77','-.')]
    if len(captures)>1:
        models.append(('Keystone + barrel','recovered_p4','#d95f02','-'))

    def shape_normalize(x):
        centered=x-x.mean(axis=1,keepdims=True)
        radius=np.sqrt(np.mean(np.sum(centered*centered,axis=2),axis=1))
        good=np.isfinite(radius)&(radius>0)
        normalized=np.full_like(centered,np.nan)
        normalized[good]=centered[good]*(rref/radius[good])[:,None,None]
        return normalized

    def point_error(x,ids,shape=False):
        points=x[ids]
        if shape: points=shape_normalize(points)
        return np.linalg.norm(points-ref[None,:,:],axis=2).reshape(-1)

    def medp95(values):
        values=values[np.isfinite(values)]
        return (float(np.median(values)),float(np.percentile(values,95))) if len(values) else (np.nan,np.nan)

    # Error panels: every model uses the common inverse-valid rows and the
    # plotted count shows the numerator and complete-frame denominator.
    for shape in (False,True):
        fig,axs=plt.subplots(len(captures),2,figsize=(12,3.1*len(captures)),squeeze=False,sharex=True)
        for ri,c in enumerate(captures):
            panel=cap==c-1; complete=int(panel.sum())
            for col,stat in enumerate(('median','p95')):
                ax=axs[ri,col]
                for label,key,color,style in models:
                    ys=[];counts=[]
                    for k in range(5):
                        ids=panel&(exp==((c-1)*5+k))&common
                        values=point_error(f[key],ids,shape)
                        q=medp95(values);ys.append(q[0] if stat=='median' else q[1]);counts.append(int(ids.sum()))
                    ax.plot(nominal,ys,marker='o',lw=1.5,ms=4,color=color,ls=style,label=label)
                ticks=[]
                for k in range(5):
                    n=int(np.sum(panel&(exp==(c-1)*5+k)))
                    v=int(np.sum(panel&(exp==(c-1)*5+k)&common))
                    ticks.append(f'{nominal[k]:g}°\n{v}/{n}')
                ax.set_xticks(nominal,ticks);ax.grid(alpha=.25)
                ax.set_ylabel('Point distance (px)')
                ax.set_title(f'Capture {c}: {stat.upper()}; common valid {int(np.sum(panel&common))}/{complete}')
        axs[-1,0].set_xlabel('Nominal gaze; common-valid / complete frames')
        axs[-1,1].set_xlabel('Nominal gaze; common-valid / complete frames')
        hs,ls=axs[0,0].get_legend_handles_labels()
        fig.legend(hs,ls,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.005))
        fig.suptitle('P4 '+('radius-normalized shape' if shape else 'inverse')+' point error by fixation',y=1.025)
        fig.tight_layout()
        fig.savefig(root/('shape_error_by_fixation.png' if shape else 'point_error_by_fixation.png'),dpi=180,bbox_inches='tight')
        plt.close(fig)

    # Triangle overlays: reference and fixation mean triangles, one correction
    # per column. Means use the common inverse-valid frames only.
    fig,axs=plt.subplots(len(captures),len(models),figsize=(5.0*len(models),4.0*len(captures)),squeeze=False)
    fixation_colors=plt.cm.coolwarm(np.linspace(0,1,5))
    for ri,c in enumerate(captures):
        panel=cap==c-1
        for mi,(label,key,_,_) in enumerate(models):
            ax=axs[ri,mi]
            loop=np.r_[0:3,0]
            ax.plot(ref[loop,0],ref[loop,1],color='black',lw=2.0,label='Reference')
            for k in range(5):
                ids=panel&(exp==((c-1)*5+k))&common
                if not np.any(ids): continue
                mean=f[key][ids].mean(axis=0)
                ax.plot(mean[loop,0],mean[loop,1],color=fixation_colors[k],lw=1.25,marker='o',ms=2.5)
            ax.set_xlim(-220,220);ax.set_ylim(-220,220);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
            ax.set_title(f'Capture {c}: {label}')
            if ri==len(captures)-1: ax.set_xlabel('x (px)')
            if mi==0: ax.set_ylabel('y (px)')
    fixation_handles=[Line2D([0],[0],color=fixation_colors[k],lw=2,label=f'{nominal[k]:+g}° mean') for k in range(5)]
    fixation_handles.insert(0,Line2D([0],[0],color='black',lw=2,label='Reference'))
    fig.legend(handles=fixation_handles,loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(.5,1.005))
    fig.suptitle('Reference and fixation-mean inverse triangles; equal aspect',y=1.025)
    fig.tight_layout();fig.savefig(root/'triangle_overlays.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Full-model vertex clouds after centering and normalizing each frame to
    # the reference RMS radius. Tails outside the view remain counted.
    full_shape=shape_normalize(f['recovered_p4'])
    fig,axs=plt.subplots(len(captures),3,figsize=(13,3.3*len(captures)),squeeze=False,sharex=True,sharey=True)
    view=5.5
    for ri,c in enumerate(captures):
        panel=cap==c-1
        for j,ax in enumerate(axs[ri]):
            total=outside=0
            for k in range(5):
                ids=np.flatnonzero(panel&(exp==((c-1)*5+k))&common)
                if not len(ids): continue
                all_delta=full_shape[ids,j]-ref[j]
                total+=len(ids);outside+=int(np.sum(np.any(np.abs(all_delta)>view,axis=1)))
                take=ids[np.linspace(0,len(ids)-1,min(150,len(ids)),dtype=int)]
                delta=full_shape[take,j]-ref[j]
                ax.scatter(delta[:,0],delta[:,1],s=5,color=fixation_colors[k],alpha=.16,linewidths=0,rasterized=True)
                avg=all_delta.mean(axis=0)
                ax.scatter([avg[0]],[avg[1]],s=28,color=fixation_colors[k],edgecolor='black',linewidth=.35,zorder=3)
            ax.scatter([0],[0],marker='D',s=35,color='black',zorder=5)
            ax.axhline(0,color='.75',lw=.5);ax.axvline(0,color='.75',lw=.5)
            ax.set_xlim(-view,view);ax.set_ylim(-view,view);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
            ax.set_title(f'Capture {c}, vertex {j+1}')
            ax.text(.03,.04,f'Common n={total}\nOutside view: {outside}',transform=ax.transAxes,fontsize=8,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.8},va='bottom')
            if ri==len(captures)-1: ax.set_xlabel('Δx from reference (px)')
            if j==0: ax.set_ylabel('Δy from reference (px)')
    fix_handles=[Line2D([0],[0],marker='o',linestyle='none',color=fixation_colors[k],label=f'{nominal[k]:+g}°') for k in range(5)]
    fig.legend(handles=fix_handles+[Line2D([0],[0],marker='D',linestyle='none',color='black',label='Reference vertex')],
               loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(.5,1.005))
    fig.suptitle(('Keystone + barrel' if len(models)>2 else 'Keystone')+
                 ' inverse, radius-normalized vertex errors; cloud subsample 150/fixation',y=1.025)
    fig.tight_layout();fig.savefig(root/'normalized_vertex_clouds.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Framewise radius ratio distributions, with only common-valid rows.
    fig,axs=plt.subplots(1,len(captures),figsize=(4.2*len(captures),5),squeeze=False,sharey=True)
    for ri,c in enumerate(captures):
        ax=axs[0,ri];panel=cap==c-1;groups=[];positions=[];labels=[]
        for k in range(5):
            ids=panel&(exp==((c-1)*5+k))&common
            stride=len(models)+1
            for mi,(name,key,color,_) in enumerate(models):
                x=f[key][ids];center=x-x.mean(axis=1,keepdims=True)
                rad=np.sqrt(np.mean(np.sum(center*center,axis=2),axis=1))/rref
                groups.append(rad[np.isfinite(rad)]);positions.append(k*stride+mi+1)
                if mi==0: labels.append(f'{nominal[k]:g}°\n{int(ids.sum())}')
        bp=ax.boxplot(groups,positions=positions,widths=.65,showfliers=False,patch_artist=True,
                      medianprops={'color':'black','linewidth':1})
        for box,i in zip(bp['boxes'],range(len(groups))): box.set_facecolor(models[i%len(models)][2]);box.set_alpha(.55)
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_xticks([k*stride+(len(models)+1)/2 for k in range(5)],labels)
        ax.grid(axis='y',alpha=.25);ax.set_title(f'Capture {c}; common-valid only')
        if ri==0: ax.set_ylabel('Recovered RMS radius / reference')
        ax.set_xlabel('Nominal gaze; common frame count')
    handles=[Line2D([0],[0],color=color,lw=7,alpha=.55,label=label) for label,_,color,_ in models]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.04))
    fig.suptitle('Framewise recovered radius distributions by fixation',y=1.10)
    fig.tight_layout();fig.savefig(root/'radius_distributions.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Independent points and the anchored descriptive line only.
    if post is not None:
        coeff=summary['captures'];d=np.array([r['demand_diopters_label'] for r in coeff]);kappa=np.array([r['relative_barrel_per_px2'] for r in coeff])*1e6
        order=np.argsort(d)
        fig,ax=plt.subplots(figsize=(8.5,5))
        ax.scatter(d[order],kappa[order],color='#1b9e77',s=55,label='Independent capture coefficients',zorder=3)
        for r in coeff:
            ax.annotate(f"Capture {r['capture']}, {r['demand_diopters_label']:g} D",
                        (r['demand_diopters_label'],r['relative_barrel_per_px2']*1e6),
                        xytext=(5,5),textcoords='offset points',fontsize=8)
        dref=float(coeff[0]['demand_diopters_label']);xx=np.linspace(d.min(),d.max(),100)
        ax.plot(xx,post['anchored_linear_slope_per_D_px2']*1e6*(xx-dref),color='black',ls='--',lw=1.4,
                label='Post-fit anchored linear association')
        ax.axhline(0,color='.5',lw=.7);ax.grid(alpha=.25)
        ax.set_xlabel('Nominal demand label (D)');ax.set_ylabel('Relative barrel increment κ (10⁻⁶ px⁻²)')
        ax.set_title('Independent capture coefficients; descriptive demand association')
        ax.text(.02,.02,'Capture 1 κ=0 is the empirical reference gauge. One capture per demand.',transform=ax.transAxes,fontsize=8)
        ax.legend(frameon=False,fontsize=8,loc='center left',bbox_to_anchor=(1.02,.5))
        fig.tight_layout(rect=(0,0,.76,1));fig.savefig(root/'postfit_barrel_vs_demand.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__': main()
