"""Fixed shape distributions and finite-frame binary phase correlations."""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = (64, 256)
MODES = ('nominal', 'threshold_minus5', 'threshold_plus5', 'floor100')
SETTINGS = {'nominal': (0,50), 'threshold_minus5': (-5,50),
            'threshold_plus5': (5,50), 'floor100': (0,100)}
SHAPE_KEYS = ('bright_aspect_count_iqr','bright_solidity_count_q10','bright_circularity_count_iqr')
AUTO_KEYS = tuple(f'bright_autocorr_xy_contrast_{lag}px' for lag in LAGS)
CROSS_KEYS = tuple(f'bright_pore_crosscorr_xy_contrast_{lag}px' for lag in LAGS)
KEYS = SHAPE_KEYS + AUTO_KEYS + CROSS_KEYS
DEFINITIONS = {
 SHAPE_KEYS[0]: ('ratio','Linear Q75−Q25 of major/minor axis ratio, equal weight per eligible bright component.'),
 SHAPE_KEYS[1]: ('0–1','Linear Q10 of raster solidity, equal weight per eligible bright component.'),
 SHAPE_KEYS[2]: ('dimensionless','Linear Q75−Q25 of 4π area/perimeter², equal weight per eligible bright component; raster estimates are not clipped to one.'),
 **{k: ('−2–2',f'Binary bright-mask Pearson correlation at x lag {lag}px minus y lag {lag}px, using pair-specific marginals in the finite overlapping domains.') for k,lag in zip(AUTO_KEYS,LAGS)},
 **{k: ('−2–2',f'Symmetric ±{lag}px bright-to-void binary Pearson correlation along x minus along y. Each signed direction uses its own finite overlapping domain and marginals.') for k,lag in zip(CROSS_KEYS,LAGS)},
}

def shape_summary(nodes, min_objects=20):
    required={'area','major_axis_length','minor_axis_length','solidity','perimeter','touches_edge'}
    if not required<=set(nodes):raise ValueError('Missing component geometry')
    if not nodes.touches_edge.isin([True,False]).all():raise ValueError('Unknown clipping state')
    numeric=nodes[['area','major_axis_length','minor_axis_length','solidity','perimeter']].to_numpy(float)
    if not np.isfinite(numeric).all():raise ValueError('Non-finite component geometry')
    eligible=(~nodes.touches_edge)&(nodes.minor_axis_length>0)&(nodes.perimeter>0)&(nodes.area>0)
    t=nodes.loc[eligible].copy()
    t['shape_aspect']=t.major_axis_length/t.minor_axis_length
    t['shape_circularity']=4*np.pi*t.area/t.perimeter**2
    out={k:np.nan for k in SHAPE_KEYS}
    reason='fewer_than_20_eligible_components' if len(t)<min_objects else ''
    if not reason:
        out[SHAPE_KEYS[0]]=float(np.diff(np.quantile(t.shape_aspect,[.25,.75],method='linear'))[0])
        out[SHAPE_KEYS[1]]=float(np.quantile(t.solidity,.1,method='linear'))
        out[SHAPE_KEYS[2]]=float(np.diff(np.quantile(t.shape_circularity,[.25,.75],method='linear'))[0])
    out.update(shape_objects_input=len(nodes),shape_objects_eligible=len(t),
               shape_objects_excluded=len(nodes)-len(t),shape_unavailable_reason=reason,
               shape_circularity_above_one=int((t.shape_circularity>1).sum()))
    return out,t

def shifted_pairs(a,b,lag,axis):
    a,b=np.asarray(a),np.asarray(b)
    if a.ndim!=2 or a.shape!=b.shape or a.dtype!=bool or b.dtype!=bool:raise ValueError('Aligned 2-D binary masks required')
    if axis not in (0,1) or not isinstance(lag,(int,np.integer)) or lag==0:raise ValueError('Nonzero integer lag and x/y axis required')
    if abs(lag)>=a.shape[axis]:return np.empty(0,bool),np.empty(0,bool)
    start,end=[slice(None)]*2,[slice(None)]*2
    if lag>0:start[axis]=slice(None,-lag);end[axis]=slice(lag,None)
    else:start[axis]=slice(-lag,None);end[axis]=slice(None,lag)
    return a[tuple(start)],b[tuple(end)]

def pair_counts(a,b,lag,axis,min_pairs=10000,min_phase_pixels=100):
    x,y=shifted_pairs(a,b,lag,axis)
    n=x.size;nx=int(x.sum());ny=int(y.sum());joint=int(np.count_nonzero(x&y))
    reason=''
    if n<min_pairs:reason='insufficient_pair_domain'
    elif min(nx,ny,n-nx,n-ny)<min_phase_pixels:reason='insufficient_phase_or_complement_pixels'
    rho=np.nan
    if not reason:
        px,py,p11=nx/n,ny/n,joint/n
        rho=(p11-px*py)/np.sqrt(px*(1-px)*py*(1-py))
    return dict(lag_px=lag,axis='x' if axis==1 else 'y',pair_pixels=n,
                source_phase_pixels=nx,target_phase_pixels=ny,joint_phase_pixels=joint,
                rho=float(rho),unavailable_reason=reason)

def phase_summary(bright,pore):
    rows=[];out={k:np.nan for k in AUTO_KEYS+CROSS_KEYS}
    for lag in LAGS:
        for kind,target in (('auto',bright),('cross',pore)):
            vals={}
            for axis in (1,0):
                signed=(lag,) if kind=='auto' else (lag,-lag)
                current=[pair_counts(bright,target,s,axis) for s in signed]
                vals[axis]=np.mean([r['rho'] for r in current])
                rows.extend(dict(kind=kind,**r) for r in current)
            key=f'bright_{"autocorr" if kind=="auto" else "pore_crosscorr"}_xy_contrast_{lag}px'
            out[key]=float(vals[1]-vals[0])
            out[key+'_unavailable_reason']='' if np.isfinite(out[key]) else 'insufficient_pair_or_phase_coverage'
    return out,pd.DataFrame(rows)

def usable(table,key,view='phase_usable'):
    if view not in ('phase_usable','quality_matched','ordinary_reference'):raise ValueError('Unknown quality view')
    ok=np.isfinite(pd.to_numeric(table[key],errors='coerce'))
    ok &= table.bright_low_contrast.eq(False).fillna(False) if 'bright_low_contrast' in table else False
    if key in CROSS_KEYS or view!='phase_usable':
        ok &= table.grey_pore.eq(False).fillna(False) if 'grey_pore' in table else False
    if view=='ordinary_reference':
        ok &= ~table.batch.eq('Batch_3') | (table.cracked_known.eq(False).fillna(False) if 'cracked_known' in table else False)
    return np.asarray(ok,bool)
