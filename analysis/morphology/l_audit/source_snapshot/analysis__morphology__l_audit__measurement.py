"""Raw-image replay of L's frozen summaries; original images are never written."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label,regionprops_table
from polaron_qc import features
from analysis.morphology.k_pilot.descriptors import phase_summary

PROPS=('label','area','equivalent_diameter_area','major_axis_length','minor_axis_length','orientation','bbox')

def component_table(mask,floor):
    t=pd.DataFrame(regionprops_table(label(mask),properties=PROPS))
    t=t[t.area >= floor].copy();h,w=mask.shape
    t['touches_edge']=(t['bbox-0'].eq(0)|t['bbox-1'].eq(0)|t['bbox-2'].eq(h)|t['bbox-3'].eq(w))
    t['aspect']=t.major_axis_length/np.maximum(t.minor_axis_length,1)
    t['theta']=(np.pi/2-t.orientation)%np.pi
    return t

def shape(t,floor=50):
    t=t[(~t.touches_edge)&t.area.ge(floor)];oriented=t[t.aspect.ge(1.5)]
    aspect=float(np.average(t.aspect,weights=t.area)) if len(t)>=20 else np.nan
    moment=np.average(np.exp(2j*oriented.theta),weights=oriented.area) if len(oriented)>=20 else np.nan
    return {'bright_aspect_aw':aspect,'bright_alignment_strength':float(abs(moment)),
            'bright_n_interior':len(t),'bright_n_oriented':len(oriented),
            'bright_interior_area_share':float(t.area.sum())} # area in pixels; denominator added by caller

def masks(sm,lo,hi):
    return ndi.binary_opening(sm<lo,iterations=1),ndi.binary_opening(sm>hi,iterations=1)

def summaries(pore,bright):
    pt=component_table(pore,30);bt=component_table(bright,50)
    result={'pore_frac':float(pore.mean()),'bright_frac':float(bright.mean()),
            'pore_max_d':float(pt.equivalent_diameter_area.max()) if len(pt) else np.nan,
            'crack_frac':float(pt.loc[pt.major_axis_length>500,'area'].sum()/pore.size),
            'bright_d50':float(features._area_weighted_d(bt.equivalent_diameter_area.to_numpy(),bt.area.to_numpy(),.5)) if len(bt) else np.nan,
            **shape(bt)}
    result['bright_interior_area_share']/=max(float(bt.area.sum()),1.)
    return result,bt

def adaptive_copy(raw):
    top,bottom=features.bright_bands(raw);trim=features.trimmed(raw)
    sm=features.smooth_bse(trim);th=features.phase_thresholds(sm)
    pore,bright=masks(sm,th['th_lo'],th['th_hi']);values,bt=summaries(pore,bright)
    phase,pairs=phase_summary(bright,pore)
    values.update(phase,th_lo=th['th_lo'],th_hi=th['th_hi'],H=trim.shape[0],W=trim.shape[1],
                  trim_top=top,trim_bottom=bottom,bright_low_contrast=th['bright_low_contrast'],
                  pore_mode_resolved=th['pore_mode_resolved'],bright_sep=th['bright_sep'],
                  bse_p1=float(np.percentile(raw,1)),grey_pore=bool(np.percentile(raw,1)>10))
    return values,trim,pore,bright,bt,pairs
