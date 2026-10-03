"""E31L.1 statistical audit helpers; no production verdict or learned classifier.

Permutation reference fitting is allocation-local. Unknown quality and nonfinite
inputs abstain; only acquisition/redundancy prediction uses fold-local imputation.
"""
from itertools import combinations
from math import comb
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from polaron_qc.stats import _energy_stat

BASE = ['crack_frac', 'pore_max_d', 'pore_frac', 'bright_frac', 'bright_d50']
CAND = ['bright_aspect_aw', 'bright_alignment_strength', 'bright_pore_crosscorr_xy_contrast_256px']
KEYS = BASE + CAND
PANELS = {'primary5': list(range(5)), 'morphology3': list(range(5, 8)), 'extended8': list(range(8)),
          **{'primary5_plus_' + k: list(range(5)) + [i+5] for i,k in enumerate(CAND)}}
SEED = 20261004
MIN_SITES = 5

def canonical(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite two-dimensional input required; do not impute QC measurements')
    return x[np.lexsort(x.T[::-1])]

def scale_fit(ref):
    med = np.median(ref, axis=0)
    mad = 1.4826 * np.median(np.abs(ref-med), axis=0)
    sd = np.std(ref, axis=0)
    fallback = np.where(mad > 0, 'mad', np.where(sd > 0, 'sd', 'unit'))
    return med, np.where(mad > 0, mad, np.where(sd > 0, sd, 1.)), fallback

def eligible(table, keys, shared=False):
    ok = np.isfinite(table[keys].to_numpy(float)).all(axis=1)
    ok &= table.bright_low_contrast.eq(False).fillna(False).to_numpy(bool)
    if shared or CAND[2] in keys:
        ok &= table.grey_pore.eq(False).fillna(False).to_numpy(bool)
    if CAND[0] in keys:
        ok &= table.bright_n_interior.ge(20).to_numpy(bool)
    if CAND[1] in keys:
        ok &= table.bright_n_oriented.ge(20).to_numpy(bool)
    return ok

def allocation_orders(n, nr, exact_limit=20000, n_mc=4999, seed=SEED):
    nb = n-nr
    if min(nr,nb) < 1: raise ValueError('Two nonempty samples required')
    total = comb(n,nb)
    if total <= exact_limit:
        query = np.array(list(combinations(range(n), nb)), int)
        complement = np.ones((len(query), n), bool)
        complement[np.arange(len(query))[:,None],query] = False
        ref = np.broadcast_to(np.arange(n), complement.shape)[complement].reshape(-1,nr)
        return np.concatenate([ref,query],axis=1), 'exact', total
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_mc,n)),axis=1), 'monte_carlo', total

def statistics(pooled, nr, orders, panels=PANELS, chunk=128):
    """Same allocation stream for every panel; reference fit repeated per panel."""
    result = {p:np.empty(len(orders)) for p in panels}
    for start in range(0,len(orders),chunk):
        ix = slice(start,start+chunk)
        stack = pooled[orders[ix]]
        for name,cols in panels.items():
            result[name][ix] = _energy_stat(stack[:,:,cols], nr, np.full(len(cols),1/np.sqrt(len(cols))))
    return result

def tail_p(null, observed, method):
    # Numerical equality is a tie, not a spurious reject.
    hits = np.count_nonzero(null >= observed - 1e-12 * max(1.,abs(observed)))
    return hits/len(null) if method=='exact' else (hits+1)/(len(null)+1)

def holm(p):
    p=np.asarray(p,float)
    if not np.isfinite(p).all():return np.full_like(p,np.nan)
    order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def policies(p):
    adjusted=holm([p['primary5'],p['morphology3']])
    return {'primary5':p['primary5'] <= .05,'extended8':p['extended8'] <= .05,
            'holm_union':bool((adjusted <= .05).any()),
            'naive_or':min(p['primary5'],p['morphology3']) <= .05}

def energy_tests(ref,query,n_mc=4999,exact_limit=20000,seed=SEED,panels=PANELS):
    ref,query=canonical(ref),canonical(query)
    if min(len(ref),len(query)) < 2:raise ValueError('Numerical sensitivity requires at least two sites per side')
    pooled=np.vstack([ref,query]);nr=len(ref)
    orders,method,total=allocation_orders(len(pooled),nr,exact_limit,n_mc,seed)
    null=statistics(pooled,nr,orders,panels)
    observed=statistics(pooled,nr,np.arange(len(pooled))[None],panels)
    p={name:tail_p(null[name],observed[name][0],method) for name in panels}
    valid=min(len(ref),len(query))>=MIN_SITES
    rows=[]
    for name,cols in panels.items():
        med,scale,fallback=scale_fit(ref[:,cols])
        rows.append(dict(panel=name,statistic=float(observed[name][0]),p=p[name],n_ref=len(ref),
                         n_batch=len(query),n_allocations=len(orders),total_allocations=total,method=method,
                         alert_available=valid,alert=(p[name] <= .05) if valid else None,
                         fallback_sd=int((fallback=='sd').sum()),fallback_unit=int((fallback=='unit').sum())))
    return rows,p,null

def exact_ranks(values):
    # Rank equal statistics together, including self (upper tail).
    a=np.asarray(values,float)
    return (len(a)+1-rankdata(np.round(a,12),method='min'))/len(a)

def wilson(hits,n):
    if n<=0:return np.nan,np.nan
    z=1.959963984540054;p=hits/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

def paired_interval(values,n_boot=4000,seed=SEED):
    v=np.asarray(values,float);rng=np.random.default_rng(seed)
    means=v[rng.integers(0,len(v),(n_boot,len(v)))].mean(axis=1)
    return float(v.mean()),*np.quantile(means,[.025,.975]).tolist()

def median_interval(ref,query,n_boot=4000,seed=SEED):
    rng=np.random.default_rng(seed)
    r=np.asarray(ref,float);b=np.asarray(query,float)
    boot=np.median(b[rng.integers(0,len(b),(n_boot,len(b)))],axis=1)-np.median(r[rng.integers(0,len(r),(n_boot,len(r)))],axis=1)
    return float(np.median(b)-np.median(r)),*np.quantile(boot,[.025,.975]).tolist()

def neighbour_score(ref,query):
    if len(ref)<3:raise ValueError('Three reference neighbours required')
    med,scale,fallback=scale_fit(ref)
    r=(ref-med)/scale/np.sqrt(ref.shape[1]);q=(query-med)/scale/np.sqrt(ref.shape[1])
    d=cdist(q,r);near=np.argmin(d,axis=1)
    contributions=(q-r[near])**2
    return np.sort(d,axis=1)[:,:3].mean(axis=1),near,contributions,fallback

def gaussian_draw(rng,n):
    common=rng.normal(size=(n,1))*.5
    block=rng.normal(size=(n,2))*.5
    return common+block[:,[0]*5+[1]*3]+rng.normal(size=(n,8))*np.sqrt(.5)
