"""Fit and preserve the already declared D52 specification on known sites only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys

import joblib

from .runtime import HERE, ROOT, MODEL_VERSION, PRIMARY_FAMILY, empty_output, sha, write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'freeze')
    args=parser.parse_args(argv)
    empty_output(args.out)
    from polaron_qc import categorise as cat
    assert cat.PRIMARY_FAMILY == PRIMARY_FAMILY
    assert len(cat.FAMILIES[PRIMARY_FAMILY]) == 29
    assert not set(cat.FAMILIES[PRIMARY_FAMILY]) & set(cat.ACQ_FEATURES)
    assert all('H' not in features for features in cat.FAMILIES.values())
    started=datetime.now(timezone.utc).isoformat()
    source_paths=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'polaron_qc').glob('*.py'))
    source_paths += ['analysis/submission_test/compose_submission.py']
    source_paths += sorted(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
    source_paths += ['analysis/submission_v2/protocol.md','analysis/morphology/metric_register.json']
    hashes={rel:sha(ROOT/rel) for rel in source_paths}
    # Raw known-data hashes are provenance. No incoming images enter this function.
    raw=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
         for p in sorted((ROOT/'Dataset').glob('Batch_*/*.tif'))]
    args.out.mkdir(parents=True,exist_ok=True)
    for rel in source_paths:
        target=args.out/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    print('Freezing D52 v2: known sites only; full fold-local evaluation and saved fits...',flush=True)
    result=cat.run([str(ROOT/'Dataset'/b) for b in ['Batch_3','Batch_1','Batch_2']],
                   out_dir=str(args.out/'evaluation'),n_perm=200,seed=0,n_jobs=-1,verbose=False)
    known=result['known'];known.to_csv(args.out/'known_sites.csv',index=False)
    primary=result['results'][PRIMARY_FAMILY]
    joblib.dump(dict(models=result['models'],ood_models=result['ood_models'],known=known,
                     primary_deletion_models=primary['fold_models'],deleted_sites=primary['sites'].tolist()),
                args.out/'models.joblib',compress=3)
    for rel,expected in hashes.items():
        if sha(ROOT/rel)!=expected:raise ValueError(f'Source changed while freezing: {rel}')
    for row in raw:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Known raw data changed while freezing')
    eval_names=['categoriser_summary.csv',f'reliability_{PRIMARY_FAMILY}.csv']
    metadata=dict(model_version=MODEL_VERSION,experiment='E35S',decision='D54S',primary_family=PRIMARY_FAMILY,
                  ref_batch='Batch_3',started_utc=started,captured_utc=datetime.now(timezone.utc).isoformat(),
                  head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_dirty=True,
                  provenance='D52 specification revised after first-drop inspection, before its truth; declared before final evaluation. This snapshot fits known data only.',
                  no_incoming_data_used_for_fit_or_evaluation=True,source_sha256=hashes,raw_training_manifest=raw,
                  training_site_counts=known.batch.value_counts().to_dict(),families=cat.FAMILIES,
                  primary_features=result['models'][PRIMARY_FAMILY].features,chosen_C={f:m.C for f,m in result['models'].items()},
                  seed=0,C_grid=list(cat.C_GRID),n_inner=cat.N_INNER,n_perm=200,
                  primary_deletion_fit_count=len(primary['fold_models']),models_sha256=sha(args.out/'models.joblib'),
                  known_sites_sha256=sha(args.out/'known_sites.csv'),
                  evaluation_sha256={name:sha(args.out/'evaluation'/name) for name in eval_names},
                  dependencies={d:importlib.metadata.version(d) for d in ['numpy','pandas','scipy','scikit-learn','scikit-image','tifffile','joblib']},
                  python=sys.version.split()[0],scores_calibrated=False,texture_origin='unresolved; acquisition-sensitive appearance')
    write_json(args.out/'snapshot.json',metadata)
    print(f'Saved {MODEL_VERSION}; primary={PRIMARY_FAMILY}, features=29, known counts={metadata["training_site_counts"]}',flush=True)


if __name__=='__main__':main()
