"""Integrity checks and isolated imports for the D52 model snapshot."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_VERSION = "categoriser-v2-D52"
PRIMARY_FAMILY = "material"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False)+'\n')


def empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Preserve existing outputs; choose a new empty directory: {path}")


def load_runtime(freeze):
    freeze = Path(freeze).resolve()
    snapshot = json.loads((freeze/'snapshot.json').read_text())
    if snapshot.get('model_version') != MODEL_VERSION or snapshot.get('primary_family') != PRIMARY_FAMILY:
        raise ValueError('This runner requires the declared D52 v2 material snapshot')
    for rel, expected in snapshot['source_sha256'].items():
        if sha(freeze/'sources'/rel) != expected:
            raise ValueError(f'Frozen source snapshot changed: {rel}')
        if rel.startswith(('analysis/submission_v2/','analysis/submission_test/')) and sha(ROOT/rel) != expected:
            raise ValueError(f'Frozen runner changed: {rel}; use the saved version or register a new version')
    for dependency, expected in snapshot['dependencies'].items():
        if importlib.metadata.version(dependency) != expected:
            raise ValueError(f'Frozen dependency changed: {dependency}')
    if sha(freeze/'models.joblib') != snapshot['models_sha256']:
        raise ValueError('Frozen model hash mismatch')
    if sha(freeze/'known_sites.csv') != snapshot['known_sites_sha256']:
        raise ValueError('Frozen training table hash mismatch')
    for name, expected in snapshot['evaluation_sha256'].items():
        if sha(freeze/'evaluation'/name) != expected:
            raise ValueError(f'Frozen evaluation evidence changed: {name}')
    package = (freeze/'sources/polaron_qc').resolve()
    for name,module in tuple(sys.modules.items()):
        if name == 'polaron_qc' or name.startswith('polaron_qc.'):
            if not Path(module.__file__).resolve().is_relative_to(package):
                raise RuntimeError('Use a fresh Python process: another polaron_qc version is already imported')
    sys.path.insert(0,str(freeze/'sources'))
    cat = importlib.import_module('polaron_qc.categorise')
    return snapshot,cat
