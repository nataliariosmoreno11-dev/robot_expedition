import shutil

import numpy as np
import pytest

from api.store import Store
from pipelines.policies import ModelIntegrityError, ModelUnavailable


def test_validated_adapter_serves_without_original_zip_and_rejects_tampering(tmp_path, monkeypatch):
    run_id = '92cdc692fdca465b917fa7c5e8a9902d'
    for folder in ('manifests', 'catalog/adapters', 'docs/evidencia'):
        (tmp_path / folder).mkdir(parents=True)
    for name in (f'manifests/{run_id}.json', f'catalog/adapters/{run_id}.npz',
                 'docs/evidencia/adaptador-numpy.json'):
        shutil.copy2(name, tmp_path / name)
    monkeypatch.setenv('POLICY_BACKEND', 'numpy')
    store = Store(root=tmp_path)
    assert 0 <= store.policy(run_id)(np.zeros(20)) < 5
    assert not (tmp_path / 'models' / run_id / 'ppo_model.zip').exists()
    path = tmp_path / f'catalog/adapters/{run_id}.npz'
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(ModelIntegrityError):
        Store(root=tmp_path).policy(run_id)


def test_adapter_requires_export_evidence(tmp_path, monkeypatch):
    run_id = '92cdc692fdca465b917fa7c5e8a9902d'
    for folder in ('manifests', 'catalog/adapters'):
        (tmp_path / folder).mkdir(parents=True)
    for name in (f'manifests/{run_id}.json', f'catalog/adapters/{run_id}.npz'):
        shutil.copy2(name, tmp_path / name)
    monkeypatch.setenv('POLICY_BACKEND', 'numpy')
    with pytest.raises(ModelUnavailable):
        Store(root=tmp_path).policy(run_id)
