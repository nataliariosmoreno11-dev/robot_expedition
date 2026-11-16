from pipelines.train import prune_intermediate_checkpoints
import numpy as np
import pytest
from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum


def test_focused_rehearsal_keeps_current_mission_and_prior_levels():
    manager = CampaignCurriculum(initial_level=6, rehearsal_probability=.6, rehearsal_weights={3: 4})
    rng = np.random.default_rng(42)
    counts = np.bincount([manager.select_level(rng) for _ in range(10000)], minlength=7)
    assert .37 < counts[6] / 10000 < .43
    assert .27 < counts[3] / 10000 < .33
    assert all(counts[level] > 500 for level in (1, 2, 4, 5))
    assert not manager.mastered  # Sampling alone cannot approve scenarios.


@pytest.mark.parametrize('weight', [0, -1, float('inf'), float('nan')])
def test_invalid_rehearsal_weights_rejected(weight):
    with pytest.raises(ValueError):
        CampaignCurriculum(rehearsal_weights={3: weight})


def test_pruning_keeps_recovery_and_mastery_weights(tmp_path):
    for name in ('checkpoint-10', 'checkpoint-20', 'checkpoint-30',
                 'retention-safe', 'mastered-scenario-06-20', 'rejected-25'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'ppo_model.zip').write_bytes(b'weights')
    (tmp_path / 'ppo_model.zip').write_bytes(b'root weights')
    prune_intermediate_checkpoints(tmp_path, 'checkpoint', 2)
    assert not (tmp_path / 'checkpoint-10').exists()
    for name in ('checkpoint-20', 'checkpoint-30', 'retention-safe',
                 'mastered-scenario-06-20', 'rejected-25'):
        assert (tmp_path / name / 'ppo_model.zip').read_bytes() == b'weights'
    assert (tmp_path / 'ppo_model.zip').read_bytes() == b'root weights'


def test_pruning_never_follows_directory_symlinks(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    external = tmp_path / 'other'
    external.mkdir()
    (external / 'important').write_bytes(b'keep')
    (models / 'checkpoint-1').symlink_to(external, target_is_directory=True)
    (models / 'checkpoint-2').mkdir()
    (models / 'checkpoint-3').mkdir()
    prune_intermediate_checkpoints(models, 'checkpoint', 1)
    assert (external / 'important').read_bytes() == b'keep'
    assert (models / 'checkpoint-1').is_symlink()
    assert (models / 'checkpoint-3').is_dir()
