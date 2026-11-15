from scripts.training_notifications import collect, promotions, tick, write


def test_promotion_queues_once_and_preserves_recipient_delivery():
    state = {'seen': ['1'], 'events': {}}
    found = {'1': {'level': 1}, '2': {'level': 2}}
    collect(state, found, ['a@example.org', 'b@example.org'])
    state['events']['2']['pending'].remove('a@example.org')
    state['events']['2']['sent'].append('a@example.org')
    collect(state, {}, ['a@example.org', 'b@example.org'])
    collect(state, found, ['a@example.org', 'b@example.org'])
    assert list(state['events']) == ['2']
    assert state['events']['2']['pending'] == ['b@example.org']


def test_baseline_and_offline_queue(tmp_path):
    directory = tmp_path / 'logs/training_notifications'
    directory.mkdir(parents=True)
    write(directory / 'config.json', {'recipients': ['a@example.org']})
    checkpoint = tmp_path / 'models/run/mastered-scenario-01-100'
    checkpoint.mkdir(parents=True)
    write(checkpoint / 'curriculum_state.json', {'mastered': [1]})
    assert tick(tmp_path)['events'] == {}
    second = checkpoint.parent / 'mastered-scenario-02-200'
    second.mkdir()
    assert list(promotions(tmp_path)) == ['1']  # Incomplete checkpoint ignored.
    write(second / 'curriculum_state.json', {'mastered': [1, 2]})
    state = tick(tmp_path)
    assert state['status'] == 'waiting_for_smtp'
    assert state['events']['2']['pending'] == ['a@example.org']
