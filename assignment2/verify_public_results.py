"""Verify the public evidence subset without torch, model weights, or rollouts.

Run from any directory: python -m assignment2.verify_public_results
This command reads original CSV/JSON and the publication manifest; it writes nothing.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'assignment2/results'


def records(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def close(actual, expected, name):
    if not math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-9):
        raise AssertionError(f'{name}: calculated {actual}, recorded {expected}')


def evaluation(folder, csv_name, summary_name, expected_count):
    data = records(folder / csv_name)
    summary = read(folder / summary_name)
    assert len(data) == expected_count == summary['episodes']
    assert len({row['seed'] for row in data}) == expected_count
    rewards = [float(row['reward']) for row in data]
    close(statistics.mean(rewards), summary['mean'], str(folder) + ' mean')
    close(statistics.pstdev(rewards), summary['std_population'], str(folder) + ' SD')
    for key, actual in [('median', statistics.median(rewards)), ('minimum', min(rewards)), ('maximum', max(rewards))]:
        if key in summary:close(actual, summary[key], key)
    assert all(row['terminated'].lower() == 'true' or row['truncated'].lower() == 'true' for row in data)
    if 'perfect_episodes' in summary:
        assert sum(reward == 500 for reward in rewards) == summary['perfect_episodes']
        assert all(float(row['reward']) == int(row['length']) for row in data)
    return summary


def main():
    manifest = read(ROOT / 'PUBLICATION_MANIFEST.json')
    for item in manifest['evidence_files']:
        path = ROOT / item['path']
        assert path.is_file(), f'Missing evidence: {item["path"]}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Changed evidence: {item["path"]}'

    checked = {}
    for game, comparison in [('cartpole', 'comparison.csv'), ('mspacman', 'validation/comparison.csv')]:
        rows = records(RESULTS / game / comparison)
        for row in rows:
            run = RESULTS / game / ('validation' if game == 'mspacman' else '') / row['run']
            config = read(run / 'metrics/config.json')
            assert config['env_id'] == ('CartPole-v1' if game == 'cartpole' else 'MsPacman-v5')
            if game == 'cartpole':
                result = evaluation(run, 'evaluation500/episodes.csv', 'evaluation500/summary.json', 500)
            else:
                assert config['gamma'] == .99
                result = evaluation(run, 'metrics/diagnostic_evaluation30.csv', 'metrics/diagnostic_evaluation30.json', 30)
            close(result['mean'], row['evaluation_mean'], row['run'])
            close(result['std_population'], row['evaluation_std'], row['run'])
            training = records(run / 'logs/episodes.csv')
            assert len(training) == int(row['episodes'])
            key = 'env_steps' if game == 'cartpole' else 'steps'
            assert int(training[-1][key]) == int(row[key])
        checked[game + '_comparison_runs'] = len(rows)

    cp = RESULTS / 'cartpole/seed33'
    ms = RESULTS / 'mspacman/main_seed41'
    cp_eval = evaluation(cp, 'evaluation500/episodes.csv', 'evaluation500/summary.json', 500)
    ms_eval = evaluation(ms, 'evaluation500/episodes.csv', 'evaluation500/summary.json', 500)
    assert read(cp / 'metrics/config.json')['gamma'] == .95
    assert read(ms / 'metrics/config.json')['gamma'] == .99
    for base, expected in [(cp, (942, 80014)), (ms, (3114, 2000528))]:
        train = records(base / 'logs/episodes.csv')
        summary = read(base / 'metrics/training_summary.json')
        step_key = 'env_steps' if base == cp else 'steps'
        assert len(train) == expected[0] == summary['episodes']
        assert int(train[-1][step_key]) == expected[1] == summary[step_key]
        if base == ms:
            scores = [float(row['reward']) for row in train]
            close(statistics.mean(scores[:100]), 272.1, 'MsPacman first100')
            close(statistics.mean(scores[-100:]), 994.2, 'MsPacman final100')
            assert int(train[-1]['total_raw_frames']) == 7992770
    stress = records(cp / 'robustness/episodes.csv')
    assert len(stress) == 405 and sum(float(r['reward']) == 500 for r in stress) == 293
    for base, tolerance in [(cp, 2e-5), (ms, 1e-3)]:
        bellman = records(base / 'diagnostics/bellman_reference.csv')
        error = max(float(row['absolute_error']) for row in bellman)
        assert error < tolerance
        close(error, read(base / 'diagnostics/summary.json')['max_bellman_absolute_error'], 'Bellman reference CSV')
    print(json.dumps(dict(status='PASS', **checked,
        cartpole_mean=cp_eval['mean'], cartpole_std_population=cp_eval['std_population'],
        cartpole_perfect=cp_eval['perfect_episodes'], mspacman_mean=ms_eval['mean'],
        mspacman_std_population=ms_eval['std_population'],
        evidence_hashes_verified=len(manifest['evidence_files']),
        scope='Saved public data only; no inference, training, or historical-target reconstruction'), indent=2))


if __name__ == '__main__':
    main()
