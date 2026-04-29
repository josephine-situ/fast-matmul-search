"""Tests for experiment selection."""

from omega_analysis import select_experiments


def test_select_experiments_does_not_require_precomputed_score(monkeypatch):
	targets = [
		{
			'case': (2, 2, 2),
			'tensor_entries': 64,
			'best_rank': 7,
			'marginal_omega': 0.25,
			'limited_prior_work': True,
		},
		{
			'case': (2, 3, 3),
			'tensor_entries': 216,
			'best_rank': 11,
			'marginal_omega': 0.10,
			'limited_prior_work': False,
		},
	]

	monkeypatch.setattr('omega_analysis.get_sorted_targets', lambda: targets)

	experiments = select_experiments(max_tensor_entries=5000, n_targets=2)

	assert len(experiments) == 4
	assert all('priority' in exp for exp in experiments)
	assert {exp['purpose'] for exp in experiments} == {'validate', 'improve'}
