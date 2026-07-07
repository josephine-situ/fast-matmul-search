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
			'case': (3, 3, 3),
			'tensor_entries': 729,
			'best_rank': 23,
			'marginal_omega': 0.10,
			'limited_prior_work': False,
		},
	]

	monkeypatch.setattr('omega_analysis.get_sorted_targets', lambda: targets)

	experiments = select_experiments(max_tensor_entries=5000, n_targets=2)

	# (2,2,2): rank 7 equals the proven lower bound, so only 'validate' is
	# generated (rank 6 would contradict the bound and is filtered out).
	# (3,3,3): rank 23 > lower bound 19, so both 'validate' and 'improve'.
	assert len(experiments) == 3
	assert all('priority' in exp for exp in experiments)
	assert {exp['purpose'] for exp in experiments} == {'validate', 'improve'}
