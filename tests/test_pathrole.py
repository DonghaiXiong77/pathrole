import numpy as np
import pandas as pd
import pytest

from pathrole import (Programme, RoleLibrary, adjusted_effects, builtin_library,
                      score_programmes, simulate_cohort)
from pathrole.singlecell import compartment_fractions, compartment_profile, rank_biserial


# ------------------------------------------------------------------ roles
def test_programme_rejects_too_few_genes():
    with pytest.raises(ValueError, match='at least 2 genes'):
        Programme('Pyroptosis', 'execution', ('GSDMD',))


def test_programme_deduplicates_and_strips():
    p = Programme('M', 'sensing', (' A ', 'B', 'A', ''))
    assert p.genes == ('A', 'B')


def test_library_rejects_duplicate_names():
    p = Programme('M', 'sensing', ('A', 'B'))
    with pytest.raises(ValueError, match='duplicate programme names'):
        RoleLibrary([p, p])


def test_paired_arms_finds_only_complete_pairs():
    lib = RoleLibrary([
        Programme('Pyroptosis', 'sensing', ('A', 'B')),
        Programme('Pyroptosis', 'execution', ('C', 'D')),
        Programme('Other', 'sensing', ('E', 'F')),
    ])
    pairs = lib.paired_arms()
    assert [m for m, _, _ in pairs] == ['Pyroptosis']


def test_tsv_roundtrip(tmp_path):
    lib = builtin_library()
    path = tmp_path / 'lib.tsv'
    lib.to_tsv(path)
    back = RoleLibrary.from_tsv(path)
    assert back.names == lib.names
    assert back['Necroptosis execution'].genes == lib['Necroptosis execution'].genes


def test_from_rows_reports_missing_column():
    with pytest.raises(ValueError, match='missing column'):
        RoleLibrary.from_rows([{'modality': 'M', 'role': 'sensing'}])


def test_coverage_counts_present_genes():
    lib = RoleLibrary([Programme('M', 'sensing', ('A', 'B', 'C'))])
    assert lib.coverage(['A', 'C', 'Z'])['M sensing'] == (2, 3)


# ---------------------------------------------------------------- scoring
def _toy_expr(seed=0, n_genes=12, n_samples=30):
    rng = np.random.default_rng(seed)
    genes = [f'G{i:02d}' for i in range(n_genes)]
    samples = [f'S{i:02d}' for i in range(n_samples)]
    return pd.DataFrame(rng.lognormal(3, 0.5, (n_genes, n_samples)),
                        index=genes, columns=samples)


def test_scores_are_centred_per_programme():
    expr = _toy_expr()
    lib = RoleLibrary([Programme('M', 'sensing', tuple(expr.index[:6]))])
    s = score_programmes(expr, lib)
    assert s.shape == (expr.shape[1], 1)
    assert abs(float(s['M sensing'].mean())) < 1e-9


def test_scoring_drops_small_programmes_with_warning():
    expr = _toy_expr()
    lib = RoleLibrary([Programme('M', 'sensing', tuple(expr.index[:6])),
                       Programme('M', 'execution', ('NOT_PRESENT1', 'NOT_PRESENT2'))])
    with pytest.warns(UserWarning, match='dropped'):
        s = score_programmes(expr, lib)
    assert list(s.columns) == ['M sensing']


def test_scoring_raises_when_nothing_scoreable():
    expr = _toy_expr()
    lib = RoleLibrary([Programme('M', 'sensing', ('X1', 'X2', 'X3'))])
    with pytest.raises(ValueError, match='no programme had enough genes'):
        score_programmes(expr, lib, on_missing='ignore')


def test_scoring_rejects_negative_input_when_logging():
    expr = _toy_expr()
    expr.iloc[0, 0] = -1.0
    lib = RoleLibrary([Programme('M', 'sensing', tuple(expr.index[:6]))])
    with pytest.raises(ValueError, match='negative values'):
        score_programmes(expr, lib)


def test_scoring_rejects_duplicate_genes():
    expr = _toy_expr()
    expr = pd.concat([expr, expr.iloc[[0]]])
    lib = RoleLibrary([Programme('M', 'sensing', tuple(expr.index[:6]))])
    with pytest.raises(ValueError, match='duplicate gene names'):
        score_programmes(expr, lib)


# ----------------------------------------------------------------- adjust
def _scores_groups(seed=0, n=60):
    rng = np.random.default_rng(seed)
    idx = [f'S{i:02d}' for i in range(n)]
    g = pd.Series(['A'] * (n // 2) + ['B'] * (n - n // 2), index=idx)
    s = pd.DataFrame({'p1': rng.normal(0, 1, n), 'p2': rng.normal(0, 1, n)},
                     index=idx)
    return s, g


def test_adjusted_effects_recovers_a_known_coefficient():
    s, g = _scores_groups()
    s['p1'] = (g == 'B').astype(float) * 2.0        # exact, noiseless effect
    res = adjusted_effects(s, g)
    assert res.group_levels == ('A', 'B')
    assert res.table.loc['p1', 'beta_unadjusted'] == pytest.approx(2.0, abs=1e-8)


def test_covariate_removes_a_pure_composition_effect():
    rng = np.random.default_rng(1)
    n = 120
    idx = [f'S{i:03d}' for i in range(n)]
    g = pd.Series(np.where(np.arange(n) % 2 == 0, 'A', 'B'), index=idx)
    # composition differs by group and fully determines the score
    comp = pd.Series(np.where(g == 'B', 0.6, 0.4) + rng.normal(0, 0.02, n), index=idx)
    s = pd.DataFrame({'p1': 3.0 * comp + rng.normal(0, 0.01, n)}, index=idx)
    res = adjusted_effects(s, g, comp)
    assert res.table.loc['p1', 'fdr_unadjusted'] < 1e-6      # naive sees it
    assert abs(res.table.loc['p1', 'beta_adjusted']) < 0.05   # adjusted does not
    assert res.table.loc['p1', 'pct_attenuation'] > 80


def test_attenuation_is_nan_for_a_near_zero_effect():
    s, g = _scores_groups()
    s['p1'] = 0.0
    s.loc[s.index[0], 'p1'] = 1e-6
    res = adjusted_effects(s, g)
    assert np.isnan(res.table.loc['p1', 'pct_attenuation'])


def test_requires_exactly_two_groups():
    s, g = _scores_groups()
    g.iloc[:5] = 'C'
    with pytest.raises(ValueError, match='exactly two groups'):
        adjusted_effects(s, g)


def test_rejects_tiny_groups():
    s, g = _scores_groups(n=20)
    g.iloc[:] = 'A'
    g.iloc[:2] = 'B'
    with pytest.raises(ValueError, match='at least 3 samples'):
        adjusted_effects(s, g)


def test_rejects_disjoint_indices():
    s, g = _scores_groups()
    g.index = [f'X{i}' for i in range(len(g))]
    with pytest.raises(ValueError, match='share no samples'):
        adjusted_effects(s, g)


def test_reference_choice_flips_the_sign():
    s, g = _scores_groups()
    s['p1'] = (g == 'B').astype(float) * 2.0
    a = adjusted_effects(s, g, reference='A').table.loc['p1', 'beta_unadjusted']
    b = adjusted_effects(s, g, reference='B').table.loc['p1', 'beta_unadjusted']
    assert a == pytest.approx(-b, abs=1e-8)


def test_fdr_is_monotone_in_p():
    s, g = _scores_groups(seed=3)
    res = adjusted_effects(s, g)
    t = res.table.sort_values('p_unadjusted')
    assert (t['fdr_unadjusted'].diff().dropna() >= -1e-12).all()


# ------------------------------------------------------------- singlecell
def test_rank_biserial_extremes():
    assert rank_biserial([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert rank_biserial([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)


def test_compartment_profile_flags_a_restricted_programme():
    rng = np.random.default_rng(2)
    n = 400
    idx = [f'C{i:04d}' for i in range(n)]
    lab = pd.Series(np.where(np.arange(n) < n // 2, 'myeloid', 'tumour'), index=idx)
    sc = pd.DataFrame({'restricted': np.where(lab == 'myeloid', 2.0, 0.0)
                       + rng.normal(0, 0.1, n),
                       'flat': rng.normal(0, 1, n)}, index=idx)
    prof = compartment_profile(sc, lab, focus='myeloid')
    assert prof.loc['restricted', 'rank_biserial_myeloid'] > 0.95
    assert abs(prof.loc['flat', 'rank_biserial_myeloid']) < 0.2
    assert prof.loc['n_cells', 'myeloid'] == n // 2


def test_compartment_profile_rejects_unknown_focus():
    idx = [f'C{i}' for i in range(50)]
    lab = pd.Series(['a'] * 50, index=idx)
    sc = pd.DataFrame({'p': np.zeros(50)}, index=idx)
    with pytest.raises(ValueError, match='focus compartment'):
        compartment_profile(sc, lab, focus='b')


def test_compartment_fractions_sum_to_one():
    idx = [f'C{i}' for i in range(60)]
    lab = pd.Series(['a', 'b', 'c'] * 20, index=idx)
    smp = pd.Series(['S1'] * 30 + ['S2'] * 30, index=idx)
    f = compartment_fractions(lab, smp)
    assert np.allclose(f.sum(axis=1), 1.0)


# ------------------------------------------------------------- simulation
def test_simulator_marks_null_and_real_programmes():
    sim = simulate_cohort(n_samples=80, n_null_programmes=3,
                          n_real_programmes=3, seed=5)
    assert len(sim.null_programmes) == 3
    assert len(sim.real_programmes) == 3
    assert sim.expression.shape[1] == 80
    assert (sim.expression.to_numpy() >= 0).all()


def test_confounding_creates_false_positives_that_adjustment_removes():
    sim = simulate_cohort(n_samples=240, frac_delta=0.16, seed=7)
    scores = score_programmes(sim.expression, sim.library)
    naive = adjusted_effects(scores, sim.groups, None)
    adj = adjusted_effects(scores, sim.groups, sim.fractions['compartment_1'])
    nulls = sim.null_programmes
    fp_naive = int((naive.table.loc[nulls, 'fdr_unadjusted'] < 0.05).sum())
    fp_adj = int((adj.table.loc[nulls, 'fdr_adjusted'] < 0.05).sum())
    assert fp_naive > 0, 'confounding should fool the naive analysis'
    assert fp_adj < fp_naive, 'adjustment should remove false positives'
    reals = sim.real_programmes
    assert int((adj.table.loc[reals, 'fdr_adjusted'] < 0.05).sum()) >= len(reals) - 1


def test_no_confounding_leaves_both_models_agreeing():
    sim = simulate_cohort(n_samples=200, frac_delta=0.0, seed=11)
    scores = score_programmes(sim.expression, sim.library)
    naive = adjusted_effects(scores, sim.groups, None)
    adj = adjusted_effects(scores, sim.groups, sim.fractions['compartment_1'])
    nulls = sim.null_programmes
    assert int((naive.table.loc[nulls, 'fdr_unadjusted'] < 0.05).sum()) <= 1
    assert int((adj.table.loc[nulls, 'fdr_adjusted'] < 0.05).sum()) <= 1


def test_simulator_is_reproducible():
    a = simulate_cohort(n_samples=40, seed=3).expression
    b = simulate_cohort(n_samples=40, seed=3).expression
    pd.testing.assert_frame_equal(a, b)
